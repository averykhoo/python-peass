"""
PEASS PyTorch Auditory Nerve Model
File path: peass/backend_torch/auditory_model.py
"""
import math
import warnings
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from .gammatone import GammatoneAnalyzerTorch
from .utils import fast_resample_poly_torch
from .utils import next_fast_fft_length
from ..config import ModulationProcessingType

# -----------------------------------------------------------------------------
# NUMBA JIT COMPILATION (WITH SAFE IMPORT FALLBACK)
# -----------------------------------------------------------------------------
# Mirrors the numpy backend's idiom: numba is an optional extra, so the kernel is
# defined only if it imports, and every call site keeps a working torch fallback.
try:
    import numba

    _HAS_NUMBA = True


    @numba.njit(cache=True, fastmath=False)
    def _numba_adaptation_loop(frames: np.ndarray, thresholds: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """Native transcription of ``_adaptation_loop_forward``'s inner loop.

        ``frames`` is the already-clamped, already-transposed (num_steps, num_rows)
        view, so this kernel owns nothing but the recurrence itself. The operation
        order is deliberately identical to the torch version -- running product,
        divide, then ``c*(1-g) + s*g`` as two multiplies and an add -- and
        ``fastmath=False`` keeps LLVM from reassociating it.

        That buys exact bit-equality with the torch loop on the reference platform,
        but do not read it as a portable guarantee: whether a toolchain contracts
        ``a*b + c`` into an FMA varies by LLVM and torch build, and CPython 3.14 was
        measured 1.8e-14 from this kernel where 3.10 was exactly equal. The agreement
        that *is* portable is ~1e-14 relative, which is still fourteen orders below
        any real transcription error -- see the tolerance note in
        ``tests/unit/backend_torch/test_torch_auditory_model.py``.

        The win is not arithmetic, it is dispatch: the torch loop pays ~7 kernel
        launches per timestep on tensors holding one element per band, so it is
        latency-bound on the dispatcher rather than on FLOPs.
        """
        num_steps, num_rows = frames.shape
        state = np.empty((5, num_rows), dtype=frames.dtype)
        for stage_idx in range(5):
            for row_idx in range(num_rows):
                state[stage_idx, row_idx] = thresholds[stage_idx]

        input_weight = 1.0 - gains

        outputs = np.empty((num_steps, num_rows), dtype=frames.dtype)
        for step_idx in range(num_steps):
            for row_idx in range(num_rows):
                floored = frames[step_idx, row_idx]
                # Running product over the *previous* step's state. Each stage is
                # folded in before that stage is overwritten, which is what makes
                # the in-place update equivalent to torch's cumprod-then-update.
                running_product = 1.0
                compressed = 0.0
                for stage_idx in range(5):
                    running_product *= state[stage_idx, row_idx]
                    compressed = floored / running_product
                    updated = compressed * input_weight[stage_idx] + state[stage_idx, row_idx] * gains[stage_idx]
                    state[stage_idx, row_idx] = max(updated, thresholds[stage_idx])
                outputs[step_idx, row_idx] = compressed

        return outputs


    @numba.njit(cache=True, fastmath=False)
    def _numba_fused_haircell_adaptation(
            subbands: np.ndarray,
            haircell_gain: float,
            thresholds: np.ndarray,
            gains: np.ndarray,
            abs_thresh: float,
            final_thresh: float,
            output_scale: float,
    ) -> np.ndarray:
        """Rectify, transduce, floor, adapt and rescale in one row-major pass.

        This is the whole of ``simulate_inner_haircell_transduction`` followed by
        ``simulate_auditory_nerve_adaptation``, for one ``(rows, num_samples)``
        block, and it is a *fast path* -- the two torch functions remain the
        definition and still serve grad / non-CPU / no-Numba callers. Note that
        float32 callers are NOT among them: the gammatone analyzer promotes before
        this is reached, so a float32 caller of the metric runs this kernel too. See
        ``_can_fuse_haircell_adaptation``.

        Two pieces of arithmetic differ from what those two functions do, and both
        are deliberate:

        * The haircell lowpass is evaluated as the exact one-pole recurrence
          ``y[n] = g*y[n-1] + (1-g)*x[n]`` rather than as its impulse response
          truncated at 10 ms and convolved by FFT. Those are the same filter to
          ``g**(0.01*fs) == exp(-20*pi) == 5.2e-28`` relative -- the exponent is
          ``fs``-independent, so the truncation is twelve orders below float64
          eps at every rate. Measured against a double-double (~106-bit) oracle on
          the reference clip's real subbands, the recurrence is the *more* accurate
          of the two: 1.96e-16 worst absolute error against the FFT path's 4.31e-16
          (2.1e-16 vs 4.6e-16 of the signal's own peak). See
          ``tests/unit/backend_torch/test_torch_auditory_model.py`` for the pinned
          comparison. Note ``np.longdouble`` is float64 on Windows/MSVC, so an
          80-bit oracle is not available on the reference platform and a
          longdouble "oracle" here silently measures nothing.
        * The five adaptation stages are transcribed exactly as in
          ``_numba_adaptation_loop`` -- running product over the *previous* step's
          state, one divide, then ``c*(1-g) + s*g`` as two multiplies and an add,
          under ``fastmath=False``. Same order, same result.

        The final affine ``(100/(1 - final_thresh)) * (adapted - final_thresh)`` is
        folded in here as ``output_scale`` so it costs no extra pass.

        The win is memory traffic, not arithmetic: the unfused path materialises
        the relu, an rfft spectrum, an irfft output, a clamp, two full transposes
        and two more full-size buffers for the affine. This writes one buffer.
        """
        num_rows, num_samples = subbands.shape
        outputs = np.empty((num_rows, num_samples), dtype=subbands.dtype)

        haircell_weight = 1.0 - haircell_gain
        input_weight = 1.0 - gains

        state = np.empty(5, dtype=subbands.dtype)
        for row_idx in range(num_rows):
            haircell_state = 0.0
            for stage_idx in range(5):
                state[stage_idx] = thresholds[stage_idx]

            for sample_idx in range(num_samples):
                # 1. Half-wave rectification (F.relu).
                value = subbands[row_idx, sample_idx]
                if value < 0.0:
                    value = 0.0

                # 2. Haircell 1 kHz lowpass, as the exact causal one-pole IIR.
                haircell_state = haircell_gain * haircell_state + haircell_weight * value
                floored = haircell_state

                # 3. Absolute-hearing-threshold floor (torch.clamp).
                if floored < abs_thresh:
                    floored = abs_thresh

                # 4. Five-stage adaptation cascade, folded over the running product.
                running_product = 1.0
                compressed = 0.0
                for stage_idx in range(5):
                    running_product *= state[stage_idx]
                    compressed = floored / running_product
                    updated = compressed * input_weight[stage_idx] + state[stage_idx] * gains[stage_idx]
                    state[stage_idx] = max(updated, thresholds[stage_idx])

                # 5. Global dB offset scaling.
                outputs[row_idx, sample_idx] = output_scale * (compressed - final_thresh)

        return outputs

except ImportError:
    _HAS_NUMBA = False


@lru_cache(maxsize=4)
def _get_adaptation_constants(fs: float, device_str: str, dtype: torch.dtype):
    device = torch.device(device_str)
    abs_thresh = 10.0 ** (-100.0 / 20.0)
    bws = 1.0 / (math.pi * torch.tensor([0.005, 0.05, 0.129, 0.253, 0.5], device=device, dtype=dtype))
    gains = torch.exp(-math.pi * bws / fs)
    thresholds = torch.tensor([abs_thresh ** (0.5 ** i) for i in range(1, 6)], device=device, dtype=dtype)
    return abs_thresh, gains, thresholds


@lru_cache(maxsize=4)
def _get_modulation_constants(mod_type: str, device_str: str, dtype: torch.dtype):
    device = torch.device(device_str)
    if mod_type == "LOWPASS":
        centers = torch.tensor([0.0], device=device, dtype=dtype)
        bandwidths = torch.tensor([15.92], device=device, dtype=dtype)
    else:
        centers = torch.cat([
            torch.tensor([0.0, 5.0], device=device, dtype=dtype),
            10.0 * (5.0 / 3.0) ** torch.arange(6, device=device, dtype=dtype)
        ])
        bandwidths = torch.cat([
            torch.tensor([5.0, 5.0], device=device, dtype=dtype),
            5.0 * (5.0 / 3.0) ** torch.arange(6, device=device, dtype=dtype)
        ])
    return centers, bandwidths


def simulate_inner_haircell_transduction(subbands: torch.Tensor, fs: float) -> torch.Tensor:
    """Models the nonlinear mechanical-to-neural transduction of the inner hair cells."""
    # Half-wave Rectification
    rectified = F.relu(subbands)

    # 1 kHz Lowpass via causal FIR approximation for fast parallel execution
    gain = math.exp(-math.pi * 2000.0 / fs)
    b0 = 1.0 - gain

    # Pre-compute the causal IIR impulse response out to 60 dB decay (10 ms).
    decay_samples = int(fs * 0.01)
    ir = b0 * (gain ** torch.arange(decay_samples, device=subbands.device, dtype=subbands.dtype))

    orig_shape = rectified.shape
    T = orig_shape[-1]
    rect_flat = rectified.view(-1, T)

    # Causal FIR filtering via FFT linear convolution. This is the true
    # convolution y[n] = sum_k x[n-k]*ir[k] (no kernel flip needed, unlike the
    # cross-correlation F.conv1d), matching the NumPy lfilter([b0], [1, -gain]).
    # FFT convolution is O(B*L) memory vs conv1d's O(B*K*L) im2col, which would
    # otherwise blow up (tens of GB) on realistic-length, many-band batches.
    # conv_length is an arbitrary integer and lands on slow prime-factor transform
    # sizes, so pad up to the next 5-smooth length (13x on the measured shapes).
    conv_length = T + decay_samples - 1
    fft_length = next_fast_fft_length(conv_length)
    spectrum = torch.fft.rfft(rect_flat, n=fft_length, dim=-1) * torch.fft.rfft(ir, n=fft_length)
    transduced = torch.fft.irfft(spectrum, n=fft_length, dim=-1)[:, :T]

    return transduced.view(*orig_shape)


def _straight_through_max(value: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    """max(value, floor) in the forward pass, softplus gradient in the backward.

    The exact max makes the forward values match the NumPy reference bit-for-bit;
    the smooth softplus surrogate supplies the gradient so the metric stays
    differentiable. (A plain softplus max was the dominant source of torch-vs-numpy
    divergence: it drifts through the 5-stage feedback cascade, dropping auditory
    representation correlation to ~0.9; this recovers ~1.0.)

    ``hard + (soft - soft.detach())`` rather than ``soft + (hard - soft).detach()``:
    the two are algebraically equal but only this one is bit-exact in the forward
    pass, because ``soft - soft.detach()`` is exactly 0.0 in floating point whereas
    ``soft + (hard - soft)`` re-rounds. That rounding compounded through the 12000+
    step recurrence into a ~2.5e-8 drift away from the NumPy reference.
    """
    soft = torch.nn.functional.softplus(1000.0 * (value - floor)) / 1000.0 + floor
    hard = torch.maximum(value, floor)
    return hard.detach() + (soft - soft.detach())


def _raw_adaptation_loop(subbands_flat: torch.Tensor, thresholds: torch.Tensor, gains: torch.Tensor,
                         abs_thresh: float) -> torch.Tensor:
    """Differentiable 5-stage adaptation recurrence (straight-through max).

    See ``_adaptation_loop_forward`` for the vectorisation; this variant swaps the
    hard ``maximum`` for the straight-through estimator so the cascade stays
    differentiable, and accumulates into a list because autograd retains every
    intermediate anyway (a preallocated buffer would only add graph nodes).
    """
    num_rows, num_steps = subbands_flat.shape

    state = thresholds.view(5, 1).repeat(1, num_rows)
    floors = thresholds.view(5, 1)
    input_weight = (1.0 - gains).view(5, 1)
    state_weight = gains.view(5, 1)

    hearing_floor = torch.full((1, 1), abs_thresh, dtype=subbands_flat.dtype, device=subbands_flat.device)
    frames = _straight_through_max(subbands_flat, hearing_floor).t().contiguous().unbind(0)

    outputs: list[torch.Tensor] = []
    for t in range(num_steps):
        compressed = frames[t] / torch.cumprod(state, dim=0)
        state = _straight_through_max(torch.addcmul(compressed * input_weight, state, state_weight), floors)
        outputs.append(compressed[4])

    return torch.stack(outputs, dim=1)


def _adaptation_loop_forward(subbands_flat: torch.Tensor, thresholds: torch.Tensor, gains: torch.Tensor,
                             abs_thresh: float) -> torch.Tensor:
    """Gradient-free 5-stage adaptation recurrence.

    The recurrence is an inherently sequential nonlinear cascade (each timestep
    divides by the previous state), so the time axis cannot be parallelized. What
    *can* be collapsed is the cascade itself: every division at step t reads the
    stage states from step t-1, so

        compressed[i] = floored[t] / (state[0] * ... * state[i])[t-1]

    which is one ``cumprod`` plus one broadcast divide for all five stages at once,
    instead of five interleaved divide/update pairs. The five EMA updates then also
    fold into a single vectorised expression over the (5, rows) state.

    That takes the hot loop from ~75 tensor dispatches per timestep to 7. Since the
    per-step tensors are tiny (one element per band), the loop is dispatch-bound
    rather than FLOP-bound, so the op count is what sets the wall clock -- measured
    ~8.5x on CPU, and the same reduction in kernel launches applies on GPU.

    Values are bit-identical to the straight-through variant's forward pass; only
    the backward differs, which is why this path is taken solely when no gradient
    is required.
    """
    num_rows, num_steps = subbands_flat.shape

    state = thresholds.view(5, 1).repeat(1, num_rows)
    floors = thresholds.view(5, 1)
    input_weight = (1.0 - gains).view(5, 1)
    state_weight = gains.view(5, 1)

    # Hoisted out of the loop: the absolute-hearing-threshold floor is elementwise.
    frames = torch.clamp(subbands_flat, min=abs_thresh).t().contiguous().unbind(0)

    # addcmul folds the second multiply into the add without changing the result;
    # torch.lerp would drop one more dispatch but re-associates the EMA and costs
    # ~25x accuracy against the NumPy reference for ~6% of end-to-end runtime.
    outputs = torch.empty((num_steps, num_rows), dtype=subbands_flat.dtype, device=subbands_flat.device)
    for t in range(num_steps):
        compressed = frames[t] / torch.cumprod(state, dim=0)
        state = torch.maximum(torch.addcmul(compressed * input_weight, state, state_weight), floors)
        outputs[t] = compressed[4]

    return outputs.t().contiguous()


# TorchScript is deprecated in favour of torch.compile/torch.export, but neither is
# usable here: inductor's CPU backend needs an MSVC/GCC toolchain that is not a
# reasonable install requirement, and torch.export fully unrolls this loop at ~9
# graph nodes per timestep (~437k nodes for 2s of audio, re-exported per input
# length). Scripting is still worth ~2x on this function, so it stays until a
# replacement actually exists. The two other loops in this module were scripted for
# ~1.05x -- inside noise -- and have been dropped rather than carry the warning.
#
# Applied as a call rather than a decorator purely so the deprecation filter can be
# scoped to this one site instead of silencing the category process-wide.
#
# Guarded because torch documents jit.script as unsupported on Python 3.14+ ("may
# break"), and this runs at import time -- an unguarded failure here would take the
# whole backend down rather than costing some speed. The plain Python function is a
# correct, if slower, stand-in, and on the common CPU/float64 path the Numba kernel
# above means this fallback is not even on the hot path.
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*", category=DeprecationWarning)
        _adaptation_loop_forward = torch.jit.script(_adaptation_loop_forward)
except Exception:  # pragma: no cover - depends on the interpreter/torch combination
    pass


def simulate_auditory_nerve_adaptation(subbands: torch.Tensor, fs: float) -> torch.Tensor:
    abs_thresh, gains, thresholds = _get_adaptation_constants(fs, str(subbands.device), subbands.dtype)

    orig_shape = subbands.shape
    T = orig_shape[-1]

    flat = subbands.reshape(-1, T)
    if torch.is_grad_enabled() and flat.requires_grad:
        adapted = _raw_adaptation_loop(flat, thresholds, gains, abs_thresh)
    elif _HAS_NUMBA and flat.device.type == "cpu" and flat.dtype == torch.float64:
        # The recurrence is sequential in time and tiny per step (one element per
        # band), so the torch path is bound by kernel-launch latency, not by work.
        # A native kernel removes the dispatch entirely; the clamp and transpose stay
        # in torch so the kernel receives exactly the buffer the scripted loop would.
        frames = torch.clamp(flat, min=abs_thresh).t().contiguous()
        adapted = torch.from_numpy(_numba_adaptation_loop(
            frames.numpy(),
            thresholds.numpy(),
            gains.numpy(),
        )).t().contiguous()
    else:
        adapted = _adaptation_loop_forward(flat, thresholds, gains, abs_thresh)
    adapted = adapted.view(*orig_shape)

    final_thresh = abs_thresh ** (0.5 ** 5)
    return (100.0 / (1.0 - final_thresh)) * (adapted - final_thresh)


def _can_fuse_haircell_adaptation(subbands: torch.Tensor) -> bool:
    """Gate for the fused Numba fast path: CPU, float64, no gradient, Numba present.

    Deliberately the same four conditions as the adaptation-only Numba branch in
    ``simulate_auditory_nerve_adaptation``, since this is the same kind of path --
    forward-only, native, optional-dependency -- just covering two stages instead
    of one. Everything it rejects falls through to the torch functions unchanged.

    The ``float64`` condition is **not** a float32 opt-out, and must not be read as
    one. ``GammatoneAnalyzerTorch`` promotes unconditionally -- ``process`` casts its
    rows to ``complex128`` and ``process_real`` to ``float64`` -- so the subbands the
    only production call site passes here are float64 whatever dtype the caller
    handed the metric. A float32 caller of ``calculate_auditory_quality_features``
    therefore *does* run this kernel and does move by the haircell roundoff below
    (measured 2.4e-12, 2.3e-15 of peak, on the internal representation; <=4.4e-16
    absolute on the four features). The condition is live only for a hypothetical
    caller reaching these internals directly, and is kept because the kernel really
    would be wrong on a float32 buffer -- not because anything reaches it that way.
    ``tests/unit/backend_torch/test_torch_auditory_model.py`` pins both readings.
    """
    return (
            _HAS_NUMBA
            and subbands.device.type == "cpu"
            and subbands.dtype == torch.float64
            and not (torch.is_grad_enabled() and subbands.requires_grad)
    )


def _fused_haircell_adaptation(subbands: torch.Tensor, fs: float) -> torch.Tensor:
    """``simulate_auditory_nerve_adaptation(simulate_inner_haircell_transduction(...))``.

    The constants are taken from the same cache the unfused path uses, so the
    kernel sees exactly the thresholds and gains the torch loop would have seen;
    only the haircell filter's *evaluation* differs (see the kernel's docstring).

    ``subbands`` arrives from ``GammatoneAnalyzerTorch.process_real`` as a contiguous
    real block, so ``reshape`` and ``.numpy()`` are both views and the kernel reads
    unit-stride rows. It used to arrive as the stride-2 ``.real`` view of the
    analyzer's complex output, and was handed over strided on purpose: the
    contiguising copy would have been another full-size buffer, and `ARCHIVE.md`
    records the same experiment on the NumPy backend's fused kernel at 1.03x -- that
    loop is latency-bound on its five serial divisions, not on load bandwidth. The
    real-output analysis path made the block contiguous for free, as a side effect of
    not building the complex one; nothing here had to change for it, and the kernel
    still accepts either layout.
    """
    abs_thresh, gains, thresholds = _get_adaptation_constants(fs, str(subbands.device), subbands.dtype)
    haircell_gain = math.exp(-math.pi * 2000.0 / fs)
    final_thresh = abs_thresh ** (0.5 ** 5)

    orig_shape = subbands.shape
    flat = subbands.reshape(-1, orig_shape[-1])

    adapted = _numba_fused_haircell_adaptation(
        flat.numpy(),
        haircell_gain,
        thresholds.numpy(),
        gains.numpy(),
        abs_thresh,
        final_thresh,
        100.0 / (1.0 - final_thresh),
    )
    return torch.from_numpy(adapted).view(*orig_shape)


def generate_auditory_internal_representation_torch(
        signal_data: torch.Tensor, fs: float,
        modulation_type: ModulationProcessingType = ModulationProcessingType.LOWPASS
) -> tuple[torch.Tensor, float]:
    is_1d = signal_data.dim() == 1
    if is_1d:
        signal_data = signal_data.unsqueeze(0)

    scaled = 10.0 * signal_data
    low_freq = 235.0
    high_freq = min(0.5 * fs, 14500.0)

    original_fs = fs
    if fs / 2.0 < 1.5 * high_freq:
        new_fs = int(round(1.5 * fs))
        scaled = fast_resample_poly_torch(scaled, new_fs, int(fs), axis=-1)
        fs = float(new_fs)

    analyzer = GammatoneAnalyzerTorch(fs, low_freq, 1000.0, high_freq, 1.0, signal_data.device, signal_data.dtype)
    # Real-output analysis path: this stage discards the imaginary part, so the
    # subbands come back as a contiguous real block from a half-length inverse
    # transform instead of `.real` on a complex one. See `process_real`.
    subbands = analyzer.process_real(scaled)

    if _can_fuse_haircell_adaptation(subbands):
        adapted = _fused_haircell_adaptation(subbands, fs)
    else:
        transduced = simulate_inner_haircell_transduction(subbands, fs)
        adapted = simulate_auditory_nerve_adaptation(transduced, fs)

    mod_str = "LOWPASS" if modulation_type == ModulationProcessingType.LOWPASS else "FILTERBANK"
    centers, bandwidths = _get_modulation_constants(mod_str, str(signal_data.device), signal_data.dtype)

    target_fs = 100.0 if mod_str == "LOWPASS" else 800.0
    down_factor = 100 if mod_str == "LOWPASS" else 800
    downsampled = fast_resample_poly_torch(adapted, down_factor, int(fs), axis=-1)

    if downsampled.dim() == 2:
        downsampled = downsampled.unsqueeze(0)

    B, num_bands, num_samples = downsampled.shape
    num_mods = len(centers)

    internal_representation = torch.zeros((B, num_bands, num_samples, num_mods), dtype=torch.complex128,
                                          device=signal_data.device)

    decay_samples = int(target_fs * 0.5)
    time_indices = torch.arange(decay_samples, device=signal_data.device, dtype=signal_data.dtype)

    N_fft = 2 ** math.ceil(math.log2(num_samples + decay_samples))
    X = torch.fft.fft(downsampled.to(torch.complex128), n=N_fft, dim=-1)

    for m_idx in range(num_mods):
        g = math.exp(-math.pi * bandwidths[m_idx].item() / target_fs)
        b0 = 1.0 - g
        a1 = g * torch.exp(2j * math.pi * centers[m_idx] / target_fs)
        ir = b0 * (a1 ** time_indices)
        H = torch.fft.fft(ir, n=N_fft)
        y = torch.fft.ifft(X * H.unsqueeze(0).unsqueeze(0), n=N_fft, dim=-1)
        internal_representation[..., m_idx] = y[..., :num_samples]

    # Out-of-place assignment to preserve gradients
    channels_to_magnitude = (centers > 10.0).view(1, 1, 1, -1)
    internal_representation = torch.where(
        channels_to_magnitude,
        torch.abs(internal_representation).to(torch.complex128),
        internal_representation.real.to(torch.complex128)
    )

    out = internal_representation.real
    if is_1d:
        out = out.squeeze(0)
    return out, target_fs
