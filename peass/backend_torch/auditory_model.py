"""
PEASS PyTorch Auditory Nerve Model
File path: peass/backend_torch/auditory_model.py
"""
import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from .gammatone import GammatoneAnalyzerTorch
from .utils import fast_resample_poly_torch
from .utils import next_fast_fft_length
from ..config import ModulationProcessingType


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


@torch.jit.script
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


@torch.jit.script
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


@torch.jit.script
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


def simulate_auditory_nerve_adaptation(subbands: torch.Tensor, fs: float) -> torch.Tensor:
    abs_thresh, gains, thresholds = _get_adaptation_constants(fs, str(subbands.device), subbands.dtype)

    orig_shape = subbands.shape
    T = orig_shape[-1]

    flat = subbands.reshape(-1, T)
    if torch.is_grad_enabled() and flat.requires_grad:
        adapted = _raw_adaptation_loop(flat, thresholds, gains, abs_thresh)
    else:
        adapted = _adaptation_loop_forward(flat, thresholds, gains, abs_thresh)
    adapted = adapted.view(*orig_shape)

    final_thresh = abs_thresh ** (0.5 ** 5)
    return (100.0 / (1.0 - final_thresh)) * (adapted - final_thresh)


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
    subbands = analyzer.process(scaled).real

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
