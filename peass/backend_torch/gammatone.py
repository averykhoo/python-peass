"""
PEASS PyTorch Gammatone Filterbank
File path: peass/backend_torch/gammatone.py

Uses mass-parallel FFTs in place of sequential recursive time loops.
"""
import math
from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def _get_gammatone_H_torch(N_fft: int, fs: float, cfs_tuple: tuple, norms_tuple: tuple, coefs_tuple: tuple,
                           device_str: str):
    device = torch.device(device_str)
    # `dtype=` is load-bearing and must not be dropped: `fftfreq` otherwise inherits
    # torch's *default* dtype, which is float32 unless the process changed it. The grid
    # values themselves are exact either way (1/N is a power of two, so both dtypes
    # return the same numbers) -- the loss is in the line below, where a Python complex
    # scalar times a float32 tensor gives complex64, rounding pi and running `exp` in
    # single precision. That is ~2e-7 on `z_inv`, and the 4th power of a near-resonant
    # denominator amplifies it ~100x: measured 1.4e-5 of peak on H, 1.0e-6 of peak on
    # the subbands, and 6.1e-5 of peak on the quietest decomposition component.
    freqs_norm = torch.fft.fftfreq(N_fft, dtype=torch.float64, device=device)
    z_inv = torch.exp(-2j * math.pi * freqs_norm)

    coefs = torch.tensor(coefs_tuple, dtype=torch.complex128, device=device).view(-1, 1)
    norms = torch.tensor(norms_tuple, dtype=torch.float64, device=device).view(-1, 1)

    # Square trick for fast power-of-4
    denom = 1.0 - coefs * z_inv.unsqueeze(0)
    denom_sq = denom * denom
    return norms / (denom_sq * denom_sq)


@lru_cache(maxsize=16)
def _get_gammatone_H_real_torch(N_fft: int, fs: float, cfs_tuple: tuple, norms_tuple: tuple, coefs_tuple: tuple,
                                device_str: str):
    """Half-spectrum filter for the REAL-output analysis path (`process_real`).

    For real input ``x`` the full-length response ``y = ifft(fft(x) * H)`` satisfies

        Re(y) = (y + conj(y)) / 2
              = ifft( X[k] * (H[k] + conj(H[-k])) / 2 )

    using ``conj(X[-k]) == X[k]`` (Hermitian symmetry of a real signal's spectrum).
    The bracketed factor is exactly ``Hmod[k]``, and ``X[k] * Hmod[k]`` is itself
    Hermitian, so ``Re(y) == irfft(rfft(x) * Hmod)`` -- a half-length inverse
    transform producing a real result directly.

    ``Hmod`` is built straight on the half grid, so the full-length ``H`` is never
    materialised. With ``w = exp(-2j*pi*k/N)`` and the filter's own
    ``H[k] = norms / (1 - c*w)^4``, the reflected term is

        conj(H[-k]) = conj( norms / (1 - c*conj(w))^4 ) = norms / (1 - conj(c)*w)^4

    since ``norms`` is real -- so the reflected term is the conjugate of the same
    expression evaluated at ``conj(w)``, and both live on ``k = 0..N/2``.

    The grid is `fftfreq`'s first half rather than `rfftfreq`, so ``w`` is bit-for-bit
    the entries `_get_gammatone_H_torch` uses and ``forward`` *is* ``H[k]``. The two
    conventions agree everywhere except the Nyquist bin, where `fftfreq` says -0.5 and
    `rfftfreq` says +0.5; that bin also reflects onto itself, so ``conj(H[-N/2])`` is
    ``conj(H[N/2])`` and the conjugate-grid formula does not apply there. It is set
    directly instead. (The convention still matters even though the grid is now built
    in float64: on the old float32 grid ``exp(-+ i*pi)`` carried an ~8.7e-8 imaginary
    residue and the two signs differed by ~1e-7 relative, which is exactly why the
    ``dtype=`` on the ``fftfreq`` call below is load-bearing. In float64 the residue
    is ~1.2e-16 and the two signs differ by ~2.4e-16 -- quieter, not gone.)

    Returns shape ``(num_bands, N_fft // 2 + 1)`` -- half the elements of
    `_get_gammatone_H_torch`'s full grid, agreeing with ``(H[k] + conj(H[-k])) / 2``
    taken from it to a few hundred ULP.

    Not bit-identical to it, despite being exact on Windows: the two build ``z_inv``
    by calling a complex ``torch.exp`` on grids of different length, and that is free
    to vectorize differently per length and per platform libm. CI caught the
    difference on ubuntu-latest at ``N_fft=4096``. Nothing downstream needs the
    stronger property -- ``process_real`` uses this filter and never the reflection.
    """
    device = torch.device(device_str)
    # See `_get_gammatone_H_torch`: the explicit `dtype=` must stay. Still `fftfreq`'s
    # first half rather than `rfftfreq`, so `forward` remains bit-for-bit the full
    # grid's `H[k]` and the Nyquist-convention note above still applies.
    freqs_norm = torch.fft.fftfreq(N_fft, dtype=torch.float64, device=device)[:N_fft // 2 + 1]
    z_inv = torch.exp(-2j * math.pi * freqs_norm)

    coefs = torch.tensor(coefs_tuple, dtype=torch.complex128, device=device).view(-1, 1)
    norms = torch.tensor(norms_tuple, dtype=torch.float64, device=device).view(-1, 1)

    # Square trick for fast power-of-4, once per reflection
    denom = 1.0 - coefs * z_inv.unsqueeze(0)
    denom_sq = denom * denom
    forward = norms / (denom_sq * denom_sq)

    denom_r = 1.0 - coefs * torch.conj(z_inv).unsqueeze(0)
    denom_r_sq = denom_r * denom_r
    reflected = torch.conj(norms / (denom_r_sq * denom_r_sq))

    combined = (forward + reflected) * 0.5
    if N_fft % 2 == 0:
        # Self-reflecting bin: (H + conj(H)) / 2 is exactly Re(H).
        combined[:, -1] = forward[:, -1].real
    return combined


@lru_cache(maxsize=16)
def _get_synthesizer_params_torch(delay_sec: float, fs: float, cfs_tuple: tuple, norms_tuple: tuple, coefs_tuple: tuple,
                                  device_str: str):
    device = torch.device(device_str)
    cfs = torch.tensor(cfs_tuple, dtype=torch.float64, device=device)
    norms = torch.tensor(norms_tuple, dtype=torch.float64, device=device)
    coefs = torch.tensor(coefs_tuple, dtype=torch.complex128, device=device)

    target_delay = int(round(delay_sec * fs))
    impulse = torch.zeros(target_delay + 2, dtype=torch.complex128, device=device)
    impulse[0] = 1.0

    # Inline the impulse response calculation to avoid cyclical dependencies
    # Applies 0.2s padding to prevent IIR circular convolution wrap-around
    pad_len = int(0.2 * fs)
    N_fft = 2 ** math.ceil(math.log2(impulse.shape[-1] + pad_len))

    X = torch.fft.fft(impulse, n=N_fft)
    # Same default-dtype trap as `_get_gammatone_H_torch`; the explicit `dtype=` must
    # stay. On the float32 grid this cost 5.8e-6 relative on `phase_factors` and
    # 3.2e-6 on `gains` (the integer `delays` were unaffected -- the argmax below does
    # not flip at that scale, checked at both 0.004 s and 0.016 s).
    freqs_norm = torch.fft.fftfreq(N_fft, dtype=torch.float64, device=device)
    z_inv = torch.exp(-2j * math.pi * freqs_norm)
    denom = 1.0 - coefs.view(-1, 1) * z_inv.unsqueeze(0)
    denom_sq = denom * denom
    H = norms.view(-1, 1) / (denom_sq * denom_sq)
    ir = torch.fft.ifft(X.unsqueeze(0) * H, n=N_fft, dim=1)[:, :impulse.shape[-1]]

    slice_dur = torch.abs(ir[:, :target_delay + 1])
    max_idx = torch.argmax(slice_dur, dim=1)
    delays = target_delay - max_idx

    prev_vals = torch.where(max_idx > 0, ir[torch.arange(ir.shape[0]), max_idx - 1], torch.tensor(0j, device=device))
    next_vals = ir[torch.arange(ir.shape[0]), max_idx + 1]
    slopes = next_vals - prev_vals
    slopes = slopes / (torch.abs(slopes) + 1e-15)
    phase_factors = 1j / slopes

    z = torch.exp(2j * math.pi * cfs / fs)
    gains = torch.ones_like(cfs)

    def z_response(z_pts):
        z_col = z_pts.unsqueeze(1)
        denom = 1.0 - coefs.unsqueeze(0) / z_col
        denom_sq = denom * denom
        return norms.unsqueeze(0) / (denom_sq * denom_sq)

    pos = z_response(z) * phase_factors.unsqueeze(0) * (z.unsqueeze(1) ** -delays.unsqueeze(0))
    neg = z_response(torch.conj(z)) * phase_factors.unsqueeze(0) * (torch.conj(z).unsqueeze(1) ** -delays.unsqueeze(0))
    combo = (pos + torch.conj(neg)) / 2.0

    for _ in range(100):
        spec = combo @ gains.to(torch.complex128)
        gains = gains / (torch.abs(spec) + 1e-15).real

    return delays, phase_factors, gains


def calculate_erb(fc: torch.Tensor) -> torch.Tensor:
    """
    Glasberg & Moore ERB, matching the MATLAB reference's `erbBW.m`.

    Used ONLY to pick the per-band decimation factor in the analysis filterbank
    (`myPemoAnalysisFilterBank.m:53`), NOT to build the gammatone filter coefficients.
    See GAMMATONE_ERB_INTERCEPT_HZ below for the (deliberately different) form used by
    the filter constructor.
    """
    return 24.7 * (0.00437 * fc + 1.0)


# Gammatone filter bandwidth constants, from `gammatone/Gfb_set_constants.m` (GFB_L and
# GFB_Q), i.e. equation (17) in Hohmann 2002. The filter constructor
# (`gammatone/Gfb_Filter_new.m:61`) computes its audiological ERB as
#     (GFB_L + fc / GFB_Q) * bandwidth_factor
#
# This is the SAME empirical fit as `calculate_erb` above, re-parameterized with one
# constant rounded: 1 / (24.7 * 0.00437) = 9.264488..., which Hohmann rounds to 9.265.
# The two therefore differ by ~5.5e-5 relative on the slope term. The MATLAB reference
# deliberately keeps both forms and applies each in a different place -- Hohmann's
# rounded form for the filter coefficients, Glasberg & Moore's form for the decimation
# factors -- so do NOT collapse them onto one formula.
GAMMATONE_ERB_INTERCEPT_HZ = 24.7  # GFB_L
GAMMATONE_ERB_QUALITY_FACTOR = 9.265  # GFB_Q

# Target size of one frequency-domain intermediate in `GammatoneAnalyzerTorch.process`
# and `.process_real`, both of which chunk their batch to stay under it. Not a
# correctness limit -- rows are filtered independently, so any value computes the same
# quantity -- but do NOT read that as bit-invariance, which it is not, and never was:
# torch's batched FFT does not sum a row the same way whether that row is transformed
# alone or alongside others. Measured on a (9, 20000) float64 input at fs=24000, all
# rows in one call against nine single-row calls: `process` 7.1e-16 (5.3e-16 of peak),
# `process_real` 5.6e-16 (4.2e-16 of peak). Through the public auditory entry point,
# where the adaptation cascade amplifies it, forcing one row per chunk moves the
# internal representation 4.5e-12 (4.4e-15 of peak) on the real path and 3.6e-12 on the
# complex one. That is roundoff, not a bug, and it is the reason this constant is a
# memory knob and not a tunable: changing it changes the last digits of the output.
#
# This is a memory bound, NOT a speedup -- measured 2026-08-10, mono 1.313 s vs 1.306 s
# and stereo 3.724 s vs 3.739 s against no chunking at all, i.e. neutral either way.
# What it buys is that the two `(rows, num_bands, N_fft)` complex intermediates stop
# scaling with the batch: 8 rows (4 sources x stereo) would otherwise allocate ~1 GB
# twice over to produce a few hundred MB of subbands. Tighter budgets do start to cost
# (134 MB measured 1.342 s / 3.868 s), so do not lower this expecting a win.
_FFT_CHUNK_BUDGET_BYTES = 256 * 1024 * 1024


def calculate_audiological_erb(fc: torch.Tensor, bandwidth_factor: float = 1.0) -> torch.Tensor:
    """
    Audiological ERB used to derive the gammatone filter coefficients.

    Mirrors `audiological_erb` in `gammatone/Gfb_Filter_new.m:61` (Hohmann 2002 eq. 13
    and 17). Intentionally distinct from `calculate_erb`; see the constants above.
    """
    return (GAMMATONE_ERB_INTERCEPT_HZ + fc / GAMMATONE_ERB_QUALITY_FACTOR) * bandwidth_factor


def get_erb_center_frequencies(filters_per_erb, low, center, high, device=None, dtype=None):
    # Only processes Python scalar floats at initialization
    def freq2erb(f): return 9.265 * math.log(1.0 + f / (24.7 * 9.265))

    # Processes multi-element PyTorch tensors; must use torch.exp
    def erb2freq(e):
        return (torch.exp(e / 9.265) - 1.0) * (24.7 * 9.265)

    e_low = freq2erb(low)
    e_center = freq2erb(center)
    e_high = freq2erb(high)

    num_below = math.floor((e_center - e_low) * filters_per_erb)
    start_erb = e_center - (num_below / filters_per_erb)

    erbs = torch.arange(start_erb, e_high + 1e-9, 1.0 / filters_per_erb, device=device, dtype=dtype)
    return erb2freq(erbs)


class GammatoneAnalyzerTorch:
    def __init__(self, fs, low, center, high, filters_per_erb, device, dtype):
        self.fs = fs
        self.center_frequencies = get_erb_center_frequencies(filters_per_erb, low, center, high, device, dtype)

        # erbBW form: these bandwidths drive the decimation factors, not the filter coefficients
        self.bandwidths = calculate_erb(self.center_frequencies)

        # Gammatone Constants
        # Hohmann's rounded form (Gfb_Filter_new.m:61), NOT the erbBW form used for decimation
        audiological_bandwidths = calculate_audiological_erb(self.center_frequencies)
        gamma_const = (math.pi * math.factorial(6) * (2.0 ** -6) / (math.factorial(3) ** 2))
        decay_const = audiological_bandwidths / gamma_const

        lambda_decay = torch.exp(-2.0 * math.pi * decay_const / fs)
        phase_step = 2.0 * math.pi * self.center_frequencies / fs

        self.coefs = lambda_decay * torch.exp(1j * phase_step)
        self.norms = 2.0 * (1.0 - torch.abs(self.coefs)) ** 4

        # Calculate Decimation
        decimation_alpha = 2.0
        self.decimations = torch.clamp(torch.floor(fs / (self.bandwidths * decimation_alpha)), min=1).to(torch.int32)

    def process(self, x: torch.Tensor) -> torch.Tensor:
        """
        Massively parallel FFT evaluation of 4th-order complex IIR filters.
        x shape: (*, T) -> Returns (*, num_bands, T)

        The frequency-domain product and its inverse transform run in row chunks rather
        than over the whole batch at once. Both intermediates are
        ``(rows, num_bands, N_fft)`` complex, and ``N_fft`` is the transform length --
        roughly 1.5x the signal -- so at full batch they are by far the largest
        allocations in the pipeline. Chunking caps them at ``_FFT_CHUNK_BUDGET_BYTES``.
        Every row is filtered independently, so the chunk width does not change what is
        computed, but it does change the last digits: torch's batched FFT is not
        bit-invariant in the number of rows (measured 7.1e-16 here). See that constant:
        this is a memory bound and measured speed-neutral, not a speedup.

        The valid region is also copied out rather than returned as a slice. The slice
        keeps its ``N_fft``-wide parent buffer alive for as long as the caller holds
        the result -- through the modulation multiply, in practice -- which is ~1.5x
        the memory the subbands actually need.
        """
        original_shape = x.shape[:-1]
        T = x.shape[-1]

        x_flat = x.reshape(-1, T)
        pad_len = int(0.2 * self.fs)
        N_fft = 2 ** math.ceil(math.log2(T + pad_len))
        num_bands = len(self.center_frequencies)

        # Global module cache call!
        H = _get_gammatone_H_torch(
            N_fft, self.fs, tuple(self.center_frequencies.tolist()),
            tuple(self.norms.tolist()), tuple(self.coefs.tolist()), str(x.device)
        )
        H = H.unsqueeze(0)

        bytes_per_row = num_bands * N_fft * 16
        rows_per_chunk = max(1, _FFT_CHUNK_BUDGET_BYTES // bytes_per_row)

        chunks = []
        for start in range(0, x_flat.shape[0], rows_per_chunk):
            rows = x_flat[start:start + rows_per_chunk]
            spectrum = torch.fft.fft(rows.to(torch.complex128), n=N_fft, dim=-1)
            filtered = torch.fft.ifft(spectrum.unsqueeze(1) * H, n=N_fft, dim=2)
            # Copy out the valid (linear-convolution) region so the N_fft-wide buffer
            # is released at the end of the iteration instead of being kept alive by
            # a view for the rest of the decomposition.
            chunks.append(filtered[..., :T].contiguous())

        combined = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        return combined.view(*original_shape, num_bands, T)

    def process_real(self, x: torch.Tensor) -> torch.Tensor:
        """``process(x).real`` for real ``x``, without ever forming the complex subbands.

        x shape: (*, T), real -> Returns (*, num_bands, T), real float64.

        The auditory path (`backend_torch/auditory_model.py`) discards the imaginary
        part immediately, so it never needs the analytic subbands the decomposition
        path is built around. For real input the real part is recoverable from a
        half-spectrum transform -- see `_get_gammatone_H_real_torch` for the identity
        ``Re(ifft(fft(x) * H)) == irfft(rfft(x) * Hmod)``. That halves the inverse
        transform -- one per band per row, and by far the dominant cost here -- halves
        the cached filter, and returns a *contiguous real* block rather than the
        stride-2 `.real` view of a complex buffer that stays alive behind it.

        `process` is deliberately left untouched: the decomposition genuinely needs
        the complex subbands, and this is a separate entry point rather than a flag.

        Not bit-identical to ``process(x).real`` -- ``rfft``/``irfft`` sum the same
        terms in a different order. It computes the same quantity; measured deviation
        is at the ULP level.
        """
        if x.is_complex():
            raise ValueError("process_real requires a real input; use process() for complex signals")

        original_shape = x.shape[:-1]
        T = x.shape[-1]

        x_flat = x.reshape(-1, T)
        pad_len = int(0.2 * self.fs)
        N_fft = 2 ** math.ceil(math.log2(T + pad_len))
        num_bands = len(self.center_frequencies)

        # Global module cache call!
        H = _get_gammatone_H_real_torch(
            N_fft, self.fs, tuple(self.center_frequencies.tolist()),
            tuple(self.norms.tolist()), tuple(self.coefs.tolist()), str(x.device)
        )
        H = H.unsqueeze(0)

        # Same `_FFT_CHUNK_BUDGET_BYTES` bound as `process`, sized against this path's
        # own largest intermediate: the (rows, num_bands, N_fft//2 + 1) complex product.
        # The real output block is (rows, num_bands, N_fft) float64, marginally smaller.
        # As in `process`, the width does not change what is computed but does change
        # the last digits -- `rfft` is no more row-invariant than `fft` (measured
        # 5.6e-16 batched against per-row). That constant's comment carries the numbers.
        bytes_per_row = num_bands * (N_fft // 2 + 1) * 16
        rows_per_chunk = max(1, _FFT_CHUNK_BUDGET_BYTES // bytes_per_row)

        chunks = []
        for start in range(0, x_flat.shape[0], rows_per_chunk):
            rows = x_flat[start:start + rows_per_chunk]
            spectrum = torch.fft.rfft(rows.to(torch.float64), n=N_fft, dim=-1)
            filtered = torch.fft.irfft(spectrum.unsqueeze(1) * H, n=N_fft, dim=2)
            # Copy out the valid (linear-convolution) region so the N_fft-wide buffer
            # is released at the end of the iteration instead of being kept alive by
            # a view for the rest of the pipeline.
            chunks.append(filtered[..., :T].contiguous())

        combined = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        return combined.view(*original_shape, num_bands, T)


class GammatoneSynthesizerTorch:
    def __init__(self, analyzer: GammatoneAnalyzerTorch, delay_sec: float):
        self.fs = analyzer.fs
        self.cfs = analyzer.center_frequencies

        # Instantly fetch from global cache!
        self.delays, self.phase_factors, self.gains = _get_synthesizer_params_torch(
            delay_sec, self.fs, tuple(self.cfs.tolist()),
            tuple(analyzer.norms.tolist()), tuple(analyzer.coefs.tolist()), str(self.cfs.device)
        )

        # Per-band (delay, gain) as Python scalars, read once here rather than once per
        # band per call. `delays` is `target_delay - argmax(|ir|[:target_delay + 1])`, so
        # it is non-negative by construction and bounded by `target_delay`; `process`
        # relies on that (a negative shift has no defined meaning for a causal delay
        # line, and the previous `gather` formulation would have indexed out of bounds
        # for one).
        self._delay_gain_plan = list(zip(
            (int(d) for d in self.delays.tolist()),
            (float(g) for g in self.gains.tolist()),
        ))

    def process(self, subbands: torch.Tensor, alignment: torch.Tensor = None) -> torch.Tensor:
        """Phase-align, delay and mix the subbands down to one fullband waveform.

        ``subbands`` is ``(*, Bands, Time)`` with an optional leading batch dimension.
        ``alignment`` is an optional ``(Bands, Time)`` (or ``(Bands, 1)``) complex tensor
        that pre-multiplies each band; it defaults to the per-band phase factors alone.
        The decomposition passes the *fused* modulation-times-phase matrix, which is why
        this hook exists -- see ``_get_synthesis_alignment_matrix_torch``.

        The band loop is deliberate. The vectorized formulation this replaced built the
        whole phase-aligned block (``(subbands * phase).real``), then an ``int64`` index
        tensor the same shape, then gathered, then ``where``d, then contracted with
        ``einsum`` -- four full passes plus the index tensor, over a block that is 62 MB
        mono and 123 MB stereo for the 5 s reference clip, none of which fits in cache.
        Here each band's product is a single row (~1.9 MB), and ``add_(..., alpha=gain)``
        folds the shift, the mask and the mixer gain into one accumulate:
        ``out[d:] += gain * (subbands[b, :Time-d] * a[b]).real``. The
        ``where(valid, shifted, 0)`` mask becomes the untouched ``out[:d]``, which is
        exactly the zeros it produced.

        Measured on the reference clip: 2.65x mono / 2.54x stereo on the whole chain in
        isolation (50.6 -> 19.1 ms and 92.3 -> 36.3 ms), and the temporaries it allocates
        drop from ~251 MB / ~437 MB to ~3 MB / ~6 MB. Do NOT also replace ``.real`` of the
        complex product with ``x.real*a.real - x.imag*a.imag``: that is bit-identical but
        0.85x, because the real/imag views are stride-2 and lose the vectorized complex
        multiply.

        Autograd-safe: ``out`` starts as a fresh zero tensor that requires no grad, the
        in-place adds are recorded normally (``add_`` saves nothing it later needs), and
        no tensor that requires grad is written into.
        """
        if alignment is None:
            alignment = self.phase_factors.unsqueeze(-1)

        time_steps = subbands.shape[-1]
        # Real counterpart of the complex product dtype, without materializing it.
        real_dtype = torch.empty(0, dtype=torch.promote_types(subbands.dtype, alignment.dtype)).real.dtype
        out = torch.zeros(subbands.shape[:-2] + (time_steps,), dtype=real_dtype, device=subbands.device)

        for band, (delay, gain) in enumerate(self._delay_gain_plan):
            length = time_steps - delay
            if length <= 0:
                continue  # delayed entirely past the end of the signal: contributes zeros
            product = subbands[..., band, :length] * alignment[band, :length]
            out.narrow(-1, delay, length).add_(product.real, alpha=gain)

        return out
