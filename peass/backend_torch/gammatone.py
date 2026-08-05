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
    freqs_norm = torch.fft.fftfreq(N_fft, device=device)
    z_inv = torch.exp(-2j * math.pi * freqs_norm)

    coefs = torch.tensor(coefs_tuple, dtype=torch.complex128, device=device).view(-1, 1)
    norms = torch.tensor(norms_tuple, dtype=torch.float64, device=device).view(-1, 1)

    # Square trick for fast power-of-4
    denom = 1.0 - coefs * z_inv.unsqueeze(0)
    denom_sq = denom * denom
    return norms / (denom_sq * denom_sq)


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
    freqs_norm = torch.fft.fftfreq(N_fft, device=device)
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
        """
        original_shape = x.shape[:-1]
        T = x.shape[-1]

        x_flat = x.reshape(-1, T)
        pad_len = int(0.2 * self.fs)
        N_fft = 2 ** math.ceil(math.log2(T + pad_len))

        # Convert audio to freq domain
        X = torch.fft.fft(x_flat.to(torch.complex128), n=N_fft, dim=-1)

        # Global module cache call!
        H = _get_gammatone_H_torch(
            N_fft, self.fs, tuple(self.center_frequencies.tolist()),
            tuple(self.norms.tolist()), tuple(self.coefs.tolist()), str(x.device)
        )

        Y = X.unsqueeze(1) * H.unsqueeze(0)
        y = torch.fft.ifft(Y, n=N_fft, dim=2)
        y = y[..., :T]

        # Slice valid region (linear convolution match)
        return y.view(*original_shape, len(self.center_frequencies), T)


class GammatoneSynthesizerTorch:
    def __init__(self, analyzer: GammatoneAnalyzerTorch, delay_sec: float):
        self.fs = analyzer.fs
        self.cfs = analyzer.center_frequencies

        # Instantly fetch from global cache!
        self.delays, self.phase_factors, self.gains = _get_synthesizer_params_torch(
            delay_sec, self.fs, tuple(self.cfs.tolist()),
            tuple(analyzer.norms.tolist()), tuple(analyzer.coefs.tolist()), str(self.cfs.device)
        )

    def process(self, subbands: torch.Tensor) -> torch.Tensor:
        # subbands shape: (*, Bands, Time)
        aligned = (subbands * self.phase_factors.view(-1, 1)).real
        Time = aligned.shape[-1]

        # Vectorized delay shifting using advanced indexing (no loops, entirely autograd safe)
        idx = torch.arange(Time, device=aligned.device).unsqueeze(0) - self.delays.unsqueeze(1)
        valid = idx >= 0
        idx_clamped = torch.clamp(idx, min=0)

        # Prepare shape expansion for any batch dimensions
        shape_prefix = [1] * (aligned.dim() - 2)
        idx_clamped = idx_clamped.view(*shape_prefix, self.delays.shape[0], Time).expand_as(aligned)
        valid = valid.view(*shape_prefix, self.delays.shape[0], Time).expand_as(aligned)

        shifted = torch.gather(aligned, -1, idx_clamped)
        out = torch.where(valid, shifted, 0.0)

        return torch.einsum('b, ...bt -> ...t', self.gains.to(out.dtype), out)
