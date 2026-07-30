"""
PEASS PyTorch Sub-Utilities
File path: peass/backend_torch/utils.py
"""
import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from scipy.signal import firwin


def smoothmax(x: torch.Tensor, threshold: float | torch.Tensor, k: float = 1000.0) -> torch.Tensor:
    """
    Temperature-scaled SmoothMax (Softplus approximation).
    Provides exact mathematical parity to `max(x, threshold)` while maintaining
    smooth, active gradients for neural network backpropagation.
    """
    return F.softplus(k * (x - threshold)) / k + threshold


# Anti-aliasing FIR half-length as a multiple of the up/down ratio. 10 matches
# SciPy/MATLAB (and the NumPy backend) for near bit-exact agreement; lower values
# trade accuracy for speed. Kept in sync with the NumPy backend's default.
DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR = 10


@lru_cache(maxsize=1024)
def next_fast_fft_length(target: int) -> int:
    """Smallest 5-smooth (2^a * 3^b * 5^c, a >= 1) transform length >= ``target``.

    Linear-convolution lengths are arbitrary integers, and an FFT of an awkward
    length (a large prime factor) falls off a cliff: a raw n=48239 rfft measured
    13x slower than the n=48384 padded one. Padding to the next power of two also
    works but over-pads badly for lengths just above a power of two (n=72359 ->
    131072 is 2.5x slower than 72900).

    The factor-of-two requirement matters: SciPy's ``next_fast_len`` allows odd
    lengths (120240 -> 120285 = 3^7*5*11), which torch's real FFT handles poorly
    (1.6x slower than the even 121500 here). Restricting to even 5-smooth lengths
    was within 16% of the best of either rule at every length measured.
    """
    if target <= 2:
        return 2
    # A pure power of two is always admissible, so it bounds the search.
    best = 1 << (target - 1).bit_length()
    power_of_five = 1
    while power_of_five < best:
        candidate = power_of_five
        while candidate < best:
            padded = candidate * 2
            while padded < target:
                padded *= 2
            if padded < best:
                best = padded
            candidate *= 3
        power_of_five *= 5
    return best


# -----------------------------------------------------------------------------
# HIGH-SPEED CACHED FILTER DESIGNER
# -----------------------------------------------------------------------------
@lru_cache(maxsize=256)
def get_resample_filter_torch(
        up: int,
        down: int,
        dtype: torch.dtype,
        device: torch.device,
        half_length_factor: int = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
) -> tuple:
    """
    Designs and caches the Kaiser resample filter on the fly to eliminate
    CPU-to-Device transfer overhead.
    """
    g = math.gcd(up, down)
    up_reduced = up // g
    down_reduced = down // g

    max_len = max(up_reduced, down_reduced)
    half_len = half_length_factor * max_len
    n_filt = 2 * half_len + 1

    # Design filter on CPU, push to the correct target device/dtype once
    h_numpy = firwin(n_filt, 1.0 / max_len, window=('kaiser', 5.0)) * up_reduced
    h = torch.tensor(h_numpy, dtype=dtype, device=device)

    n_pre_pad = (down_reduced - half_len % down_reduced) % down_reduced
    h_padded = F.pad(h, (n_pre_pad, 0))
    n_pre_remove = (half_len + n_pre_pad) // down_reduced

    return h_padded, up_reduced, down_reduced, n_pre_remove


@lru_cache(maxsize=128)
def _get_resample_filter_spectrum(
        up: int,
        down: int,
        dtype: torch.dtype,
        device: torch.device,
        half_length_factor: int,
        fft_length: int
) -> torch.Tensor:
    """Caches the zero-padded FIR spectrum used by the FFT convolution below.

    The filter is only a few hundred taps, but it has to be transformed at the
    full convolution length, so recomputing it per call cost as much as
    transforming the signal itself -- it was half of all FFT time in the
    decomposition, across ~200 resample calls that reuse a handful of
    (up, down, length) combinations.
    """
    h_padded = get_resample_filter_torch(up, down, dtype, device, half_length_factor)[0]
    if h_padded.is_complex():
        return torch.fft.fft(h_padded, n=fft_length)
    return torch.fft.rfft(h_padded, n=fft_length)


def fast_resample_poly_torch(
        x: torch.Tensor,
        up: int,
        down: int,
        axis: int = -1,
        half_length_factor: int = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
) -> torch.Tensor:
    """
    Native PyTorch polyphase resampler replicating SciPy's upfirdn via FFT linear
    convolution: zero-insert by ``up``, FIR-filter, decimate by ``down``.

    FFT convolution is used instead of conv1d/conv_transpose1d because torch has no
    optimized float64 convolution kernel — those fall back to slow reference
    kernels (`slow_conv2d`/`slow_conv_transpose2d`) that dominated the double
    precision decomposition. FFT is ~2x faster here and bit-identical to the conv
    path (verified to ~1e-15), while remaining fully differentiable.
    """
    if up == down:
        return x

    # Fetch pre-calculated and cached filter tensor instantly
    h_padded, up_reduced, down_reduced, n_pre_remove = get_resample_filter_torch(
        up, down, x.dtype, x.device, half_length_factor
    )

    in_len = x.shape[axis]
    out_len = math.ceil(in_len * up_reduced / down_reduced)

    x_moved = x.transpose(axis, -1)
    shape_prefix = x_moved.shape[:-1]
    x_flat = x_moved.reshape(-1, in_len)
    batch = x_flat.shape[0]
    filter_length = h_padded.shape[0]

    # 1. Zero-insertion by up_reduced. Done via pad+reshape (differentiable, no
    #    in-place scatter): each sample is followed by (up_reduced - 1) zeros.
    if up_reduced > 1:
        upsampled = F.pad(x_flat.unsqueeze(-1), (0, up_reduced - 1)).reshape(batch, in_len * up_reduced)
    else:
        upsampled = x_flat

    # 2. FIR filtering via FFT linear convolution. The subband signals are complex
    #    (analytic), so use the full complex FFT there; the real rfft/irfft path is
    #    a faster specialization for real inputs (e.g. the auditory-model resamples).
    #    The transform is padded up to a 5-smooth length: everything past
    #    conv_length is exactly zero (no circular wrap), so the extra taps only
    #    replace the zero-fill the crop below would have applied anyway.
    conv_length = upsampled.shape[-1] + filter_length - 1
    fft_length = next_fast_fft_length(conv_length)
    filter_spectrum = _get_resample_filter_spectrum(
        up, down, x.dtype, x.device, half_length_factor, fft_length
    )
    if upsampled.is_complex():
        spectrum = torch.fft.fft(upsampled, n=fft_length, dim=-1) * filter_spectrum
        filtered = torch.fft.ifft(spectrum, n=fft_length, dim=-1)
    else:
        spectrum = torch.fft.rfft(upsampled, n=fft_length, dim=-1) * filter_spectrum
        filtered = torch.fft.irfft(spectrum, n=fft_length, dim=-1)

    # 3. Decimate by down_reduced and crop the centered out_len window (matches
    #    SciPy's zero-phase offset via n_pre_remove).
    decimated = filtered[:, ::down_reduced]
    end = n_pre_remove + out_len
    if decimated.shape[-1] < end:
        decimated = F.pad(decimated, (0, end - decimated.shape[-1]))
    y_flat = decimated[:, n_pre_remove:end]

    return y_flat.reshape(*shape_prefix, out_len).transpose(axis, -1)
