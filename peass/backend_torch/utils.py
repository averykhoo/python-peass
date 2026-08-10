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


@lru_cache(maxsize=256)
def _get_polyphase_kernel(
        up: int,
        down: int,
        dtype: torch.dtype,
        device: torch.device,
        half_length_factor: int
) -> torch.Tensor:
    r"""The Kaiser FIR folded into its ``(2*hf + 1, rate)`` polyphase kernel.

    The designed filter is ``2 * hf * rate + 1`` taps, where ``rate`` is whichever of
    ``up``/``down`` is not 1 -- so it grows with the rate, but the number of taps *per
    polyphase phase* is always ``L = 2 * hf + 1`` (21 at the default ``hf = 10``),
    independent of the rate. That is the whole reason the GEMM path below beats the FFT
    one: the real filter is 21 taps, not 24581.

    Reshaping the zero-padded filter to ``(L, rate)`` gives ``hp[j, p] = h[j*rate + p]``,
    i.e. row ``j`` of phase ``p``'s branch. The flips fold the convolution's index
    reversal in here once, at design time, instead of per call.
    """
    h_padded = get_resample_filter_torch(up, down, dtype, device, half_length_factor)[0]
    rate = max(up // math.gcd(up, down), down // math.gcd(up, down))
    taps_per_phase = 2 * half_length_factor + 1
    kernel = F.pad(h_padded, (0, taps_per_phase * rate - h_padded.shape[0]))
    kernel = kernel.reshape(taps_per_phase, rate).flip(0)
    if up == 1:
        # Decimation additionally reverses within each phase; see the derivation in
        # `_polyphase_decimate`.
        kernel = kernel.flip(1)
    return kernel.contiguous()


def _split_real_imag(x_flat: torch.Tensor) -> torch.Tensor:
    """Stack a complex ``(batch, n)`` as a real ``(2*batch, n)``: real rows, then imag.

    The resample filter is real, so a complex signal against it is a complex-by-real
    product — but torch promotes the filter and runs full complex arithmetic, which is
    four real multiplies and two adds per tap where one multiply and one add will do.
    Splitting the signal costs two copies of the signal and buys back 4x the FLOPs in
    the GEMM, which is where all the time is.

    This is exact, not an approximation: the discarded terms are the ``b*0`` and
    ``a*0`` products of a complex multiply by a real number, and the taps accumulate
    in the same order either way.
    """
    return torch.cat([x_flat.real, x_flat.imag], dim=0)


def _merge_real_imag(y_flat: torch.Tensor, batch: int) -> torch.Tensor:
    """Inverse of :func:`_split_real_imag`."""
    return torch.complex(y_flat[:batch], y_flat[batch:])


def _polyphase_interpolate(
        x_flat: torch.Tensor,
        up: int,
        in_len: int,
        out_len: int,
        half_length_factor: int
) -> torch.Tensor:
    r"""Zero-insert by ``up`` then FIR-filter, as one GEMM against the polyphase kernel.

    With ``hf = half_length_factor``, SciPy's zero-phase crop starts at
    ``n_pre_remove = hf*up``, so writing the output index as ``n = q*up + p``::

        y[q*up + p] = sum_{j=0}^{2hf} branch_p[j] * x[hf + q - j]

    Re-indexing by ``j' = 2hf - j`` turns the reversed sum into a forward sliding
    window over ``x`` zero-padded by ``hf`` on both sides, and the phase index ``p``
    becomes a plain matrix column::

        Y[q, p] = sum_{j'} window[q, j'] * kernel[j', p]

    so the whole interpolation is ``(batch*in_len, 2hf+1) @ (2hf+1, up)`` followed by a
    reshape -- the ``(q, p)`` grid is already in output order. No zero-inserted signal
    and no length-``in_len*up`` spectrum is ever materialised.
    """
    batch = x_flat.shape[0]
    is_complex = x_flat.is_complex()
    rows = _split_real_imag(x_flat) if is_complex else x_flat

    kernel = _get_polyphase_kernel(up, 1, rows.dtype, rows.device, half_length_factor)
    padded = F.pad(rows, (half_length_factor, half_length_factor))
    windows = padded.unfold(-1, 2 * half_length_factor + 1, 1)
    interleaved = (windows @ kernel).reshape(rows.shape[0], in_len * up)

    if is_complex:
        interleaved = _merge_real_imag(interleaved, batch)
    return interleaved[:, :out_len]


def _polyphase_decimate(
        x_flat: torch.Tensor,
        down: int,
        in_len: int,
        out_len: int,
        half_length_factor: int
) -> torch.Tensor:
    r"""FIR-filter then keep every ``down``-th sample, without filtering the discards.

    Here ``n_pre_remove = hf``, so ``y[n] = v[(hf + n)*down]`` of the full convolution.
    Splitting the tap index as ``k = j*down + c`` and re-indexing by ``j' = 2hf - j``
    gives::

        y[n] = sum_{j'} sum_{c} kernel[j', c] * X[n + j', c]

    where ``X[r, c]`` is the padded input reshaped to ``(rows, down)``. That reshape is
    free -- consecutive ``c`` are consecutive samples.

    Written that way it is ``2hf + 1`` matrix-vector products, but they all read the
    same block, so it collapses into a single GEMM plus a cheap gather. Contract the
    block against *all* the phases at once::

        M = X @ kernel^T        # (rows, 2hf+1),  M[r, j] = sum_c X[r, c] * kernel[j, c]
        y[n] = sum_j M[n + j, j]

    The GEMM is ``rows x down x (2hf+1)``, i.e. ``in_len * (2hf+1)`` multiply-adds
    regardless of ``down`` -- against an FFT of the *undecimated* length per band for
    the general path -- and ``M`` is only ``out_len x 21``, so the shifted-diagonal sum
    that follows is negligible.
    """
    taps_per_phase = 2 * half_length_factor + 1
    batch = x_flat.shape[0]
    is_complex = x_flat.is_complex()
    rows = _split_real_imag(x_flat) if is_complex else x_flat
    kernel = _get_polyphase_kernel(1, down, rows.dtype, rows.device, half_length_factor)

    num_rows = out_len + 2 * half_length_factor
    # Row r holds x[(r - hf)*down - down + 1 : (r - hf)*down + 1], reversed within the
    # row by the kernel's second flip, so the leading pad is one row minus one sample.
    left_pad = (half_length_factor + 1) * down - 1
    right_pad = num_rows * down - left_pad - in_len
    padded = F.pad(rows, (left_pad, right_pad))
    blocks = padded.reshape(rows.shape[0], num_rows, down)

    phase_sums = blocks @ kernel.T
    accumulator = phase_sums[:, :out_len, 0]
    for tap_idx in range(1, taps_per_phase):
        accumulator = accumulator + phase_sums[:, tap_idx:tap_idx + out_len, tap_idx]

    if is_complex:
        accumulator = _merge_real_imag(accumulator, batch)
    return accumulator


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

    Mixed rates (both ``up`` and ``down`` > 1 after reduction) go through an FFT linear
    convolution. That is used instead of conv1d/conv_transpose1d because torch has no
    optimized float64 convolution kernel — those fall back to slow reference
    kernels (`slow_conv2d`/`slow_conv_transpose2d`) that dominated the double
    precision decomposition. FFT is ~2x faster here and bit-identical to the conv
    path (verified to ~1e-15), while remaining fully differentiable.

    Pure interpolation and pure decimation instead take the polyphase GEMM path above.
    That covers almost every call the decomposition makes: the filterbank has 32 bands
    with 32 *distinct* decimation factors, so band grouping yields 32 groups of one or
    two rows each, and the FFT parallelises over the batch dimension it no longer has.
    Worse, the FFT works at the *undecimated* length either way — for a band decimated
    by 1229 it transformed a 121500-point spectrum to produce 98 output samples, and
    for the matching synthesis upsample it transformed 121500 points to filter 327.
    The polyphase form does the same arithmetic in ``2*half_length_factor + 1`` taps
    per output sample, so it is both asymptotically better and allocation-free.
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

    if down_reduced == 1:
        y_flat = _polyphase_interpolate(x_flat, up_reduced, in_len, out_len, half_length_factor)
        return y_flat.reshape(*shape_prefix, out_len).transpose(axis, -1)
    if up_reduced == 1:
        y_flat = _polyphase_decimate(x_flat, down_reduced, in_len, out_len, half_length_factor)
        return y_flat.reshape(*shape_prefix, out_len).transpose(axis, -1)

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
