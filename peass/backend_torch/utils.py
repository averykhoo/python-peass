"""
PEASS PyTorch Sub-Utilities
File path: peass/backend_torch/utils.py
"""
import math
from fractions import Fraction
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

    Keyed per *rate*, and deliberately kept that way:
    :func:`_get_modulated_polyphase_kernel` is the per-*band* sibling that carries a
    gammatone band's modulation in the taps, and it has its own cache. Widening this one
    to hold both would need 26..42 entries per filterbank configuration and would evict
    the plain real kernels that the fullband 3/2 and 2/3 resamples, ``auditory_model.py``
    and every unmodulated complex caller still need.
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


# Denominator at which the exact rational ``fc / fs`` is *split* (not truncated) into a
# coarse rational plus a float64 remainder, so that the coarse part's modular multiply
# stays inside int64. See :func:`_reduced_phase_exp` for why this must be a split and not
# a `limit_denominator` approximation: 10**12 < 2**40, which with the index split below
# leaves the modular product bounded by 2**61.
_PHASE_DENOMINATOR_LIMIT = 10 ** 12

# The index is split as ``n = hi * _PHASE_INDEX_SPLIT + lo`` before the modular multiply,
# so that neither partial product leaves int64. With ``num < 2**40`` and ``lo < 2**21``
# the low product is under ``2**61``; the high product is under ``2**60`` for any
# ``|n| < 2**41``, i.e. 2.2e12 samples -- 47000 years at 48 kHz, and 35 TB of complex128,
# so the bound is structural rather than a limit worth branching on. This replaces an
# earlier ``numerator * max|index| >= 2**62`` guard whose fallback was a per-sample pure
# Python ``Fraction`` loop: that guard tripped at 23.1M upsampled samples (~321 s at
# 48 kHz) and turned the cache fill into tens of millions of Python-level operations per
# band with no warning.
_PHASE_INDEX_SPLIT = 2 ** 21


@lru_cache(maxsize=256)
def _modulation_phase(center_frequency_hz: float, sampling_frequency_hz: float) -> Fraction:
    """``fc / fs`` as an **exact** rational -- no approximation of any kind.

    ``Fraction(float)`` is exact for any float64 and both arguments are float64, so this
    is the true ratio the band was designed at, and it is what :func:`_reduced_phase_exp`
    reduces. The denominator is large -- measured over every band of every supported
    geometry (8k/16k/22.05k/44.1k/48k, 26..42 bands each), worst ``1.013e19`` at
    ``fc = 42.330280964441435``, ``fs = 72000`` -- which is why the reduction splits it
    rather than working with it directly.

    **This function used to return ``limit_denominator(10**12)`` of the ratio and that was
    a live accuracy defect**, measured 2026-08-18. ``limit_denominator`` is a *best
    rational approximation*: it is free to return a denominator far below the limit when
    the continued fraction has a large early partial quotient, and the analyzer's ERB grid
    puts one centre frequency exactly one ULP above the base frequency
    (``cf = 1000.0000000000001``). At ``fs = 24000`` the exact ratio
    ``2932031007402667/70368744177664000`` collapsed to exactly ``1/24``, discarding
    1.137e-16 *relative* -- which is 3.572e-12 rad at ``n = T = 120000`` and measured
    3.4373e-12 of that band's own peak on the real 32-band filterbank, against ~1e-15 on
    every other band. At a fixed clip *duration* it is rate-invariant, because the ratio
    error scales as ``1/fs`` and the index range as ``fs``: the denominator collapses to
    12 at 8 kHz, 24 at 16 kHz, 1323 at 22.05k and 44.1k, 72 at 48 kHz, always on that one
    band and on no other.

    That is the identical failure mode this docstring used to *reject* the separate
    ``limit_denominator(fc)/limit_denominator(fs)`` recipe of
    ``.scratch/handoff-2026-08-15/harnesses/verify_exp.py`` for (5 of 176 swept centre
    frequencies do not round-trip, worst 1.14e-16 relative). Limiting the ratio commits
    the same 1.14e-16 by the same mechanism. Neither is safe; nothing may approximate the
    ratio.

    It was invisible for the same reason ``verify_exp.py``'s was: every check compared the
    fold against a reference built from *this function's own* return value, so the cap
    error cancelled on both sides. Only a reference built from an independent, uncapped
    ``Fraction(cf) / Fraction(fs)`` can see it -- which is what
    ``tests/.../test_torch_utils.py::test_torch_reduced_phase_exp_matches_an_exact_fraction_reference``
    now does, at the collapsing centre frequency.
    """
    return Fraction(center_frequency_hz) / Fraction(sampling_frequency_hz)


def _reduced_phase_exp(
        phase: Fraction,
        sign: int,
        start: int,
        step: int,
        count: int,
        device: torch.device,
        dtype: torch.dtype = torch.complex128
) -> torch.Tensor:
    r"""``exp(sign * 2j*pi * phase * (start + step*i))`` for ``i = 0..count-1``.

    The phase is reduced modulo one turn **before** the exponential, and that reduction --
    not the fold's algebra -- is where all of the accuracy lives. Measured on this
    geometry: ``2*pi*fc*n/fs`` reaches **2.3465e5 rad** at ``fc = 7468.977``,
    ``fs = 24000``, ``n = 120000``, so ``eps*theta = 5.2e-11 rad`` and any ``torch.exp``
    of the raw argument carries that. The modulation matrix this replaced was built
    exactly that way; folding it into the taps with the same naive exponential reproduces
    that matrix to only 7.538e-12 relative over the 32 bands, while the identical fold
    with this reduction on both sides measures **9.538e-16**. Four orders of magnitude,
    from the exponential alone. Reduced-fold against the *unreduced* matrix -- i.e. the
    move this change actually lands -- is 1.128e-11, and it is toward the true value, not
    away from it.

    (``TODO.md`` and ``.scratch/handoff-2026-08-15/notes/p5_notes.md`` state the argument
    reaches "~2.2e8 rad". That is wrong by three orders and must not be repeated: it
    would predict ~5e-8 of error, which nobody has ever measured, whereas
    ``2.2e-16 * 2.35e5 = 5.2e-11`` matches what was.)

    **The exact ratio is split, never approximated.** ``phase`` carries a denominator up
    to 1.013e19 (measured worst over every supported geometry), which no int64 modular
    multiply can hold, so it is written as ``coarse + residual``:

    * ``coarse = phase.limit_denominator(_PHASE_DENOMINATOR_LIMIT)`` drives an exact
      integer reduction ``(num * n) mod den``, with ``n`` itself split at
      ``_PHASE_INDEX_SPLIT`` so neither partial product leaves int64;
    * ``residual = float(phase - coarse)`` carries *everything the cap dropped*, in
      float64, and is applied as ``n * residual`` turns on top.

    That second term is what makes this correct. ``limit_denominator`` is a best rational
    approximation and can collapse hard -- ``cf = 1000.0000000000001`` at ``fs = 24000``
    goes to exactly ``1/24``, throwing away 1.137e-16 relative, which was measured as
    3.4373e-12 of that band's peak (see :func:`_modulation_phase`). Carrying the remainder
    costs one multiply and one add and cannot itself lose anything that matters:
    ``|residual| < 1/(q * 1e12) <= 1e-12`` turns per index, so ``n*residual`` is at most
    ~1.2e-7 turns at ``n = T = 120000`` and its own float64 rounding is ~1e-23 turns,
    seven orders below the ULP of the sum it is added to.

    The coarse reduction is exact end to end: ``den <= 1e12 < 2**53``, so the residue and
    the divisor are both exact in float64 and ``reduced / den`` is correctly rounded.

    Measured. Against a 50-digit ``mpmath`` reference at the collapsing centre frequency
    the split is **9.75e-16** where the capped-only form is 1.18e-12. Swept over all 176
    bands of the five supported geometries at ``count = 120000``, split minus capped-only:
    every band except the collapsing one moves by at most **1.6e-16** (median ~1e-18, i.e.
    the cap was already tight there and the remainder only trims the last bit), while the
    collapsing band moves by 7.14e-12 (8 kHz), 3.57e-12 (16 kHz), 2.59e-12 (22.05 kHz),
    1.30e-12 (44.1 kHz), 1.19e-12 (48 kHz).

    Reducing ``turns`` into ``(-0.5, 0.5]`` instead of ``[0, 1)`` halves the magnitude of
    the ``2*pi*turns`` argument and measures 5.16e-16 against the same gold reference,
    versus 9.14e-16 here -- a real 1.8x, recorded and deliberately not taken: both forms
    sit on the float64 floor of ``cos``/``sin`` at an argument of magnitude 2*pi, four
    orders inside the bar, and it is an accuracy micro-optimisation orthogonal to the
    defect this split exists to fix. Do not take it without a measurement that shows it
    changing something downstream.

    ``torch.remainder`` follows the *divisor's* sign, so a negative ``start`` -- the
    synthesis tap index runs down to ``-hf*U = -4090`` -- still lands in ``[0, den)``, and
    ``torch.div(..., rounding_mode="floor")`` splits negative indices consistently with
    it.

    ``dtype`` is the complex dtype of the result. The reduction always runs in float64 and
    is rounded once at the end -- reducing in the signal's precision would defeat the
    point of reducing at all.

    Cost: this runs once per cache fill, never per call, exactly like
    :func:`_get_polyphase_kernel`, and it is now O(count) torch work with no Python-level
    loop at any length. Summed over the 32 bands it reduces ~157k points (77,312 tap
    phases + 80,257 residual points) against the 3.84M-element modulation matrix it
    removes -- ~2% of the work it deletes.
    """
    coarse = phase.limit_denominator(_PHASE_DENOMINATOR_LIMIT)
    denominator = coarse.denominator
    numerator = coarse.numerator % denominator               # 0 <= num < den, exactly
    residual = float(phase - coarse)                         # turns per unit index

    index = torch.arange(count, dtype=torch.int64, device=device) * step + start
    high = torch.div(index, _PHASE_INDEX_SPLIT, rounding_mode="floor")
    low = index - high * _PHASE_INDEX_SPLIT
    high_step = (numerator * _PHASE_INDEX_SPLIT) % denominator
    reduced = (numerator * low + high_step * high).remainder(denominator)

    turns = reduced.to(torch.float64) / denominator + index.to(torch.float64) * residual
    return torch.polar(torch.ones_like(turns), (sign * 2.0 * math.pi) * turns).to(dtype)


def _complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """The complex dtype a folded (modulated) kernel of real precision ``real_dtype`` takes.

    ``promote_types(x, complex64)`` is the standard widening: float32 and float16 go to
    complex64, float64 to complex128, and a complex dtype passed in comes back unchanged.
    The folded path builds its phase vector in float64 for accuracy and rounds once, so
    this is the only place that decides the precision the fold actually runs at.
    """
    return torch.promote_types(real_dtype, torch.complex64)


@lru_cache(maxsize=256)
def _get_modulation_vector(
        phase: Fraction,
        sign: int,
        start: int,
        step: int,
        count: int,
        device: torch.device,
        dtype: torch.dtype = torch.complex128
) -> torch.Tensor:
    """Cached :func:`_reduced_phase_exp` -- the residual/pre-multiply the fold leaves behind.

    Folding a band's modulation into the taps does not make it vanish; it moves it off the
    full-rate signal and onto the *decimated* one. For analysis that is
    ``m[(hf+i)*down]``, ``out_len`` long (294..8572 per band here, against the 120000 the
    modulation matrix multiplied); for synthesis it is ``m[up*i]`` on the decimated
    **input**, ``in_len`` long (the same 294..8572, against the 120336-wide alignment row
    it replaces). Either way ~0.35% of one full-rate pass summed over all 32 bands.

    Applied as its own elementwise multiply, deliberately. Folding it into the last term
    of the shifted-diagonal sum is not expressible -- it is a factor on the whole sum, and
    a factor does not distribute into one addend -- and fusing it into the final *add*
    would still take two kernels, because torch has ``addcmul`` (``a + b*c``) and nothing
    for ``(a + b) * c``. At 0.35% there is nothing to spend on it.

    ``maxsize=256`` is sized by **bands**, not by rates: one entry per band per side, 84
    at 48 kHz (42 bands x 2), where :func:`_get_polyphase_kernel`'s same-sized cache holds
    one entry per *rate*. ``dtype`` is part of the key so the residual comes back in the
    same precision the folded kernel ran at; the filterbank only ever asks for
    complex128.
    """
    return _reduced_phase_exp(phase, sign, start, step, count, device, dtype)


@lru_cache(maxsize=256)
def _get_modulated_polyphase_kernel(
        up: int,
        down: int,
        phase: Fraction,
        real_dtype: torch.dtype,
        device: torch.device,
        half_length_factor: int
) -> torch.Tensor:
    r""":func:`_get_polyphase_kernel` with one gammatone band's modulation in the taps.

    Complex modulation distributes through convolution exactly, so with
    ``m[n] = exp(-2j*pi*fc*n/fs)`` (unit modulus, hence ``m[a-b] = m[a]*conj(m[b])``) the
    full-rate demodulate-then-decimate collapses onto the 21 taps::

        y[i] = m[(hf+i)*down] * sum_k ( h[k] * conj(m[k]) ) * x[(hf+i)*down - k]

    and the mirror for interpolate-then-modulate, whose taps carry ``m_s[k - hf*up]``.
    **Both sides use ``sign = +1``; only the start offset differs** -- the analysis
    conjugation flips the modulation's minus to a plus, and the synthesis tap needs no
    conjugation but is offset by ``-n_pre_remove = -hf*up``. A sign slip shows up as an
    O(1) error; a dropped offset is a per-band *constant rotation* with no shape change
    and no test failure, so it is only visible to a per-band parity check.

    **The flat tap index needs no shift.** ``get_resample_filter_torch`` returns
    ``n_pre_pad == 0`` identically whenever ``up == 1`` or ``down == 1`` (``half_len`` is
    then a multiple of ``down_reduced``), so ``h_padded`` is the designed filter
    unshifted and its flat index *is* SciPy's ``k``. The modulation therefore multiplies
    the flat filter **before** the pad/reshape/flip, and the flips carry it to the right
    ``(j, c)`` for free; applying it afterwards means re-deriving
    ``h[(L-1-j)*down + (down-1-c)]``, which buys nothing and is easy to get wrong. The
    ``flip(1)`` is decimation-only for the same reason it is in
    :func:`_get_polyphase_kernel`; omitting it there, or adding it on the interpolation
    side, silently permutes the lanes.

    ``real_dtype`` is the *signal's real* dtype, not the signal's. Handing
    ``get_resample_filter_torch`` a complex dtype **silently succeeds** -- it returns a
    complex filter with an all-zero imaginary part and caches a duplicate of every filter,
    which is exactly the 64-entry / 2.5 MB regression recorded at
    :func:`fast_resample_poly_torch`. Design real, promote via the complex phase vector.
    The promotion lands in :func:`_complex_dtype_for` of ``real_dtype``, not in
    complex128 unconditionally: the phase vector is reduced in float64 whatever the
    signal's precision, but the *kernel* has to come back in the precision the GEMM will
    run at, or a complex64 signal meets a complex128 kernel and ``bmm`` raises
    "expected m1 and m2 to have the same dtype".

    Own cache, keyed per band (``phase``) rather than per rate, and ``maxsize=256`` is
    sized by **bands**: 84 entries at 48 kHz for both sides. Resident cost 1.298 MB per
    side for the 32 bands here, against the 61.44 MB modulation matrix removed. ``None``
    is not a valid ``phase``, so the unmodulated kernels can never collide with these.
    """
    h_padded = get_resample_filter_torch(up, down, real_dtype, device, half_length_factor)[0]
    rate = max(up, down)                                # one of the two is 1 on this path
    taps_per_phase = 2 * half_length_factor + 1
    start = 0 if up == 1 else -half_length_factor * up
    taps = h_padded * _reduced_phase_exp(
        phase, 1, start, 1, h_padded.shape[0], device, _complex_dtype_for(real_dtype)
    )
    kernel = F.pad(taps, (0, taps_per_phase * rate - taps.shape[0]))
    kernel = kernel.reshape(taps_per_phase, rate).flip(0)
    if up == 1:
        kernel = kernel.flip(1)
    return kernel.contiguous()


@lru_cache(maxsize=256)
def _mixed_polyphase_geometry(up: int, down: int, half_length_factor: int) -> tuple:
    r"""Integer index algebra for the general ``up/down`` polyphase form.

    SciPy's output is ``y[n] = sum_k h[k] * v[(n_pre_remove + n)*D - k]`` with ``v`` the
    input zero-inserted by ``U``, so only ``k`` congruent to ``s = (n_pre_remove + n)*D``
    modulo ``U`` survives. Writing ``k = p + U*j`` collapses the sum onto one branch::

        y[n] = sum_j h[p(n) + U*j] * x[Q(n) - j],   p(n) = s mod U,  Q(n) = s div U

    Splitting the output index as ``n = m*U + r`` makes both fixed per residue, because
    ``s`` gains exactly ``m*U*D``::

        p(n) = p(r),   Q(n) = Q(r) + m*D

    so residue ``r`` is a decimation-by-``D`` of ``x`` against its own ``L``-tap branch,
    ``L = ceil(len(h)/U)`` -- 21 taps at the default ``hf`` for ``3/2``, against the
    ~120k-point transform the FFT route runs instead.

    Reversing the branch (``j' = L-1-j``) turns it into a forward window starting at
    ``base(r) = Q(r) - L + 1``; ``offset`` shifts every window non-negative so one common
    block grid serves all residues, and ``taps`` is how many ``D``-wide blocks the widest
    window spans. Returns ``(phase, column, taps_per_phase, taps, offset)`` with the two
    index vectors as CPU int64 -- the caller moves them where they are needed.
    """
    h_padded, up_reduced, down_reduced, n_pre_remove = get_resample_filter_torch(
        up, down, torch.float64, torch.device("cpu"), half_length_factor
    )
    filter_length = h_padded.shape[0]
    taps_per_phase = -(-filter_length // up_reduced)

    start = (n_pre_remove + torch.arange(up_reduced)) * down_reduced
    phase = start % up_reduced
    first = torch.div(start, up_reduced, rounding_mode="floor") - (taps_per_phase - 1)
    offset = -min(0, int(first.min()))
    column = first + offset
    taps = -(-(int(column.max()) + taps_per_phase) // down_reduced)
    return phase, column, taps_per_phase, taps, offset


@lru_cache(maxsize=256)
def _get_mixed_polyphase_kernel(
        up: int,
        down: int,
        dtype: torch.dtype,
        device: torch.device,
        half_length_factor: int
) -> tuple:
    r"""The FIR laid out as the ``(down, up * taps)`` matrix the mixed-rate GEMM contracts.

    With the block grid from :func:`_mixed_polyphase_geometry`, input sample
    ``base(r) + m*D + j'`` is entry ``c`` of block ``m + t`` where ``t*D + c = column[r] +
    j'``. Scattering branch ``r``'s reversed taps into a ``taps*D``-wide row at ``column[r]``
    therefore places every tap at the ``(block, lane)`` the GEMM will read it from, and
    the residue/tap pair becomes a plain output column::

        kernel[c, r*taps + t] = h_branch_r[L-1 - (t*D + c - column[r])]

    Entries whose window index falls outside ``[0, L)`` stay zero, so the sum picks up
    only the real taps. Built once per ``(up, down, dtype, device)`` and cached.
    """
    h_padded = get_resample_filter_torch(up, down, dtype, device, half_length_factor)[0]
    filter_length = h_padded.shape[0]
    g = math.gcd(up, down)
    up_reduced, down_reduced = up // g, down // g
    phase, column, taps_per_phase, taps, offset = _mixed_polyphase_geometry(
        up, down, half_length_factor
    )
    phase, column = phase.to(device), column.to(device)

    # Branch r, reversed: branch[r, j'] = h[phase[r] + U*(L-1-j')], zero past the filter.
    tap_index = torch.arange(taps_per_phase, device=device)
    flat_index = phase[:, None] + up_reduced * (taps_per_phase - 1 - tap_index)[None, :]
    branch = torch.where(
        flat_index < filter_length,
        h_padded[flat_index.clamp(max=filter_length - 1)],
        torch.zeros((), dtype=dtype, device=device),
    )

    kernel = torch.zeros(up_reduced, taps * down_reduced, dtype=dtype, device=device)
    kernel.scatter_(1, column[:, None] + tap_index[None, :], branch)
    kernel = kernel.reshape(up_reduced, taps, down_reduced).permute(2, 0, 1)
    return kernel.reshape(down_reduced, up_reduced * taps).contiguous(), taps, offset


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

    **"The resample filter is real" is not a universal.** Where a gammatone band's
    modulation has been folded into the taps (:func:`_get_modulated_polyphase_kernel`)
    the kernel is complex by construction, the premise above does not hold, and the
    folded callers skip the split entirely and run a true complex GEMM. Keeping the
    folded kernel on this split path -- widening ``N`` to 42 with ``[K_re | K_im]`` --
    pays the same 2x flops without recovering the split, and measured **0.717x**: it
    loses. It is the same family as the block-diagonal form :func:`_merge_real_imag`
    already rejects at exactly 2.00x the multiply-adds. Do not reintroduce either as
    "the obvious way to keep the real GEMM".

    Used by the two padless fast paths, which have no pad for the split to hide inside;
    everything that still materialises a padded buffer calls :func:`_split_pad_real_imag`
    instead and gets the split for free. The contiguous ``(2*batch, n)`` result is what the
    fast paths need anyway: the plane axis has to fold into the GEMM's batch axis, and a
    de-interleaved plane *view* of a complex buffer has matrix strides ``(2*down, 2)`` --
    neither of them 1 -- which torch's CPU ``bmm`` silently makes contiguous, i.e. it would
    pay this copy inside the GEMM instead of before it, on the 21x-expanded operand.
    """
    return torch.cat([x_flat.real, x_flat.imag], dim=0)


def _split_pad_real_imag(x_flat: torch.Tensor, left: int, right: int) -> tuple:
    """Zero-pad, and for complex input do :func:`_split_real_imag` in the *same* copy.

    Returns ``(padded, was_complex)``. The layout is exactly what
    ``F.pad(_split_real_imag(x), (left, right))`` produced -- a contiguous
    ``(2*batch, left + n + right)`` with the real rows first -- so everything downstream,
    :func:`_merge_real_imag` included, is unchanged.

    The split is free on top of a pad it was already paying for. ``x.real``/``x.imag`` are
    ``view_as_real(x).select(-1, 0/1)``, i.e. stride-2 views, so ``torch.cat`` was a
    strided gather into a fresh contiguous buffer that the very next ``F.pad`` read back
    and copied again. ``view_as_real(x).permute(2, 0, 1)`` is the same two views as one
    free ``(2, batch, n)`` view, and ``F.pad`` allocates and copies whatever its input's
    strides are -- so padding *that* does the gather and the pad in one pass. Whenever at
    least one pad width is positive its output is contiguous, hence the final ``reshape``
    to ``(2*batch, ...)`` is a view and every consumer sees byte-for-byte the tensor it saw
    before. One full-signal buffer per complex call disappears.

    **The contiguity above holds only while some pad width is positive**, which is the only
    way this is ever called -- measured over a real decomposition with grad on, so all three
    padded callers run: 132 distinct ``(left, right)`` pairs, ``min left = min right = 10``,
    zero calls with ``left == right == 0``. When *every* width is ``<= 0`` ``F.pad`` narrows
    instead, and its result keeps the ``(1, 2n, 2)`` plane strides rather than coming back
    contiguous. The ``reshape`` then splits by ``batch``: at ``batch == 1`` it drops a size-1
    axis and stays a **non-contiguous** stride-``(1, 2)`` view, and at ``batch >= 2`` it
    cannot view those strides so it silently **copies**. Values are unaffected either way --
    verified bit-identical against the old ``F.pad(_split_real_imag(x), ...)`` form, and
    verified through the stride-sensitive ``unfold``/``matmul`` consumers -- so this is a
    layout and allocation caveat, not a correctness one. Reaching it needs
    ``half_length_factor == 0``; see :func:`_polyphase_decimate` on why that value is worse
    than a layout problem.

    The plane axis has to lead for this to work: ``permute(2, 0, 1)`` keeps the two planes
    as the outer axis the row-stacking wants, whereas ``view_as_real``'s native trailing
    real/imag axis is where ``unfold``/``reshape`` need the time axis. Do *not* try to skip
    the pad and feed the stride-2 view straight to ``unfold`` -- ``matmul`` can no longer
    view the result and clones the whole window expansion instead.

    ``view_as_real`` rejects a lazily-conjugated tensor where ``torch.cat`` silently
    worked, hence ``resolve_conj`` (a no-op on everything this package produces, but
    ``fast_resample_poly_torch`` is a public entry point).

    Carries the same qualifier as :func:`_split_real_imag`: the folded (modulated) kernel
    is complex, so those callers pad with a plain ``F.pad`` on the complex block instead
    and never split. The pad is unchanged; only the split half goes away.
    """
    if not x_flat.is_complex():
        return F.pad(x_flat, (left, right)), False
    planes = torch.view_as_real(x_flat.resolve_conj()).permute(2, 0, 1)
    padded = F.pad(planes, (left, right))
    return padded.reshape(2 * x_flat.shape[0], padded.shape[-1]), True


def _merge_real_imag(y_flat: torch.Tensor, batch: int) -> torch.Tensor:
    """Inverse of :func:`_split_real_imag`.

    There is deliberately no fused counterpart to :func:`_split_pad_real_imag` here. A free
    ``view_as_complex`` needs real/imag adjacent per sample at stride 1, but that axis
    enters the GEMM on the *operand* side, so it necessarily lands on an outer axis of the
    result -- and BLAS writes its output with column stride 1 regardless. The only layout
    that removes this copy contracts an interleaved buffer against a block-diagonal
    ``(2*down, 2*taps)`` kernel, which is exactly 2.00x the multiply-adds (half of them
    against structural zeros) and gives back the entire win the split exists to buy.
    ``torch.complex`` is one fused kernel over 2N elements; it is the cheap half.

    Not reached on the folded path -- a complex kernel produces a complex accumulator
    directly, so there is nothing to merge. That is the second half of P5's saving, on top
    of the modulation pass itself; the widened-real alternative that would have kept this
    function in play measured 0.717x (see :func:`_split_real_imag`).
    """
    return torch.complex(y_flat[:batch], y_flat[batch:])


def _polyphase_interpolate(
        x_flat: torch.Tensor,
        up: int,
        in_len: int,
        out_len: int,
        half_length_factor: int,
        phase: Fraction | None = None
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

    **``phase`` folds a gammatone band's re-modulation into the taps.** Passing the band's
    exact ``fc/fs`` (:func:`_modulation_phase`) makes this compute
    ``interpolate(x) * exp(+2j*pi*fc*n/fs)`` up to the ``in_len``-long pre-multiply that
    :func:`fast_interpolate_modulate_torch` applies to the *decimated* input first -- one
    complex zgemm instead of a full-rate 62 MB alignment pass, a ``_split_pad_real_imag``
    gather and a ``_merge_real_imag`` allocation. Same trade as the decimation side and the
    same verdict: the GEMM's flops double, ``K = 21`` is far too skinny for that to bind,
    and the rejected widened ``N = 42`` real dgemm measured **0.717x**.

    There is no grad branch here and none is needed -- ``F.pad``, ``unfold``, complex
    ``matmul`` and the pre-multiply are all autograd-safe, so unlike
    :func:`_polyphase_decimate` this half of the fold cannot create a grad/no-grad
    numerical split. Gradients w.r.t. all 32 complex subband tensors, folded against the
    interpolate-then-multiply route, measured 1.513e-14 of peak.

    ``phase=None`` is today's behaviour byte for byte; the real and unmodulated-complex
    callers never take the folded branch.
    """
    batch = x_flat.shape[0]
    folded = phase is not None
    if folded:
        # The split exists to keep a complex signal off a real filter; the folded kernel is
        # complex, so only the pad half of `_split_pad_real_imag` is wanted. Design in the
        # *real* dtype -- a complex dtype into `get_resample_filter_torch` silently caches a
        # zero-imaginary duplicate of every filter (see `fast_resample_poly_torch`).
        kernel = _get_modulated_polyphase_kernel(
            up, 1, phase, x_flat.real.dtype, x_flat.device, half_length_factor
        )
        # A real signal has to be widened to meet a complex kernel; `.to` returns the same
        # object when the dtypes already agree, which is what the filterbank drives.
        padded = F.pad(x_flat.to(kernel.dtype), (half_length_factor, half_length_factor))
        is_split = False
    else:
        padded, is_split = _split_pad_real_imag(x_flat, half_length_factor,
                                                half_length_factor)
        kernel = _get_polyphase_kernel(up, 1, padded.dtype, padded.device,
                                       half_length_factor)

    windows = padded.unfold(-1, 2 * half_length_factor + 1, 1)
    interleaved = (windows @ kernel).reshape(padded.shape[0], in_len * up)

    if is_split:
        interleaved = _merge_real_imag(interleaved, batch)
    return interleaved[:, :out_len]


def _polyphase_decimate_padded(
        x_flat: torch.Tensor,
        down: int,
        in_len: int,
        out_len: int,
        half_length_factor: int,
        phase: Fraction | None = None
) -> torch.Tensor:
    """The block grid built by materialising the zero-padded signal.

    Retained as the **autograd** path -- :func:`_polyphase_decimate` assembles the identical
    ``phase_sums`` with ``out=`` kernels writing into a preallocated buffer, and autograd
    supports neither ``out=`` nor the ``zero_`` that fills the margins. Same forward-only
    shape as the Numba adaptation kernel. It doubles as the reference the fast path is
    verified against; the two agree to ~1 ULP, not bit-exactly (see the fast path's note).

    ``phase`` folds a gammatone band's modulation into the taps, exactly as on the fast
    path -- see :func:`_polyphase_decimate`. **This route had to be folded too, not routed
    back to the pre-fold formulation.** ``decompose_distortion_components`` really runs
    this branch with grad live (``tests/integration/test_backprop.py``), and keeping the
    old path here would have meant giving the training and inference paths *different
    numerics*, which is the same latent class of defect as the ``hf = 0`` grad/no-grad
    disagreement documented below. One is enough. Folded, the two routes agree to
    **8.14e-17** of peak over the 32 real bands and 3.73e-16 over random complex input at
    ``down`` in {14, 17, 409} -- the same ~1 ULP reassociation the unmodulated pair
    already has, and the assertion that would catch the ``hf = 0`` class of defect.
    """
    taps_per_phase = 2 * half_length_factor + 1
    batch = x_flat.shape[0]
    folded = phase is not None

    num_rows = out_len + 2 * half_length_factor
    left_pad = (half_length_factor + 1) * down - 1
    right_pad = num_rows * down - left_pad - in_len
    if folded:
        # The split exists to keep a complex signal off a real filter; the folded kernel
        # is complex, so only the pad half of `_split_pad_real_imag` is wanted here.
        kernel = _get_modulated_polyphase_kernel(
            1, down, phase, x_flat.real.dtype, x_flat.device, half_length_factor
        )
        # A real signal has to be widened to meet a complex kernel -- `matmul` requires
        # matching dtypes and the result is complex by construction anyway. Free (`.to`
        # returns self) for the complex input the filterbank actually drives.
        padded = F.pad(x_flat.to(kernel.dtype), (left_pad, right_pad))
        is_split = False
    else:
        padded, is_split = _split_pad_real_imag(x_flat, left_pad, right_pad)
        kernel = _get_polyphase_kernel(1, down, padded.dtype, padded.device, half_length_factor)
    blocks = padded.reshape(padded.shape[0], num_rows, down)

    phase_sums = blocks @ kernel.T
    accumulator = phase_sums[:, :out_len, 0]
    for tap_idx in range(1, taps_per_phase):
        accumulator = accumulator + phase_sums[:, tap_idx:tap_idx + out_len, tap_idx]

    if is_split:
        accumulator = _merge_real_imag(accumulator, batch)
    return accumulator


def _polyphase_decimate(
        x_flat: torch.Tensor,
        down: int,
        in_len: int,
        out_len: int,
        half_length_factor: int,
        phase: Fraction | None = None
) -> torch.Tensor:
    r"""FIR-filter then keep every ``down``-th sample, without filtering the discards.

    Here ``n_pre_remove = hf``, so ``y[n] = v[(hf + n)*down]`` of the full convolution.
    Splitting the tap index as ``k = j*down + c`` and re-indexing by ``j' = 2hf - j``
    gives::

        y[n] = sum_{j'} sum_{c} kernel[j', c] * X[n + j', c]

    where ``X[r, c]`` is the input reshaped to ``(rows, down)`` on a grid whose row ``r``
    starts at input index ``r*down - left_pad``, ``left_pad = (hf+1)*down - 1``.

    Written that way it is ``2hf + 1`` matrix-vector products, but they all read the
    same block, so it collapses into a single GEMM plus a cheap gather. Contract the
    block against *all* the phases at once::

        M = X @ kernel^T        # (rows, 2hf+1),  M[r, j] = sum_c X[r, c] * kernel[j, c]
        y[n] = sum_j M[n + j, j]

    The GEMM is ``rows x down x (2hf+1)``, i.e. ``in_len * (2hf+1)`` multiply-adds
    regardless of ``down`` -- against an FFT of the *undecimated* length per band for
    the general path -- and ``M`` is only ``out_len x 21``, so the shifted-diagonal sum
    that follows is negligible.

    **The grid is never materialised.** ``X`` used to be built by zero-padding the whole
    signal, which for ``down = 1229`` copied 146k samples to wrap 121.5k of signal. It is
    almost all pad, and what is not decomposes exactly, with ``n_int = (in_len-1) // down``
    and ``rem = (in_len-1) % down``::

        rows 0 .. hf-1        entirely inside the leading zeros, since the last index of
                              row hf-1 is hf*down - 1 < left_pad for every down
        row  hf               exactly one real sample, x[0], at lane down-1, because
                              hf*down + (down-1) == left_pad by construction
        rows hf+1 .. hf+n_int x[1 : 1 + n_int*down] reshaped -- a *view*, since row hf+1
                              starts at input index (hf+1)*down - left_pad = 1
        row  hf+1+n_int       rem real samples, x[in_len-rem:], at lanes 0 .. rem-1
        rows past that        entirely past the signal

    so the interior is a view, the two ends are one broadcast and one ``rem``-wide GEMM,
    and the rest is zeroed directly; only the ``M`` buffer is allocated. The tail row is
    always in range provided ``right_pad >= (hf-1)*down + 1 > 0``, i.e. provided
    ``hf >= 1``.

    **``hf = 0`` violates that precondition and this function is silently wrong there.**
    An earlier version of this docstring asserted ``hf = 0`` (a one-tap filter design) was
    "not reachable". That was false on both counts, measured 2026-08-17: at ``hf = 0`` this
    fast path does not raise -- it returns finite numbers that disagree with
    :func:`_polyphase_decimate_padded` by **O(1)**, deviations 0.39 to 2.13 across 33 of the
    swept ``(down, in_len, rows, dtype)`` combinations, against 2.22e-16 for ``hf >= 1``.
    The grad path is unaffected because it routes to the padded reference, so the no-grad
    and grad paths disagree at ``hf = 0``.

    **Guarded at the entry point since 2026-08-18**, and deliberately there rather than
    here: ``DecompositionConfiguration.__post_init__`` now raises ``ValueError`` below 1,
    next to the ``segmentation_factor`` check. So the public API no longer reaches this
    case. This function is still *internally* wrong below 1 and the precondition above is
    still the real one -- a direct caller passing ``half_length_factor=0`` bypasses the
    dataclass entirely -- which is why the analysis is kept rather than deleted.

    Worth keeping as a general point: ``__post_init__`` validated exactly one field of
    several for a long time, and a validator that checks one field reads as though it
    checks all of them. "Nothing in the library passes that" was true, and was a statement
    about the library rather than about the API.

    Dropping lanes from the two end contractions drops exact zeros only, but ``mm`` over
    ``(rows*num_rows, down)`` becoming ``rows`` ``bmm``\ s over ``(n_int, down)`` changes
    only the GEMM's ``M``, not the contraction length or the lane order -- so BLAS may
    block it differently. That makes this a **reassociation, ~1 ULP, not bit-identical**;
    do not assert ``torch.equal`` on it. Measured against
    :func:`_polyphase_decimate_padded` over real and complex input, 1-3 rows, 13 lengths
    from 1 to 121500 and 12 decimation rates up to 1229 (936 cases, 284 of them still
    bit-identical): worst deviation 4.44e-16 **absolute**, i.e. 2 ULP at the unit input
    scale. A second sweep over ``hf`` in {1,2,3,5,10} and three input stride layouts agreed,
    and a re-run on the rates and shapes the filterbank actually drives worst-cased at
    2.22e-16 absolute. Note the bound here is stated in *absolute* terms while
    :func:`_polyphase_mixed`'s is *relative to the output peak* -- the two are not
    interchangeable and the mixed path's absolute figure is several times this one, so do
    not carry either number across. The ``out=`` kernels make it forward-only.

    **``phase`` folds a gammatone band's demodulation into the taps.** Passing the band's
    exact ``fc/fs`` (:func:`_modulation_phase`) makes this compute
    ``decimate(x * exp(-2j*pi*fc*n/fs))`` up to the ``out_len``-long residual that
    :func:`fast_demodulate_decimate_torch` applies -- one complex zgemm instead of a
    full-rate 61 MB modulation pass, a ``_split_real_imag`` gather, a real dgemm at twice
    the rows and a ``_merge_real_imag`` allocation. The GEMM's flops double (complex-by-
    complex is 4 real multiplies where complex-by-real split is 2), which the FLOP ledger
    says should lose; it does not, because ``N = 21`` is far too skinny for the GEMM to be
    flop-bound and the terms removed are streaming passes over blocks that do not fit in
    cache. The rejected alternative -- keeping the folded kernel on the split path as a
    widened ``N = 42`` real dgemm -- measured **0.717x**.

    ``phase=None`` is today's behaviour byte for byte; the real and unmodulated-complex
    callers never take the folded branch.
    """
    if torch.is_grad_enabled() and x_flat.requires_grad:
        return _polyphase_decimate_padded(
            x_flat, down, in_len, out_len, half_length_factor, phase
        )

    taps_per_phase = 2 * half_length_factor + 1
    batch = x_flat.shape[0]
    folded = phase is not None
    is_split = x_flat.is_complex() and not folded
    rows = _split_real_imag(x_flat) if is_split else x_flat
    num_rows_signal = rows.shape[0]
    if folded:
        # Design in the *real* dtype and promote through the phase vector: handing
        # `get_resample_filter_torch` a complex dtype silently caches a zero-imaginary
        # duplicate of every filter (see `fast_resample_poly_torch`).
        kernel_t = _get_modulated_polyphase_kernel(
            1, down, phase, rows.real.dtype, rows.device, half_length_factor
        ).T
        # `bmm` will not mix dtypes and `new_empty` below takes the *signal's*, so a real
        # signal against the complex kernel has to be widened here. `.to` returns the same
        # object when the dtypes already agree, which is the only case the filterbank
        # reaches (`GammatoneAnalyzerTorch.process` hands over complex128).
        rows = rows.to(kernel_t.dtype)
    else:
        kernel_t = _get_polyphase_kernel(
            1, down, rows.dtype, rows.device, half_length_factor
        ).T                                                 # (down, taps_per_phase)

    num_rows = out_len + 2 * half_length_factor
    n_interior = (in_len - 1) // down                       # fully-real grid rows
    remainder = (in_len - 1) % down                         # real lanes in the tail row

    phase_sums = rows.new_empty(num_rows_signal, num_rows, taps_per_phase)
    phase_sums[:, :half_length_factor].zero_()
    # Row hf sees x[0] alone, at lane down-1: a broadcast, not a GEMM over `down` zeros.
    torch.mul(rows[:, :1], kernel_t[down - 1], out=phase_sums[:, half_length_factor])

    first = half_length_factor + 1
    if n_interior:
        interior = rows[:, 1:1 + n_interior * down].reshape(
            num_rows_signal, n_interior, down
        )
        # `torch.matmul(..., out=)` is a trap here: when the interior slice covers whole
        # rows it takes the foldable `mm` path, tries to view the strided `out` as 2-D and
        # raises. `bmm` against a stride-0 expanded kernel is what matmul feeds bmm anyway.
        torch.bmm(
            interior,
            kernel_t.unsqueeze(0).expand(num_rows_signal, down, taps_per_phase),
            out=phase_sums[:, first:first + n_interior],
        )
    tail = first + n_interior
    if remainder:
        torch.bmm(
            rows[:, in_len - remainder:].unsqueeze(1),
            kernel_t[:remainder].unsqueeze(0).expand(
                num_rows_signal, remainder, taps_per_phase),
            out=phase_sums[:, tail:tail + 1],
        )
        tail += 1
    phase_sums[:, tail:].zero_()

    accumulator = phase_sums[:, :out_len, 0]
    for tap_idx in range(1, taps_per_phase):
        accumulator = accumulator + phase_sums[:, tap_idx:tap_idx + out_len, tap_idx]

    if is_split:
        accumulator = _merge_real_imag(accumulator, batch)
    return accumulator


def _polyphase_mixed_padded(
        x_flat: torch.Tensor,
        up: int,
        down: int,
        in_len: int,
        out_len: int,
        half_length_factor: int
) -> torch.Tensor:
    """The block grid built by materialising the offset-padded signal.

    Autograd path and verification reference for :func:`_polyphase_mixed`, for the same
    reason as :func:`_polyphase_decimate_padded`. Also the path taken for inputs shorter
    than a couple of blocks, where the leading and trailing partial blocks would coincide.
    """
    batch = x_flat.shape[0]
    # `.real` is a view and returns `self` for a real tensor, so this is the real dtype the
    # kernel is designed in either way -- fetched before the pad because it supplies
    # `offset`, which is one of the pad widths.
    kernel, taps, offset = _get_mixed_polyphase_kernel(
        up, down, x_flat.real.dtype, x_flat.device, half_length_factor
    )

    num_groups = -(-out_len // up)
    # Enough blocks for the last window (num_groups + taps - 1), and never fewer than the
    # input itself occupies -- the diagonal sum only ever reads the first of the two.
    num_blocks = max(num_groups + taps - 1, -(-(in_len + offset) // down))
    padded, is_complex = _split_pad_real_imag(
        x_flat, offset, num_blocks * down - offset - in_len
    )
    blocks = padded.reshape(padded.shape[0], num_blocks, down)

    phase_sums = (blocks @ kernel).reshape(padded.shape[0], num_blocks, up, taps)
    accumulator = phase_sums[:, :num_groups, :, 0]
    for tap_idx in range(1, taps):
        accumulator = accumulator + phase_sums[:, tap_idx:tap_idx + num_groups, :, tap_idx]

    interleaved = accumulator.reshape(padded.shape[0], num_groups * up)[:, :out_len]
    if is_complex:
        interleaved = _merge_real_imag(interleaved, batch)
    return interleaved


def _polyphase_mixed(
        x_flat: torch.Tensor,
        up: int,
        down: int,
        in_len: int,
        out_len: int,
        half_length_factor: int
) -> torch.Tensor:
    r"""General ``up/down`` rate change as one GEMM plus a shifted-diagonal sum.

    Reshaping the (offset) input to ``B[t, c] = x[t*D + c - offset]`` and contracting it
    against the kernel of :func:`_get_mixed_polyphase_kernel` gives every residue's every
    block-shift at once::

        P[s, r*taps + t] = sum_c B[s, c] * kernel[c, r*taps + t]
        y[m*U + r]       = sum_t P[m + t, r*taps + t]

    -- the same collapse the pure-decimation path uses, which is exactly this form at
    ``U = 1``. The GEMM is ``num_blocks x D x (U*taps)``, i.e. ``~in_len * U * taps``
    multiply-adds where the FFT route transforms ``in_len * U`` points three times, and
    the diagonal sum that follows is ``taps`` slice-adds over an ``out_len``-sized array.

    The trailing ``[:, :out_len]`` drops the residues of the final group that run past the
    requested length; SciPy's ``ceil(in_len*up/down)`` need not be a multiple of ``U``.

    ``B`` is a **view**, not a padded copy. ``offset`` is only tens of samples (10 at 3/2,
    15 at 2/3) but padding for it copied the entire signal. With ``lead = offset // D`` and
    ``lead_zeros = offset % D``, blocks below ``lead`` and above ``(in_len + offset) // D``
    are exactly zero; at most one block at each end is part signal and part zero, and
    dropping the zero lanes from *its* contraction drops exact zeros only. At both rates
    the filterbank uses, ``offset % D == 0`` (10 % 2, 15 % 3), so there are no partial
    blocks at all and the interior view is the whole signal.

    Sizing ``num_blocks`` no longer has to cover the input, only what the diagonal sum
    reads, which is ``num_groups + taps - 1`` blocks; where the old ``max(...)`` bound, the
    padded form was computing and discarding blocks. ``last`` is therefore clamped to the
    grid: the input may occupy blocks nothing ever reads.

    Same numerical class as :func:`_polyphase_decimate` -- a reassociation of the GEMM's
    ``M``, **~1 ULP, not bit-identical**, so do not assert ``torch.equal`` on it. Measured
    against :func:`_polyphase_mixed_padded` over 7 mixed rates from 3/2 to 1000/3, real and
    complex, 1-3 rows and 13 lengths: worst deviation 4.4e-16 **relative to the output
    peak**, 2 ULP.

    That bound is *relative*, and mixing it up with :func:`_polyphase_decimate`'s *absolute*
    4.44e-16 is an easy mistake to make. Re-measured 2026-08-17 including rate 147/160, which
    the original sweep did not cover: worst **relative** 3.39e-16 (bound holds), but worst
    **absolute** 1.33e-15 -- roughly 3x the decimate figure, because the output peak here is
    ~4 rather than ~1. Quote whichever one you actually need and say which it is.
    Forward-only for the same ``out=`` reason.
    """
    batch = x_flat.shape[0]
    kernel, taps, offset = _get_mixed_polyphase_kernel(
        up, down, x_flat.real.dtype, x_flat.device, half_length_factor
    )

    # Below two blocks past the offset the leading and trailing partial blocks would be the
    # same block, which the algebra below does not model. Both terms come from the cached
    # geometry, so this is a derived correctness bound, not a tuning knob -- it trips only
    # at in_len <= 14 for 3/2 and in_len <= 21 for 2/3, where there is nothing to save.
    if (torch.is_grad_enabled() and x_flat.requires_grad) or in_len <= offset + 2 * down:
        return _polyphase_mixed_padded(x_flat, up, down, in_len, out_len, half_length_factor)

    is_complex = x_flat.is_complex()
    rows = _split_real_imag(x_flat) if is_complex else x_flat
    num_rows_signal = rows.shape[0]

    num_groups = -(-out_len // up)
    num_blocks = num_groups + taps - 1
    lead_zeros = offset % down                   # zero lanes in the first block with data
    lead_block = offset // down
    first = lead_block + (1 if lead_zeros else 0)
    last = min((in_len + offset) // down - 1, num_blocks - 1)
    remainder = (in_len + offset) % down
    columns = up * taps

    phase = rows.new_empty(num_rows_signal, num_blocks, columns)
    phase[:, :min(lead_block, num_blocks)].zero_()
    if lead_zeros and lead_block < num_blocks:
        # `down - lead_zeros <= down < in_len` by the guard above, so this never overruns.
        lanes = down - lead_zeros
        torch.bmm(
            rows[:, :lanes].unsqueeze(1),
            kernel[lead_zeros:].unsqueeze(0).expand(num_rows_signal, lanes, columns),
            out=phase[:, lead_block:lead_block + 1],
        )
    n_interior = last - first + 1
    tail = first
    if n_interior > 0:
        start = first * down - offset
        interior = rows[:, start:start + n_interior * down].reshape(
            num_rows_signal, n_interior, down
        )
        torch.bmm(
            interior,
            kernel.unsqueeze(0).expand(num_rows_signal, down, columns),
            out=phase[:, first:first + n_interior],
        )
        tail = first + n_interior
    if remainder and tail < num_blocks:
        torch.bmm(
            rows[:, in_len - remainder:].unsqueeze(1),
            kernel[:remainder].unsqueeze(0).expand(num_rows_signal, remainder, columns),
            out=phase[:, tail:tail + 1],
        )
        tail += 1
    phase[:, tail:].zero_()

    phase_sums = phase.reshape(num_rows_signal, num_blocks, up, taps)
    accumulator = phase_sums[:, :num_groups, :, 0]
    for tap_idx in range(1, taps):
        accumulator = accumulator + phase_sums[:, tap_idx:tap_idx + num_groups, :, tap_idx]

    interleaved = accumulator.reshape(num_rows_signal, num_groups * up)[:, :out_len]
    if is_complex:
        interleaved = _merge_real_imag(interleaved, batch)
    return interleaved


def _fft_resample(
        x_flat: torch.Tensor,
        up: int,
        down: int,
        n_pre_remove: int,
        filter_length: int,
        out_len: int,
        original_dtype: torch.dtype,
        half_length_factor: int
) -> torch.Tensor:
    """Zero-insert, FIR-filter by FFT linear convolution, decimate. The general fallback.

    Kept reachable for rates whose polyphase intermediate would be disproportionate (see
    ``MIXED_POLYPHASE_MAX_ELEMENTS``), and as the independent second implementation the
    polyphase paths are cross-checked against. FFT is used rather than
    conv1d/conv_transpose1d because torch has no optimized float64 convolution kernel --
    those fall back to `slow_conv2d`/`slow_conv_transpose2d`, which dominated the double
    precision decomposition. It is ~2x faster and bit-identical to the conv path
    (verified to ~1e-15), and fully differentiable.
    """
    batch, in_len = x_flat.shape

    # 1. Zero-insertion by up. Done via pad+reshape (differentiable, no in-place
    #    scatter): each sample is followed by (up - 1) zeros.
    if up > 1:
        upsampled = F.pad(x_flat.unsqueeze(-1), (0, up - 1)).reshape(batch, in_len * up)
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
        up, down, original_dtype, x_flat.device, half_length_factor, fft_length
    )
    if upsampled.is_complex():
        spectrum = torch.fft.fft(upsampled, n=fft_length, dim=-1) * filter_spectrum
        filtered = torch.fft.ifft(spectrum, n=fft_length, dim=-1)
    else:
        spectrum = torch.fft.rfft(upsampled, n=fft_length, dim=-1) * filter_spectrum
        filtered = torch.fft.irfft(spectrum, n=fft_length, dim=-1)

    # 3. Decimate by down and crop the centered out_len window (matches SciPy's
    #    zero-phase offset via n_pre_remove).
    decimated = filtered[:, ::down]
    end = n_pre_remove + out_len
    if decimated.shape[-1] < end:
        decimated = F.pad(decimated, (0, end - decimated.shape[-1]))
    return decimated[:, n_pre_remove:end]


# Above this many elements in the mixed-rate GEMM's ``(rows, num_blocks, up, taps)``
# product, fall back to the FFT convolution. The polyphase intermediate is ``taps`` copies
# of the output, and ``taps`` is bounded by ``2*half_length_factor + 2``, so this only
# trips on rate/length combinations whose *output* is already near a gigabyte -- the FFT
# route's own working set (~4x the zero-inserted signal) is comparable there, and it
# builds it in fewer, larger allocations. 2**26 elements is 512 MB in float64.
MIXED_POLYPHASE_MAX_ELEMENTS = 1 << 26


def _mixed_polyphase_fits(
        up: int,
        down: int,
        out_len: int,
        batch: int,
        is_complex: bool,
        half_length_factor: int
) -> bool:
    """Whether the mixed-rate GEMM's intermediate stays under the fallback threshold.

    Sizes ``(rows, num_blocks, up, taps)`` from ``num_blocks * up ~ out_len``, which is
    the block count :func:`_polyphase_mixed` derives exactly; the block-edge slack it adds
    is ``up * taps`` and irrelevant at this threshold.
    """
    taps = _mixed_polyphase_geometry(up, down, half_length_factor)[3]
    rows = 2 * batch if is_complex else batch
    return rows * out_len * taps <= MIXED_POLYPHASE_MAX_ELEMENTS


def fast_resample_poly_torch(
        x: torch.Tensor,
        up: int,
        down: int,
        axis: int = -1,
        half_length_factor: int = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
) -> torch.Tensor:
    """
    Native PyTorch polyphase resampler replicating SciPy's ``resample_poly``:
    zero-insert by ``up``, FIR-filter, decimate by ``down``.

    Every rate takes a polyphase GEMM: pure interpolation, pure decimation, and — since
    the mixed-rate work landed — general ``up/down`` as well. The three are separate
    routines rather than one general one because the specialisations are strictly better
    at their own rate: at ``down == 1`` the general form degenerates to a rank-1 update
    with a ``2*half_length_factor + 1``-fold intermediate, where
    :func:`_polyphase_interpolate` does the identical FLOPs as a dense
    ``(in_len, 21) @ (21, up)`` GEMM.

    Why this matters over the FFT convolution it replaced: the FFT works at the
    *undecimated* length regardless of the rate. For a band decimated by 1229 it
    transformed a 121500-point spectrum to produce 98 output samples, and for the
    matching synthesis upsample it transformed 121500 points to filter 327. It also
    parallelises over a batch dimension the filterbank does not have — 32 bands with 32
    *distinct* decimation factors group into 32 runs of one or two rows. The polyphase
    form instead does the same arithmetic in a handful of taps per output sample.

    The FIR is real, so the mixed-rate arithmetic is a genuine reassociation of the FFT
    route's, not an approximation: measured worst relative deviation 1.3e-15 over rates
    from 3/2 to 1000/3, and *closer* to SciPy than the FFT route is (the direct 21-tap
    sum beats a transform over ~120k points). :func:`_fft_resample` remains reachable for
    the pathological sizes described at ``MIXED_POLYPHASE_MAX_ELEMENTS``.
    """
    if up == down:
        return x

    # The reduced rates are all the three polyphase branches need, and they come straight
    # from `math.gcd` -- which is verbatim how `get_resample_filter_torch` derives them, so
    # this is a textual substitution, not a numerical one. Fetching the designed filter
    # here instead cached a *second*, complex, copy of a filter that is real by
    # construction for every rate the analytic subbands drive (64 duplicate entries,
    # 2.5 MB, on one 16 kHz mono decomposition) purely to read its length, and only the
    # FFT fallback below has any use for it: the polyphase routines design their own
    # kernels in the real split dtype and fold `n_pre_remove` into their index algebra.
    divisor = math.gcd(up, down)
    up_reduced, down_reduced = up // divisor, down // divisor

    in_len = x.shape[axis]
    out_len = math.ceil(in_len * up_reduced / down_reduced)

    x_moved = x.transpose(axis, -1)
    shape_prefix = x_moved.shape[:-1]
    x_flat = x_moved.reshape(-1, in_len)
    batch = x_flat.shape[0]

    if down_reduced == 1:
        y_flat = _polyphase_interpolate(x_flat, up_reduced, in_len, out_len, half_length_factor)
    elif up_reduced == 1:
        y_flat = _polyphase_decimate(x_flat, down_reduced, in_len, out_len, half_length_factor)
    elif _mixed_polyphase_fits(up_reduced, down_reduced, out_len, batch, x.is_complex(),
                               half_length_factor):
        y_flat = _polyphase_mixed(x_flat, up_reduced, down_reduced, in_len, out_len,
                                  half_length_factor)
    else:
        # The only branch that wants the designed taps, and it wants them in the *input's*
        # dtype: `_get_resample_filter_spectrum` picks `fft` vs `rfft` off
        # `h_padded.is_complex()`, which has to match the signal branch inside
        # `_fft_resample`. Designing it here keeps the complex copy off every other rate.
        h_padded, _, _, n_pre_remove = get_resample_filter_torch(
            up, down, x.dtype, x.device, half_length_factor
        )
        y_flat = _fft_resample(x_flat, up_reduced, down_reduced, n_pre_remove,
                               h_padded.shape[0], out_len, x.dtype, half_length_factor)

    return y_flat.reshape(*shape_prefix, out_len).transpose(axis, -1)


def fast_demodulate_decimate_torch(
        x: torch.Tensor,
        down: int,
        center_frequency_hz: float,
        sampling_frequency_hz: float,
        axis: int = -1,
        half_length_factor: int = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
) -> torch.Tensor:
    """``fast_resample_poly_torch(x * exp(-2j*pi*fc*n/fs), 1, down)``, folded into the taps.

    Demodulates a gammatone band to baseband and decimates it, without ever materialising
    the full-rate modulation. The modulation rides the 21 polyphase taps
    (:func:`_get_modulated_polyphase_kernel`) and only a length-``out_len`` residual
    survives, at the decimated rate.

    The equivalence holds at the *signal's own* precision: the modulation is taken in
    ``_complex_dtype_for(x.real.dtype)``, so complex64 in gives complex64 out and a real
    input widens to the matching complex dtype rather than to complex128. (The phase is
    always reduced in float64 and rounded once -- reducing in the signal's precision would
    defeat the reduction.) An earlier revision built the kernel and the residual in
    complex128 unconditionally while sizing the accumulator off the signal, which made the
    folded path silently complex128-only: complex64 raised "Expected out tensor to have
    dtype c10::complex<double>", real float64 raised "result type ComplexDouble can't be
    cast to Double", and both gradient routes raised "expected m1 and m2 to have the same
    dtype". Unreachable from the filterbank -- ``GammatoneAnalyzerTorch.process`` hands
    over complex128 -- but the contract above is unconditional, so it had to be.

    Deliberately a new entry point rather than a keyword on
    :func:`fast_resample_poly_torch`: that function has six library call sites and 20+ in
    the tests, and widening it would give it a contract that silently admits mixed rates
    the fold cannot serve. This one's contract is the single line above, and
    ``.scratch/p5-2026-08-17/parity_analysis.py`` checks exactly that, per band, against
    ``_polyphase_decimate(band * exact_mod_row, ...)`` -- where ``exact_mod_row`` is built
    by an *independent* exact-``Fraction`` reduction in Python integers, sharing no code
    with this module: worst **1.219e-15** of peak over all 32 bands at all four batch
    geometries the pipeline drives (mono 2/1 rows, stereo 6/2 rows). Building that
    reference out of :func:`_modulation_phase` instead turns the check into a
    self-consistency test and hides any error in the ratio itself -- which is exactly how
    a 3.4373e-12 error on the base-frequency band passed a green suite once already.

    Against *unmodified* HEAD -- whose modulation matrix built its exponential on the raw
    argument -- the same comparison is **1.128e-11**, and that is the matrix being wrong,
    not this. See :func:`_reduced_phase_exp`; the fold moves the output toward truth.

    What it deletes, measured on the 16 kHz mono reference geometry: the ``(32, 120000)``
    complex128 analysis modulation matrix (**61.44 MB** resident) and the full-rate pass
    over it, against +1.298 MB of modulated kernels and +1.284 MB of residual vectors for
    all 32 bands. The split/merge round trip goes with it on this path.
    """
    phase = _modulation_phase(center_frequency_hz, sampling_frequency_hz)
    in_len = x.shape[axis]
    x_moved = x.transpose(axis, -1)
    shape_prefix = x_moved.shape[:-1]
    x_flat = x_moved.reshape(-1, in_len)
    # The whole path runs at the signal's own precision -- a complex64 band stays
    # complex64 rather than being silently widened by a complex128 modulation, and a real
    # band widens to the matching complex dtype because the result is complex either way.
    dtype = _complex_dtype_for(x_flat.real.dtype)

    if down == 1:
        # Contract guard, not a hot path. `fast_resample_poly_torch` returns `x` unchanged
        # at `up == down`, and the modulation used to be a separate pass, so `down == 1`
        # has to come back as `x * m` at the full rate. `decimations` is `clamp(..., min=1)`
        # and the measured minimum across every supported input rate is 13..15, so this is
        # unreachable from the filterbank; it is here so the docstring's equivalence is
        # true for every `down`.
        out_len = in_len
        y_flat = x_flat * _get_modulation_vector(
            phase, -1, 0, 1, in_len, x_flat.device, dtype
        )
    else:
        out_len = math.ceil(in_len / down)
        y_flat = _polyphase_decimate(x_flat, down, in_len, out_len, half_length_factor, phase)
        # The surviving residual: m[(hf+i)*down], indexed by the full-rate sample each
        # decimated output came from. The `hf` offset is `n_pre_remove`; dropping it is a
        # per-band constant rotation that no shape check and no existing test would catch.
        y_flat = y_flat * _get_modulation_vector(
            phase, -1, half_length_factor * down, down, out_len, x_flat.device, dtype
        )

    return y_flat.reshape(*shape_prefix, out_len).transpose(axis, -1)


def fast_interpolate_modulate_torch(
        x: torch.Tensor,
        up: int,
        center_frequency_hz: float,
        sampling_frequency_hz: float,
        axis: int = -1,
        half_length_factor: int = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
) -> torch.Tensor:
    """``fast_resample_poly_torch(x, up, 1) * exp(+2j*pi*fc*n/fs)``, folded into the taps.

    The mirror of :func:`fast_demodulate_decimate_torch`: interpolates a decimated
    gammatone band back to the full rate and re-modulates it to its centre frequency,
    without ever materialising the full-rate modulation. The re-modulation rides the 21
    polyphase taps and what survives is a length-``in_len`` pre-multiply on the *decimated
    input*, applied here before the GEMM.

    **The signs are not symmetric with the analysis half and must not be assumed.**
    Analysis demodulates with ``exp(-2j*pi*fc*n/fs)`` and its taps carry the *conjugate*
    ``exp(+2j*pi*fc*k/fs)``; synthesis re-modulates with ``exp(+2j*pi*fc*n/fs)`` and its
    taps carry that same sign *unconjugated*, offset by ``-n_pre_remove = -hf*up``, so
    both sides end up asking :func:`_get_modulated_polyphase_kernel` for ``sign = +1`` and
    differ only in the start offset. A sign slip is O(1); a dropped offset is a per-band
    constant rotation with no shape change, which is why the bar below is per band.

    **The synthesizer's phase factor is deliberately NOT folded in here.** It stays a
    full-rate per-band complex scalar, which is exactly
    ``GammatoneSynthesizerTorch.process``'s existing ``alignment=None`` default -- an
    already-tested path that needs no new code. Folding it into the taps as well would
    delete that scalar pass, but ``phase_factors`` is only unit modulus to ~1.3e-11
    (``gammatone.py`` normalises ``slopes``, not the reciprocal) whereas the fold's
    ``m[a-b] = m[a]*conj(m[b])`` identity needs exact unit modulus, and the variant is
    unmeasured. Do not take it without a measurement.

    Runs at the signal's own precision, for the same reason and by the same mechanism as
    :func:`fast_demodulate_decimate_torch`: the phase is reduced in float64 and rounded
    once into ``_complex_dtype_for(x.real.dtype)``.

    What it deletes, measured on the 16 kHz mono reference geometry: the ``(32, 120336)``
    complex128 fused modulation-times-phase alignment matrix (**61.61 MB** resident, built
    once and streamed four times, once per component sweep) and the full-rate pass over
    it, against +1.298 MB of modulated kernels and +1.284 MB of pre-multiply vectors for
    all 32 bands. The split/merge round trip goes with it on this path -- 128 interpolation
    calls per decomposition.

    Per-band parity against ``_polyphase_interpolate(sub, U, ...) * exact_m_s_row`` with
    the reference row built by an *independent* exact-``Fraction`` reduction sharing no
    code with this module: see ``.scratch/p5-2026-08-17/parity_synthesis.py``.
    """
    phase = _modulation_phase(center_frequency_hz, sampling_frequency_hz)
    in_len = x.shape[axis]
    x_moved = x.transpose(axis, -1)
    shape_prefix = x_moved.shape[:-1]
    x_flat = x_moved.reshape(-1, in_len)
    dtype = _complex_dtype_for(x_flat.real.dtype)

    if up == 1:
        # Contract guard, not a hot path -- the mirror of the `down == 1` guard above, and
        # unreachable from the filterbank for the same reason (`decimations` is clamped at
        # 1 and its measured minimum is 13..15 across every supported input rate).
        out_len = in_len
        return (x_flat * _get_modulation_vector(phase, 1, 0, 1, in_len, x_flat.device, dtype)
                ).reshape(*shape_prefix, out_len).transpose(axis, -1)

    out_len = in_len * up
    # The surviving pre-multiply: m[up*i] on the decimated input index i. Indices outside
    # [0, in_len) are the `F.pad` zeros the interpolation adds, so they need no modulation.
    x_flat = x_flat.to(dtype) * _get_modulation_vector(
        phase, 1, 0, up, in_len, x_flat.device, dtype
    )
    y_flat = _polyphase_interpolate(x_flat, up, in_len, out_len, half_length_factor, phase)

    return y_flat.reshape(*shape_prefix, out_len).transpose(axis, -1)
