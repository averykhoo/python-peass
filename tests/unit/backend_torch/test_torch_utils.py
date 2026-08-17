"""Tests for the PyTorch polyphase resampler.

`fast_resample_poly_torch` dispatches on the reduced rate: pure interpolation, pure
decimation and general mixed rates each take their own polyphase GEMM, and only
pathologically sized intermediates fall back to the FFT linear convolution. The
filterbank drives it almost entirely through the first two -- 32 bands with 32 distinct
decimation factors -- while the decomposition's front and back ends (3/2 up, 2/3 down)
drive the mixed one. All of them are checked here against SciPy's `resample_poly`, which
is the reference the whole resampler is written to replicate, and the mixed path is
additionally cross-checked against the FFT route it replaced, which is an independent
implementation of the same convolution.
"""
import math
from fractions import Fraction

import numpy as np
import pytest
import torch
from scipy.signal import resample_poly

from peass.backend_torch import utils as torch_utils
from peass.backend_torch.utils import DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
from peass.backend_torch.utils import fast_resample_poly_torch

# (up, down): pure interpolation, pure decimation, and mixed rates.
# 368 and 1229 are real filterbank decimation factors for a 24 kHz-class analysis rate.
RATE_CASES = [(3, 2), (2, 3), (2, 1), (1, 2), (7, 1), (1, 3), (16, 1), (1, 58),
              (368, 1), (1, 368), (1, 1229)]

# Mixed rates only -- both sides > 1 after reduction, so these exercise `_polyphase_mixed`.
# 3/2 and 2/3 are the decomposition's own front and back ends; 320/441 and 147/160 are
# real audio sample-rate conversions; 99/98 is the degenerate `taps == 1` geometry; the
# 1000/3 pair pushes the branch count and the per-phase tap count to opposite extremes.
MIXED_RATE_CASES = [(3, 2), (2, 3), (5, 3), (3, 5), (7, 4), (4, 7), (9, 8), (8, 9),
                    (16, 3), (3, 16), (441, 320), (320, 441), (147, 160), (160, 147),
                    (99, 98), (98, 99), (1000, 3), (3, 1000), (6, 4)]


def _scipy_reference(x: np.ndarray, up: int, down: int) -> np.ndarray:
    return resample_poly(x, up, down, window=('kaiser', 5.0), padtype='constant')


def _exact_modulation(center_frequency: float, sampling_frequency: float,
                      length: int, sign: int = -1) -> torch.Tensor:
    """`exp(sign*2j*pi*fc*n/fs)` reduced in exact Python integers, sharing no library code.

    `Fraction(float)` is exact, `(num*n) % den` is exact in arbitrary-precision integers
    and `int / int` is correctly rounded by Python, so this is an independent reference
    for the *phase* as well as for the fold's algebra. Building the reference out of
    `_modulation_phase` instead makes any error in the library's own ratio cancel on both
    sides -- the trap that let a 3.4373e-12 error on the base-frequency band pass a green
    suite.

    `sign = -1` is the analysis demodulation, `sign = +1` the synthesis re-modulation; the
    two are conjugates, and the halves of P5 they check are not symmetric (the analysis
    taps carry the conjugate of the modulation, the synthesis taps carry it directly with
    a `-hf*up` offset), so both are driven explicitly rather than derived from each other.
    """
    ratio = Fraction(center_frequency) / Fraction(sampling_frequency)
    numerator, denominator = ratio.numerator % ratio.denominator, ratio.denominator
    angles = [sign * 2.0 * math.pi * (((numerator * n) % denominator) / denominator)
              for n in range(length)]
    return torch.tensor([complex(math.cos(a), math.sin(a)) for a in angles],
                        dtype=torch.complex128)


def _fft_route(x: torch.Tensor, up: int, down: int) -> torch.Tensor:
    """The FFT linear convolution, called directly on a ``(batch, n)`` tensor.

    Independent of the polyphase index algebra -- it zero-inserts, convolves the whole
    signal and strides the result -- so it is the natural second opinion on the mixed
    path, at a tolerance the SciPy comparison is far too loose to pin.
    """
    divisor = math.gcd(up, down)
    half = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
    h_padded, up_r, down_r, n_pre_remove = torch_utils.get_resample_filter_torch(
        up, down, x.dtype, x.device, half)
    out_len = math.ceil(x.shape[-1] * (up // divisor) / (down // divisor))
    return torch_utils._fft_resample(x, up_r, down_r, n_pre_remove, h_padded.shape[0],
                                     out_len, x.dtype, half)


def _assert_close_relative(got: np.ndarray, reference: np.ndarray, tolerance: float) -> None:
    """Compare relative to the reference's peak -- the convention the repo reports in."""
    overlap = min(got.shape[-1], reference.shape[-1])
    scale = max(float(np.max(np.abs(reference))), 1.0)
    np.testing.assert_allclose(got[..., :overlap], reference[..., :overlap],
                               rtol=0.0, atol=tolerance * scale)


@pytest.mark.unit
@pytest.mark.parametrize("up, down", RATE_CASES)
@pytest.mark.parametrize("in_len", [327, 5000])
def test_torch_resample_matches_scipy(up, down, in_len):
    if in_len < down:
        pytest.skip("Decimation factor exceeds the signal length.")
    signal = np.random.default_rng(0).standard_normal(in_len)

    got = fast_resample_poly_torch(torch.tensor(signal).unsqueeze(0), up, down)[0].numpy()
    reference = _scipy_reference(signal, up, down)

    assert got.shape[0] == math.ceil(in_len * up / down)
    overlap = min(got.shape[0], reference.shape[0])
    np.testing.assert_allclose(got[:overlap], reference[:overlap], rtol=0.0,
                               atol=1e-12 * max(np.max(np.abs(reference)), 1.0))


@pytest.mark.unit
@pytest.mark.parametrize("up, down", [(368, 1), (1, 368), (16, 1), (1, 58)])
def test_torch_resample_complex_matches_scipy_componentwise(up, down):
    """Complex input is filtered by a real FIR, so it must equal the two real runs.

    The GEMM path splits complex signals into real and imaginary rows precisely so the
    arithmetic stays real; this pins that the split is a pure optimization.
    """
    rng = np.random.default_rng(1)
    real_part = rng.standard_normal(5000)
    imag_part = rng.standard_normal(5000)

    got = fast_resample_poly_torch(
        torch.tensor(real_part + 1j * imag_part).unsqueeze(0), up, down
    )[0].numpy()

    for component, reference_input in ((got.real, real_part), (got.imag, imag_part)):
        reference = _scipy_reference(reference_input, up, down)
        overlap = min(component.shape[0], reference.shape[0])
        np.testing.assert_allclose(component[:overlap], reference[:overlap], rtol=0.0,
                                   atol=1e-12 * max(np.max(np.abs(reference)), 1.0))


@pytest.mark.unit
@pytest.mark.parametrize("up, down", MIXED_RATE_CASES)
@pytest.mark.parametrize("in_len", [327, 1200])
@pytest.mark.parametrize("is_complex", [False, True])
def test_torch_mixed_rate_matches_scipy(up, down, in_len, is_complex):
    """The general up/down polyphase form against the reference it replicates.

    Complex input is carried through the same real GEMM as two stacked rows, so it is
    checked against SciPy component by component -- the FIR is real, and a complex signal
    against a real filter must equal the two real runs exactly.
    """
    rng = np.random.default_rng(in_len)
    signal = rng.standard_normal(in_len)
    if is_complex:
        signal = signal + 1j * rng.standard_normal(in_len)

    got = fast_resample_poly_torch(torch.tensor(signal).unsqueeze(0), up, down)[0].numpy()

    divisor = math.gcd(up, down)
    assert got.shape[0] == math.ceil(in_len * (up // divisor) / (down // divisor))
    parts = ((got.real, signal.real), (got.imag, signal.imag)) if is_complex else ((got, signal),)
    for component, reference_input in parts:
        _assert_close_relative(component, _scipy_reference(reference_input, up, down), 1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("up, down", MIXED_RATE_CASES)
@pytest.mark.parametrize("in_len", [40, 327, 1200])
@pytest.mark.parametrize("is_complex", [False, True])
def test_torch_mixed_rate_matches_fft_route(up, down, in_len, is_complex):
    """Polyphase must reassociate the FFT route's arithmetic, not approximate it.

    This is the tight check: the two routes share only the filter design, so agreement at
    1e-13 of peak (measured worst 1.3e-15) says the index algebra is right, not merely
    close. The pure-rate GEMMs cleared 2.3e-15 on the same style of sweep.
    """
    rng = np.random.default_rng(in_len + up)
    signal = rng.standard_normal((2, in_len))
    if is_complex:
        signal = signal + 1j * rng.standard_normal((2, in_len))
    tensor = torch.tensor(signal)

    got = fast_resample_poly_torch(tensor, up, down).numpy()
    _assert_close_relative(got, _fft_route(tensor, up, down).numpy(), 1e-13)


@pytest.mark.unit
@pytest.mark.parametrize("up, down", [(3, 2), (2, 3), (441, 320)])
def test_torch_mixed_rate_batched_matches_one_dimensional(up, down):
    """A 1-D signal, a batch row and a transposed view must all agree.

    Not bit-exact: BLAS blocks a 1-row GEMM differently from a 3-row one, which reorders
    the accumulation by an ULP. The point is that no row leaks into another and that the
    axis handling is orientation-independent.
    """
    rows = torch.tensor(np.random.default_rng(11).standard_normal((3, 2000)))

    batched = fast_resample_poly_torch(rows, up, down)
    for row_idx in range(rows.shape[0]):
        one_dimensional = fast_resample_poly_torch(rows[row_idx], up, down)
        torch.testing.assert_close(batched[row_idx], one_dimensional, rtol=0.0, atol=1e-13)

    along_axis_0 = fast_resample_poly_torch(rows.transpose(0, 1), up, down, axis=0)
    torch.testing.assert_close(along_axis_0.transpose(0, 1), batched, rtol=0.0, atol=1e-13)


@pytest.mark.unit
@pytest.mark.parametrize("up, down", [(3, 2), (2, 3), (5, 3)])
def test_torch_mixed_rate_gradients_match_the_fft_route(up, down):
    """The mixed path is on the training path, so gradcheck it and pin it to the FFT route.

    `gradcheck` catches a broken graph (an in-place scatter or an integer-indexed write
    would show up as a wrong or absent Jacobian); the comparison against the FFT route
    catches a graph that is intact but wired to the wrong taps.
    """
    signal = torch.tensor(np.random.default_rng(12).standard_normal((1, 30)), requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t: fast_resample_poly_torch(t, up, down), (signal,), eps=1e-6, atol=1e-8)

    divisor = math.gcd(up, down)
    weights = torch.tensor(np.random.default_rng(13).standard_normal(
        math.ceil(30 * (up // divisor) / (down // divisor))))

    def _grad(function):
        probe = signal.detach().clone().requires_grad_(True)
        return torch.autograd.grad((function(probe) * weights).sum(), probe)[0]

    polyphase = _grad(lambda t: fast_resample_poly_torch(t, up, down))
    reference = _grad(lambda t: _fft_route(t, up, down))
    assert torch.isfinite(polyphase).all()
    torch.testing.assert_close(polyphase, reference, rtol=0.0, atol=1e-14)


@pytest.mark.unit
def test_torch_mixed_rate_falls_back_to_the_fft_route_when_oversized(monkeypatch):
    """The FFT route stays reachable for intermediates the polyphase form should not build.

    Forced here by shrinking the threshold rather than by allocating a gigabyte, and
    pinned to produce the same signal either way -- the fallback is a memory decision, not
    a numerical one.
    """
    signal = torch.tensor(np.random.default_rng(14).standard_normal((1, 3000)))
    expected = fast_resample_poly_torch(signal, 3, 2)

    def _forbidden(*args, **kwargs):
        raise AssertionError("polyphase path taken despite the size guard")

    monkeypatch.setattr(torch_utils, "MIXED_POLYPHASE_MAX_ELEMENTS", 1)
    monkeypatch.setattr(torch_utils, "_polyphase_mixed", _forbidden)
    fallback = fast_resample_poly_torch(signal, 3, 2)

    assert fallback.shape == expected.shape
    _assert_close_relative(fallback.numpy(), expected.numpy(), 1e-13)


@pytest.mark.unit
@pytest.mark.parametrize("up, down", [(37, 1), (1, 37), (3, 2)])
def test_torch_resample_batches_rows_independently(up, down):
    """Batched rows must equal the same rows resampled one at a time."""
    rows = torch.tensor(np.random.default_rng(2).standard_normal((4, 3000)))

    batched = fast_resample_poly_torch(rows, up, down)
    for row_idx in range(rows.shape[0]):
        single = fast_resample_poly_torch(rows[row_idx:row_idx + 1], up, down)
        torch.testing.assert_close(batched[row_idx:row_idx + 1], single, rtol=0.0, atol=1e-13)


@pytest.mark.unit
@pytest.mark.parametrize("up, down", [(37, 1), (1, 37), (3, 2), (2, 3), (147, 160)])
def test_torch_resample_is_differentiable(up, down):
    """The resampler sits inside the backprop path, so gradients must flow.

    It is linear in its input, so the gradient of a plain sum is the column sum of the
    polyphase operator -- checked here against a finite difference on one input sample
    rather than re-deriving it.
    """
    signal = torch.tensor(np.random.default_rng(3).standard_normal(600), requires_grad=True)
    gradient = torch.autograd.grad(fast_resample_poly_torch(signal, up, down).sum(), signal)[0]
    assert torch.isfinite(gradient).all()

    step = 1e-6
    probe = signal.detach().clone()
    probe[300] += step
    moved = float(fast_resample_poly_torch(probe, up, down).sum())
    base = float(fast_resample_poly_torch(signal.detach(), up, down).sum())
    assert (moved - base) / step == pytest.approx(float(gradient[300]), rel=1e-5, abs=1e-8)


@pytest.mark.unit
def test_torch_resample_honours_axis_and_is_identity_for_equal_rates():
    rows = torch.tensor(np.random.default_rng(4).standard_normal((512, 3)))

    along_axis_0 = fast_resample_poly_torch(rows, 1, 4, axis=0)
    along_last = fast_resample_poly_torch(rows.transpose(0, 1), 1, 4, axis=-1)
    torch.testing.assert_close(along_axis_0, along_last.transpose(0, 1))

    assert fast_resample_poly_torch(rows, 3, 3) is rows


@pytest.mark.unit
def test_torch_resample_half_length_factor_changes_filter_length():
    """A shorter filter is a real accuracy knob, not a no-op."""
    signal = torch.tensor(np.random.default_rng(5).standard_normal(2000)).unsqueeze(0)

    default = fast_resample_poly_torch(signal, 1, 8,
                                       half_length_factor=DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR)
    shorter = fast_resample_poly_torch(signal, 1, 8, half_length_factor=3)

    assert default.shape == shorter.shape
    assert float(torch.max(torch.abs(default - shorter))) > 1e-6


@pytest.mark.unit
def test_torch_reduced_phase_exp_matches_an_exact_fraction_reference():
    """The modulation phase must be reduced mod one turn *before* the exponential.

    This is the whole of P5's accuracy story and it is not something the pipeline tests
    can see: `torch.exp` on the raw `2*pi*fc*n/fs` argument -- which reaches 2.3465e5 rad
    at the top band, n = T -- carries ~5.2e-11 rad of phase error, which sits ~2700x under
    the analysis parity bar and would ship silently.

    The reference is an exact rational reduction in Python integers, taken from an
    *unlimited* `Fraction(cf) / Fraction(fs)`. That is the whole point of the test and it
    must stay that way: build the reference from `_modulation_phase` and any error in the
    library's own ratio cancels on both sides, which is exactly how a live 3.4373e-12
    defect survived a green suite (see `_modulation_phase`).

    **`1000.0000000000001` is not a decorative extra centre frequency.** It is the
    analyzer's own base-frequency grid value, one ULP above 1000, and it is the one input
    on which `limit_denominator(10**12)` of the ratio collapses -- to exactly `1/24` at
    `fs = 24000`, discarding 1.137e-16 relative. The three centre frequencies this test
    used to carry could not trigger a collapse at all (`900/24000 = 3/80` is exactly
    representable), so it asserted its 2e-16 bar against a defect it could not reach. With
    the phase merely capped, this case measures ~3.5e-12 here.

    The negative range matters separately: `torch.remainder` follows the divisor's sign
    and `torch.div(..., "floor")` has to split negative indices consistently with it, and
    the synthesis tap index runs down to `-hf*up`.

    The bar is 2e-15, not the old 2e-16, because the two sides are now genuinely
    independent computations sitting on the float64 floor rather than the same rounding
    twice: `cos`/`sin` of an argument of magnitude 2*pi carries ~1e-15 on either side.
    Measured against a 50-digit mpmath reference, this implementation is 9.75e-16 at the
    collapsing centre frequency and 8.4e-16 / 9.1e-16 at the other two -- the reference
    below is itself 9.1e-16 from the same gold value. On every centre frequency whose cap
    is tight the library and the reference agree bit for bit.
    """
    sampling_frequency = 24000.0
    for center_frequency in (42.330280964441435, 900.0, 1000.0000000000001,
                             7468.976995250865):
        exact_ratio = Fraction(center_frequency) / Fraction(sampling_frequency)
        phase = torch_utils._modulation_phase(center_frequency, sampling_frequency)
        assert phase == exact_ratio, "the phase must not be approximated at all"
        for sign, start, step, count in ((-1, 0, 977, 123), (1, -4090, 1, 4091)):
            produced = torch_utils._reduced_phase_exp(
                phase, sign, start, step, count, torch.device("cpu"))
            angles = [sign * 2.0 * math.pi * float((exact_ratio * n) % 1)
                      for n in range(start, start + step * count, step)]
            expected = torch.tensor([complex(math.cos(a), math.sin(a)) for a in angles],
                                    dtype=torch.complex128)
            assert float((produced - expected).abs().max()) <= 2e-15


@pytest.mark.unit
@pytest.mark.parametrize("down, center_frequency",
                         [(14, 7000.0), (17, 5500.0), (409, 21.0),
                          (90, 1000.0000000000001)])
@pytest.mark.parametrize("rows", [1, 3])
def test_torch_modulated_decimate_matches_the_modulate_then_resample_route(
        down, center_frequency, rows):
    """`fast_demodulate_decimate_torch` is exactly `resample(x * exp(-2j*pi*fc*n/fs), 1, D)`.

    The fold rides the modulation on the 21 polyphase taps and leaves an `out_len`-long
    residual, so this is a reassociation, not an identity -- do not reach for
    `torch.equal`. Worst measured over the real 32-band filterbank at all four batch
    geometries the pipeline drives, against an independent exact reference: 1.219e-15 of
    peak.

    Three errors this catches that nothing else does: a sign slip on the tap phase (O(1)),
    a dropped `hf` offset in the residual, which is a per-band *constant rotation* with no
    shape change and no effect on any pipeline test's bar, and -- because the reference
    modulation is built independently by `_exact_modulation` rather than out of
    `_modulation_phase` -- any approximation of the ratio itself. The last case is the
    analyzer's real base-frequency band at 16 kHz (`cf` one ULP above 1000, `D = 90`),
    which measured 3.4373e-12 of peak while the ratio was being capped with
    `limit_denominator`.
    """
    sampling_frequency = 24000.0
    generator = np.random.default_rng(21)
    signal = torch.tensor(generator.standard_normal((rows, 6000))
                          + 1j * generator.standard_normal((rows, 6000)))

    modulation = _exact_modulation(center_frequency, sampling_frequency, signal.shape[-1])
    reference = fast_resample_poly_torch(signal * modulation, 1, down)
    folded = torch_utils.fast_demodulate_decimate_torch(
        signal, down, center_frequency, sampling_frequency)

    assert folded.shape == reference.shape
    assert float((folded - reference).abs().max()) <= 5e-15 * float(reference.abs().max())


@pytest.mark.unit
@pytest.mark.parametrize("dtype, tolerance",
                         [(torch.complex128, 5e-15), (torch.complex64, 1e-6),
                          (torch.float64, 5e-15), (torch.float32, 1e-6)])
def test_torch_modulated_decimate_runs_at_the_signals_own_precision(dtype, tolerance):
    """The folded path is not complex128-only, and it does not silently widen.

    The kernel and the residual are built by `torch.polar` on float64 -- the reduction has
    to run in double whatever the signal is -- and an earlier revision therefore produced a
    complex128 kernel unconditionally while sizing the accumulator off the signal. That
    made every non-complex128 entry raise rather than compute: complex64 gave "Expected out
    tensor to have dtype c10::complex<double>", real float64 gave "result type
    ComplexDouble can't be cast to Double", and both gradient routes gave "expected m1 and
    m2 to have the same dtype". The filterbank never reaches it -- the analyzer promotes to
    complex128 -- so no pipeline test could have caught this; the entry point's stated
    equivalence is unconditional, which is why it is a defect rather than a limitation.

    A real input widens to the matching complex dtype because the result is complex by
    construction; a complex64 input stays complex64 rather than being promoted by a
    complex128 modulation.
    """
    sampling_frequency, center_frequency, down = 24000.0, 1000.0000000000001, 90
    generator = np.random.default_rng(25)
    base = (generator.standard_normal((2, 3000))
            + 1j * generator.standard_normal((2, 3000)))
    signal = (torch.tensor(base) if dtype.is_complex
              else torch.tensor(base.real)).to(dtype)
    expected_dtype = torch_utils._complex_dtype_for(signal.real.dtype)

    modulation = _exact_modulation(center_frequency, sampling_frequency, signal.shape[-1])
    reference = fast_resample_poly_torch(
        signal.to(expected_dtype) * modulation.to(expected_dtype), 1, down)
    produced = torch_utils.fast_demodulate_decimate_torch(
        signal, down, center_frequency, sampling_frequency)

    assert produced.dtype == expected_dtype
    assert float((produced - reference).abs().max()) <= tolerance * float(
        reference.abs().max())

    probe = signal.to(expected_dtype).clone().requires_grad_(True)
    grad_output = torch_utils.fast_demodulate_decimate_torch(
        probe, down, center_frequency, sampling_frequency)
    (grad_output.real.sum() + grad_output.imag.sum()).backward()
    assert probe.grad.dtype == expected_dtype and torch.isfinite(probe.grad).all()


@pytest.mark.unit
@pytest.mark.parametrize("down", [14, 17, 409])
@pytest.mark.parametrize("rows", [1, 3])
def test_torch_modulated_decimate_gradients_match_the_modulate_then_resample_route(down, rows):
    """The fold is on the training path, so its gradient must match the route it replaces.

    `_polyphase_decimate` diverts `requires_grad` input to `_polyphase_decimate_padded`,
    and the decomposition really runs that branch with grad live, so the fold had to land
    in *both* bodies. Routing grad back to the pre-fold modulation matrix instead was
    rejected: it would have given the training and inference paths different numerics,
    which is the same latent defect class as the `hf = 0` grad/no-grad disagreement
    `_polyphase_decimate` already documents.

    Comparing `autograd.grad` against the modulate-then-resample route is a stronger
    statement than `gradcheck`: it pins the fold to the route it replaces, in the
    convention torch actually uses for complex autograd. Worst measured 1.367e-15.
    """
    sampling_frequency, center_frequency, in_len = 24000.0, 900.0, 4000
    generator = np.random.default_rng(22)
    base = torch.tensor(generator.standard_normal((rows, in_len))
                        + 1j * generator.standard_normal((rows, in_len)))
    out_len = math.ceil(in_len / down)
    real_weights = torch.tensor(generator.standard_normal((rows, out_len)))
    imag_weights = torch.tensor(generator.standard_normal((rows, out_len)))
    modulation = torch_utils._reduced_phase_exp(
        torch_utils._modulation_phase(center_frequency, sampling_frequency),
        -1, 0, 1, in_len, base.device)

    def _grad(function):
        probe = base.clone().requires_grad_(True)
        output = function(probe)
        loss = (output.real * real_weights).sum() + (output.imag * imag_weights).sum()
        return torch.autograd.grad(loss, probe)[0]

    folded = _grad(lambda t: torch_utils.fast_demodulate_decimate_torch(
        t, down, center_frequency, sampling_frequency))
    reference = _grad(lambda t: fast_resample_poly_torch(t * modulation, 1, down))

    assert torch.isfinite(folded).all()
    assert float((folded - reference).abs().max()) <= 1e-13 * float(reference.abs().max())


@pytest.mark.unit
@pytest.mark.parametrize("down", [14, 17, 409])
def test_torch_modulated_decimate_agrees_across_the_grad_and_no_grad_routes(down):
    """The padded (grad) and padless (no-grad) folded bodies must produce the same values.

    This is the assertion that fails if only one of the two bodies gets the fold, and it
    is the shape of check that would have caught the `hf = 0` defect. Worst measured
    8.14e-17 of peak over the real 32 bands, 3.73e-16 here -- the same ~1 ULP
    reassociation the unmodulated pair already carries.
    """
    sampling_frequency, center_frequency, in_len = 24000.0, 5500.0, 4000
    generator = np.random.default_rng(23)
    base = torch.tensor(generator.standard_normal((2, in_len))
                        + 1j * generator.standard_normal((2, in_len)))

    with torch.no_grad():
        padless = torch_utils.fast_demodulate_decimate_torch(
            base, down, center_frequency, sampling_frequency)
    padded = torch_utils.fast_demodulate_decimate_torch(
        base.clone().requires_grad_(True), down, center_frequency, sampling_frequency)

    assert padded.grad_fn is not None
    assert float((padded.detach() - padless).abs().max()) <= 1e-14 * float(padless.abs().max())


@pytest.mark.unit
def test_torch_modulated_polyphase_kernel_is_cached_per_band():
    """The modulated kernel is keyed per band; the unmodulated one stays keyed per rate.

    A cache that ignored the centre frequency, or that fed the band's modulation back into
    `_get_polyphase_kernel`, would serve one band's taps to another with no shape error.
    The rate cache must also survive untouched -- the fullband 3/2 and 2/3 resamples and
    every real caller still need it, and evicting them for 26..42 complex kernels per
    filterbank configuration is why this is a separate cache rather than a widened one.
    """
    sampling_frequency = 24000.0
    device = torch.device("cpu")
    half_length_factor = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
    low = torch_utils._modulation_phase(900.0, sampling_frequency)
    high = torch_utils._modulation_phase(7000.0, sampling_frequency)

    first = torch_utils._get_modulated_polyphase_kernel(
        1, 17, low, torch.float64, device, half_length_factor)
    assert torch_utils._get_modulated_polyphase_kernel(
        1, 17, low, torch.float64, device, half_length_factor) is first
    assert not torch.equal(first, torch_utils._get_modulated_polyphase_kernel(
        1, 17, high, torch.float64, device, half_length_factor))
    assert not torch.equal(first, torch_utils._get_modulated_polyphase_kernel(
        1, 17, low, torch.float64, device, 3))

    plain = torch_utils._get_polyphase_kernel(1, 17, torch.float64, device, half_length_factor)
    assert first.is_complex() and not plain.is_complex()
    assert first.shape == plain.shape

    # The two sides share this cache and must not share entries: they are different
    # filters (`(1, D)` is designed at cutoff 1/D, `(U, 1)` at 1/U and scaled by U), they
    # carry different tap phases (`conj(m[k])` against `m[k - hf*U]`), and only the
    # decimation one takes the lane-reversing `flip(1)`.
    interpolation = torch_utils._get_modulated_polyphase_kernel(
        17, 1, low, torch.float64, device, half_length_factor)
    assert interpolation.shape == first.shape
    assert not torch.equal(interpolation, first)


@pytest.mark.unit
def test_torch_modulated_decimate_is_a_plain_modulation_at_down_one():
    """Contract guard: `down == 1` has to reproduce the full-rate modulation multiply.

    `fast_resample_poly_torch` returns its input unchanged at `up == down`, and the
    modulation used to be a separate pass, so the folded entry point owes the same answer.
    `decimations` is clamped at 1 and its measured minimum is 13..15 across every
    supported input rate, so this branch is unreachable from the filterbank -- it is here
    so the docstring's stated equivalence holds for every `down`, not because it is hot.
    """
    sampling_frequency, center_frequency = 24000.0, 900.0
    generator = np.random.default_rng(24)
    signal = torch.tensor(generator.standard_normal((2, 500))
                          + 1j * generator.standard_normal((2, 500)))
    modulation = torch_utils._reduced_phase_exp(
        torch_utils._modulation_phase(center_frequency, sampling_frequency),
        -1, 0, 1, signal.shape[-1], signal.device)

    folded = torch_utils.fast_demodulate_decimate_torch(
        signal, 1, center_frequency, sampling_frequency)
    torch.testing.assert_close(folded, signal * modulation, rtol=0.0, atol=0.0)


@pytest.mark.unit
@pytest.mark.parametrize("up, center_frequency",
                         [(14, 7000.0), (17, 5500.0), (409, 21.0),
                          (90, 1000.0000000000001)])
@pytest.mark.parametrize("rows", [1, 3])
def test_torch_modulated_interpolate_matches_the_resample_then_modulate_route(
        up, center_frequency, rows):
    """`fast_interpolate_modulate_torch` is `resample(x, U, 1) * exp(+2j*pi*fc*n/fs)`.

    The synthesis mirror of the decimation check, and it is not a re-parametrisation of
    it: the two halves differ in sign convention and in index origin. The taps here carry
    `m[k - hf*U]` *unconjugated* against the analysis side's `conj(m[k])`, and the
    surviving vector moves from the decimated *output* to the decimated *input*. Three
    ways to get it silently wrong, none of which changes any shape:

    * the `-hf*U` offset dropped -- a per-band constant rotation, invisible to every
      pipeline bar (the synthesis parity test's correlation deficit is second order in it,
      and the temporal-alignment test only sees index shifts);
    * a `flip(1)` added to the kernel -- `_get_polyphase_kernel` applies that only when
      `up == 1`, and on the interpolation side it permutes the polyphase lanes;
    * the pre-multiply indexed by the output index rather than the input index.

    Worst measured over the real 32-band filterbank against an independent exact
    reference: 2.224e-15 of peak. The last case is the analyzer's base-frequency band,
    which is the one whose exact `fc/fs` a `limit_denominator` cap collapses.
    """
    sampling_frequency = 24000.0
    in_len = 300
    generator = np.random.default_rng(26)
    signal = torch.tensor(generator.standard_normal((rows, in_len))
                          + 1j * generator.standard_normal((rows, in_len)))

    modulation = _exact_modulation(center_frequency, sampling_frequency, in_len * up,
                                   sign=1)
    reference = fast_resample_poly_torch(signal, up, 1) * modulation
    folded = torch_utils.fast_interpolate_modulate_torch(
        signal, up, center_frequency, sampling_frequency)

    assert folded.shape == reference.shape
    assert float((folded - reference).abs().max()) <= 5e-15 * float(reference.abs().max())


@pytest.mark.unit
@pytest.mark.parametrize("dtype, tolerance",
                         [(torch.complex128, 5e-15), (torch.complex64, 1e-6),
                          (torch.float64, 5e-15), (torch.float32, 1e-6)])
def test_torch_modulated_interpolate_runs_at_the_signals_own_precision(dtype, tolerance):
    """The folded interpolation is not complex128-only, and it does not silently widen.

    Same contract and same failure mode as the decimation sibling: the phase is reduced in
    float64 whatever the signal is, so the kernel and the pre-multiply have to be rounded
    into the signal's own complex dtype or `matmul` raises "expected m1 and m2 to have the
    same dtype". Unreachable from the filterbank -- the subbands are complex128 -- but the
    entry point's stated equivalence is unconditional.
    """
    sampling_frequency, center_frequency, up = 24000.0, 1000.0000000000001, 90
    generator = np.random.default_rng(27)
    base = (generator.standard_normal((2, 200))
            + 1j * generator.standard_normal((2, 200)))
    signal = (torch.tensor(base) if dtype.is_complex
              else torch.tensor(base.real)).to(dtype)
    expected_dtype = torch_utils._complex_dtype_for(signal.real.dtype)

    modulation = _exact_modulation(center_frequency, sampling_frequency,
                                   signal.shape[-1] * up, sign=1)
    reference = (fast_resample_poly_torch(signal.to(expected_dtype), up, 1)
                 * modulation.to(expected_dtype))
    produced = torch_utils.fast_interpolate_modulate_torch(
        signal, up, center_frequency, sampling_frequency)

    assert produced.dtype == expected_dtype
    assert float((produced - reference).abs().max()) <= tolerance * float(
        reference.abs().max())

    probe = signal.to(expected_dtype).clone().requires_grad_(True)
    grad_output = torch_utils.fast_interpolate_modulate_torch(
        probe, up, center_frequency, sampling_frequency)
    (grad_output.real.sum() + grad_output.imag.sum()).backward()
    assert probe.grad.dtype == expected_dtype and torch.isfinite(probe.grad).all()


@pytest.mark.unit
@pytest.mark.parametrize("up", [14, 17, 409])
@pytest.mark.parametrize("rows", [1, 3])
def test_torch_modulated_interpolate_gradients_match_the_resample_then_modulate_route(
        up, rows):
    """The synthesis fold is on the training path too, and it has only one body.

    `_polyphase_interpolate` is the one polyphase routine with no
    `torch.is_grad_enabled()` branch -- `F.pad`, `unfold`, complex `matmul` and the
    pre-multiply are all autograd-safe -- so unlike the decimation half this fold cannot
    produce a grad/no-grad numerical split, and no `autograd.Function` is needed. What
    still has to be pinned is that the gradient matches the route the fold replaces, in
    the convention torch uses for complex autograd. Measured 1.207e-15 of peak w.r.t. all
    32 subband tensors through a whole `run_auditory_synthesis_filterbank_torch`.
    """
    sampling_frequency, center_frequency, in_len = 24000.0, 900.0, 200
    generator = np.random.default_rng(28)
    base = torch.tensor(generator.standard_normal((rows, in_len))
                        + 1j * generator.standard_normal((rows, in_len)))
    out_len = in_len * up
    real_weights = torch.tensor(generator.standard_normal((rows, out_len)))
    imag_weights = torch.tensor(generator.standard_normal((rows, out_len)))
    modulation = torch_utils._reduced_phase_exp(
        torch_utils._modulation_phase(center_frequency, sampling_frequency),
        1, 0, 1, out_len, base.device)

    def _grad(function):
        probe = base.clone().requires_grad_(True)
        output = function(probe)
        loss = (output.real * real_weights).sum() + (output.imag * imag_weights).sum()
        return torch.autograd.grad(loss, probe)[0]

    folded = _grad(lambda t: torch_utils.fast_interpolate_modulate_torch(
        t, up, center_frequency, sampling_frequency))
    reference = _grad(lambda t: fast_resample_poly_torch(t, up, 1) * modulation)

    assert torch.isfinite(folded).all()
    assert float((folded - reference).abs().max()) <= 1e-13 * float(reference.abs().max())


@pytest.mark.unit
def test_torch_modulated_interpolate_is_a_plain_modulation_at_up_one():
    """Contract guard: `up == 1` has to reproduce the full-rate modulation multiply.

    The mirror of the `down == 1` guard, unreachable from the filterbank for the same
    reason, and here so the docstring's stated equivalence holds for every `up`.
    """
    sampling_frequency, center_frequency = 24000.0, 900.0
    generator = np.random.default_rng(29)
    signal = torch.tensor(generator.standard_normal((2, 500))
                          + 1j * generator.standard_normal((2, 500)))
    modulation = torch_utils._reduced_phase_exp(
        torch_utils._modulation_phase(center_frequency, sampling_frequency),
        1, 0, 1, signal.shape[-1], signal.device)

    folded = torch_utils.fast_interpolate_modulate_torch(
        signal, 1, center_frequency, sampling_frequency)
    torch.testing.assert_close(folded, signal * modulation, rtol=0.0, atol=0.0)
