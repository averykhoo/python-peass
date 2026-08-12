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
