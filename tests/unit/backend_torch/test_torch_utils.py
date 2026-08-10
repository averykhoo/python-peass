"""Tests for the PyTorch polyphase resampler.

`fast_resample_poly_torch` dispatches on the reduced rate: pure interpolation and pure
decimation take a polyphase GEMM path, and only genuinely mixed rates fall through to
the FFT linear convolution. The filterbank drives it almost entirely through the first
two -- 32 bands with 32 distinct decimation factors -- so the GEMM path is the one that
matters, and it is checked here against SciPy's `resample_poly`, which is the reference
the whole resampler is written to replicate.
"""
import math

import numpy as np
import pytest
import torch
from scipy.signal import resample_poly

from peass.backend_torch.utils import DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
from peass.backend_torch.utils import fast_resample_poly_torch

# (up, down): pure interpolation, pure decimation, and mixed rates (the FFT path).
# 368 and 1229 are real filterbank decimation factors for a 24 kHz-class analysis rate.
RATE_CASES = [(3, 2), (2, 3), (2, 1), (1, 2), (7, 1), (1, 3), (16, 1), (1, 58),
              (368, 1), (1, 368), (1, 1229)]


def _scipy_reference(x: np.ndarray, up: int, down: int) -> np.ndarray:
    return resample_poly(x, up, down, window=('kaiser', 5.0), padtype='constant')


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
@pytest.mark.parametrize("up, down", [(37, 1), (1, 37), (3, 2)])
def test_torch_resample_batches_rows_independently(up, down):
    """Batched rows must equal the same rows resampled one at a time."""
    rows = torch.tensor(np.random.default_rng(2).standard_normal((4, 3000)))

    batched = fast_resample_poly_torch(rows, up, down)
    for row_idx in range(rows.shape[0]):
        single = fast_resample_poly_torch(rows[row_idx:row_idx + 1], up, down)
        torch.testing.assert_close(batched[row_idx:row_idx + 1], single, rtol=0.0, atol=1e-13)


@pytest.mark.unit
@pytest.mark.parametrize("up, down", [(37, 1), (1, 37)])
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
