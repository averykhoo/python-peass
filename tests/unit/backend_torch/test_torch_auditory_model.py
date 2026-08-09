import math

import pytest
import torch

from peass.backend_torch import auditory_model
from peass.backend_torch.auditory_model import simulate_auditory_nerve_adaptation
from peass.backend_torch.auditory_model import simulate_inner_haircell_transduction
from peass.backend_torch.utils import smoothmax


def test_smoothmax_parity():
    # Assert smoothmax identically matches max(x, threshold) for k=1000
    x = torch.linspace(-5.0, 5.0, 100)
    thresh = 0.5

    hard_max = torch.max(x, torch.tensor(thresh))
    soft_max = smoothmax(x, thresh, k=1000.0)

    torch.testing.assert_close(soft_max, hard_max, rtol=1e-3, atol=1e-3)


def test_auditory_nerve_compilation():
    subbands = torch.randn(4, 1000, dtype=torch.float64)  # 4 bands, 1000 samples
    transduced = simulate_inner_haircell_transduction(subbands, 16000.0)

    # Ensures the compiled loop runs without exceptions
    adapted = simulate_auditory_nerve_adaptation(transduced, 16000.0)

    assert adapted.shape == subbands.shape
    assert not torch.isnan(adapted).any()


def test_haircell_matches_reference_iir():
    """The FFT-based haircell must equal the causal one-pole IIR it approximates."""
    fs = 24000.0
    x = torch.relu(torch.randn(6, 4000, dtype=torch.float64))
    out = simulate_inner_haircell_transduction(x, fs)

    # Reference: direct causal recurrence y[n] = (1-g)*x[n] + g*y[n-1]
    gain = math.exp(-math.pi * 2000.0 / fs)
    ref = torch.empty_like(x)
    prev = torch.zeros(x.shape[0], dtype=torch.float64)
    for n in range(x.shape[1]):
        prev = (1.0 - gain) * x[:, n] + gain * prev
        ref[:, n] = prev

    # The 10 ms FIR truncation makes this near-exact (tail gain**240 ~ 0).
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-9)


def test_haircell_scales_to_long_many_band_batches():
    """Regression: the haircell must handle realistic-length, many-band batches.
    A (128, 96000) input would need ~23 GB via conv1d im2col; the FFT path keeps
    memory bounded. Runs fast because only the (cheap) haircell is exercised."""
    x = torch.randn(128, 96000, dtype=torch.float64)
    out = simulate_inner_haircell_transduction(x, 24000.0)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not auditory_model._HAS_NUMBA, reason="numba is an optional extra")
def test_numba_adaptation_kernel_is_bit_identical_to_scripted_loop():
    """The Numba kernel is a hand transcription of ``_adaptation_loop_forward``.

    Nothing in the type system keeps the two in lockstep, so this pins them: any
    reassociation in either -- an FMA contraction, a reordered EMA -- compounds
    through the feedback cascade and silently shifts every score. Bit equality, not
    a tolerance, is the assertion, because that is what was measured when the kernel
    landed and anything less would not be noticed until it was large.
    """
    fs = 24000.0
    abs_thresh, gains, thresholds = auditory_model._get_adaptation_constants(fs, "cpu", torch.float64)

    for rows, steps in ((7, 500), (27, 4000)):
        flat = torch.rand(rows, steps, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 1e-3

        expected = auditory_model._adaptation_loop_forward(flat, thresholds, gains, abs_thresh)
        frames = torch.clamp(flat, min=abs_thresh).t().contiguous()
        actual = torch.from_numpy(auditory_model._numba_adaptation_loop(
            frames.numpy(), thresholds.numpy(), gains.numpy()
        )).t().contiguous()

        assert torch.equal(actual, expected), (
            f"numba/torch adaptation drift at ({rows}, {steps}): "
            f"max|diff| = {(actual - expected).abs().max().item():.3e}"
        )


@pytest.mark.skipif(not auditory_model._HAS_NUMBA, reason="numba is an optional extra")
def test_adaptation_dispatch_agrees_across_fast_and_fallback_paths(monkeypatch):
    """The public entry point must not depend on whether the Numba path is taken."""
    subbands = torch.rand(9, 3000, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 1e-3

    fast = simulate_auditory_nerve_adaptation(subbands, 24000.0)
    monkeypatch.setattr(auditory_model, "_HAS_NUMBA", False)
    fallback = simulate_auditory_nerve_adaptation(subbands, 24000.0)

    assert torch.equal(fast, fallback)


def test_adaptation_gradient_path_is_unaffected_by_the_fast_forward():
    """The fast paths are forward-only; a grad-requiring input must still backprop."""
    subbands = (torch.rand(4, 400, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 1e-3)
    subbands.requires_grad_(True)

    simulate_auditory_nerve_adaptation(subbands, 24000.0).sum().backward()

    assert subbands.grad is not None
    assert torch.isfinite(subbands.grad).all()
    assert subbands.grad.abs().max() > 0
