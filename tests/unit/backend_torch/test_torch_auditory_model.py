import math

import numpy as np
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


# Cross-implementation agreement tolerance for the adaptation recurrence.
#
# The Numba kernel is a hand transcription of the torch loop, and on the reference
# platform (Windows, CPython 3.10, torch 2.12.1+cpu, numba 0.65.1) the two are exactly
# bit-identical -- `torch.equal` holds at every shape measured. That is *not* portable,
# and asserting it as though it were is what originally broke CI: on CPython 3.14 the
# two diverged by 1.8e-14 absolute. Cross-implementation bit-equality depends on
# whether the toolchain contracts `a*b + c` into an FMA, which differs by LLVM and torch
# build. (Locally the kernel compiles to separate `vmulsd`/`vaddsd`, no FMA.)
#
# The tolerance is set from measurement, not taste. Perturbations of this recurrence
# fall into two cleanly separated bands:
#
#   roundoff        FMA contraction ~2.6e-16 rel, algebraically-equal EMA
#                   reassociation 1.5e-14 rel
#   real bugs       using the updated state in the running product 4.4e+00 rel,
#                   resetting the running product per stage 1.0e+00 rel
#
# Fourteen orders apart, so 1e-12 catches every transcription error while tolerating
# every toolchain difference. Tightening this to exact equality does not buy accuracy,
# it just re-breaks on the next compiler.
_ADAPTATION_RTOL = 1e-12
_ADAPTATION_ATOL = 1e-12

# The public entry point needs a looser bound than the raw kernel, and not because it
# is any less exact -- it is the same roundoff, run through an amplifier.
# ``simulate_auditory_nerve_adaptation`` finishes with
#
#     100/(1 - final_thresh) * (adapted - final_thresh),  final_thresh = 0.6978305849
#
# i.e. a 330.94x scale-up around a subtraction. Outputs span four orders (median 161,
# minimum 0.024), and for the ~0.3% of samples that land near ``final_thresh`` the
# cancellation shrinks the denominator until relative error is meaningless: CI measured
# 7.9e-11 relative at a point whose output value is 0.036, from an absolute error of
# 8.7e-15 in ``adapted`` -- the same 1-ULP roundoff the kernel test tolerates at 1e-12.
#
# So this bound is set on absolute error at the output scale: roundoff in ``adapted``
# (~1e-14) times 330.94 is ~3e-12, and 1e-9 leaves ~300x margin. A real transcription
# error is O(1) in ``adapted``, hence ~1e2 here -- caught by eleven orders. This is the
# same cancellation the README already documents for the `artifacts` component.
_ADAPTED_OUTPUT_RTOL = 1e-9
_ADAPTED_OUTPUT_ATOL = 1e-9


@pytest.mark.skipif(not auditory_model._HAS_NUMBA, reason="numba is an optional extra")
def test_numba_adaptation_kernel_matches_the_torch_loop():
    """Pin the Numba kernel to ``_adaptation_loop_forward``.

    Nothing in the type system keeps a hand transcription in lockstep with the loop it
    transcribes, so this is what catches an index, stage or ordering slip -- each of
    which lands at O(1) relative error through the feedback cascade, not near this
    tolerance. See the note above for why it is a tolerance rather than bit equality.
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

        torch.testing.assert_close(
            actual, expected, rtol=_ADAPTATION_RTOL, atol=_ADAPTATION_ATOL,
            msg=lambda built: f"numba/torch adaptation drift at ({rows}, {steps}):\n{built}"
        )


@pytest.mark.skipif(not auditory_model._HAS_NUMBA, reason="numba is an optional extra")
def test_adaptation_dispatch_agrees_across_fast_and_fallback_paths(monkeypatch):
    """The public entry point must not depend on whether the Numba path is taken.

    Bounded at the output scale rather than relatively -- see the note above for why
    the affine tail makes relative error unusable near ``final_thresh``.
    """
    subbands = torch.rand(9, 3000, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 1e-3

    fast = simulate_auditory_nerve_adaptation(subbands, 24000.0)
    monkeypatch.setattr(auditory_model, "_HAS_NUMBA", False)
    fallback = simulate_auditory_nerve_adaptation(subbands, 24000.0)

    torch.testing.assert_close(fast, fallback, rtol=_ADAPTED_OUTPUT_RTOL, atol=_ADAPTED_OUTPUT_ATOL)


# -----------------------------------------------------------------------------
# The fused haircell+adaptation fast path
# -----------------------------------------------------------------------------
# `_numba_fused_haircell_adaptation` runs relu -> haircell -> clamp -> adaptation ->
# affine in one row-major pass. Everything from the clamp onwards is the same
# transcription `_numba_adaptation_loop` already carries, so it is pinned tightly
# below. The haircell is *not* the same evaluation: the torch function convolves the
# impulse response truncated at 10 ms by FFT, the kernel runs the one-pole recurrence
# that response comes from. They are the same filter to `g**(0.01*fs) == exp(-20*pi)
# == 5.2e-28` relative -- fs-independent, twelve orders below float64 eps -- so what
# separates them is roundoff, and the recurrence is the more accurate of the two
# (pinned by `test_fused_haircell_beats_the_fft_path_against_a_106_bit_oracle`).
#
# The bound is relative to the output's own peak rather than absolute, because these
# tests run at several signal scales. Measured worst on the MATLAB reference clip's
# real subbands: 6.1e-13 of peak; on the random cases below, ~1e-15. 1e-9 leaves
# ~1600x margin on the worst of those, and a real transcription slip lands at O(1)
# relative -- nine orders above it.
_FUSED_PEAK_RELATIVE_TOL = 1e-9

# Tolerance for the *transcription* comparison, where the two sides run the same
# haircell recurrence and only the arithmetic below it is under test. On this platform
# they are exactly equal, and an earlier draft of that test asserted `torch.equal`.
# That assertion is not portable and must not come back: the recurrence's inner line is
# `s = g*s + w*x`, precisely the `a*b + c` shape a toolchain may contract into a single
# FMA, and whether it does differs by LLVM and torch build. ARCHIVE.md records exactly
# this failure mode breaking CI on CPython 3.14 for the adaptation kernel next door.
#
# Sized by emulating the contracted form: replacing that line with a correctly-rounded
# (double-double) FMA in the reference path moves the output 3.3e-16 / 2.9e-18 /
# 4.6e-17 of peak on the three parametrizations below, and breaks bit-equality on all
# three. 1e-12 clears the worst of those by ~3000x while a real index, stage or
# ordering slip lands at O(1) relative -- twelve orders above it. Same value, and the
# same reasoning, as `_ADAPTATION_RTOL` above.
_FUSED_TRANSCRIPTION_PEAK_TOL = 1e-12


def _reference_haircell_iir(x: torch.Tensor, fs: float) -> torch.Tensor:
    """`y[n] = g*y[n-1] + (1-g)*max(x[n], 0)`, the filter both paths implement."""
    gain = math.exp(-math.pi * 2000.0 / fs)
    weight = 1.0 - gain
    rectified = torch.clamp(x, min=0.0).numpy()
    out = np.empty_like(rectified)
    state = np.zeros(rectified.shape[:-1])
    for n in range(rectified.shape[-1]):
        state = gain * state + weight * rectified[..., n]
        out[..., n] = state
    return torch.from_numpy(out)


@pytest.mark.skipif(not auditory_model._HAS_NUMBA, reason="numba is an optional extra")
@pytest.mark.parametrize("rows, steps, scale", [(9, 3000, 1e-3), (6, 4000, 1.0), (27, 20000, 1e-1)])
def test_fused_kernel_transcribes_the_torch_stages_it_replaces(rows, steps, scale):
    """Everything except the haircell evaluation must be the torch path's arithmetic.

    Feed the torch adaptation the recurrence's output and the fused kernel the raw
    subbands: the clamp, the five-stage cascade and the final affine are then the
    only shared work. This is what catches an index, stage or ordering slip in the
    transcription -- each lands at O(1) relative error, not near this tolerance.

    A *bound* and not `torch.equal`, even though the two sides are exactly equal here.
    The comparison crosses a numpy-vs-numba boundary at the haircell line `s = g*s + w*x`,
    which is the `a*b + c` shape a compiler may contract into one FMA; emulating that
    contraction moves the output ~3e-16 of peak and breaks equality on all three
    parametrizations, and ARCHIVE.md records the same fragility breaking CI on CPython
    3.14. See `_FUSED_TRANSCRIPTION_PEAK_TOL` for how the number was chosen. Bit-equality
    on this platform is still worth knowing about, so it is *reported* in the failure
    message rather than asserted.
    """
    fs = 24000.0
    subbands = (torch.rand(rows, steps, dtype=torch.float64,
                           generator=torch.Generator().manual_seed(3)) - 0.4) * scale

    expected = simulate_auditory_nerve_adaptation(_reference_haircell_iir(subbands, fs), fs)
    actual = auditory_model._fused_haircell_adaptation(subbands, fs)

    peak = float(expected.abs().max())
    deviation = float((actual - expected).abs().max())
    assert deviation <= _FUSED_TRANSCRIPTION_PEAK_TOL * peak, (
        f"transcription drift {deviation:.3e} at ({rows}, {steps}), peak {peak:.3e}, "
        f"{deviation / peak:.3e} of peak (bit-equal on the reference platform)"
    )


@pytest.mark.skipif(not auditory_model._HAS_NUMBA, reason="numba is an optional extra")
@pytest.mark.parametrize("rows, steps, scale", [(9, 3000, 1e-3), (6, 4000, 1.0), (27, 20000, 1e-1)])
def test_fused_kernel_agrees_with_the_unfused_torch_path(rows, steps, scale):
    """The fast path must not change the answer beyond haircell roundoff."""
    fs = 24000.0
    subbands = (torch.rand(rows, steps, dtype=torch.float64,
                           generator=torch.Generator().manual_seed(3)) - 0.4) * scale

    unfused = simulate_auditory_nerve_adaptation(simulate_inner_haircell_transduction(subbands, fs), fs)
    fused = auditory_model._fused_haircell_adaptation(subbands, fs)

    peak = float(unfused.abs().max())
    deviation = float((fused - unfused).abs().max())
    assert deviation <= _FUSED_PEAK_RELATIVE_TOL * peak, (
        f"fused/unfused drift {deviation:.3e} at ({rows}, {steps}), peak {peak:.3e}"
    )


def test_fused_haircell_beats_the_fft_path_against_a_106_bit_oracle():
    """The recurrence is not an approximation of the FFT FIR -- it is the better one.

    Both evaluate the same one-pole filter, so "which is right" needs arithmetic
    wider than float64. `np.longdouble` is *not* that on Windows/MSVC (it is float64,
    `finfo(longdouble).eps == 2.22e-16`), and a longdouble "oracle" there silently
    measures nothing -- it would report the recurrence as exact by construction.
    This builds a genuine ~106-bit oracle with double-double (Dekker/Knuth) products
    and sums, which is platform-independent.

    Measured on the reference clip's real subbands, worst absolute error against that
    oracle: FFT-FIR 4.31e-16, recurrence 1.96e-16 (4.6e-16 vs 2.1e-16 of the signal's
    own peak). The assertion here is the weaker, robust form -- the recurrence is at
    least as close everywhere -- since the exact ratio is toolchain-dependent.
    """
    fs = 24000.0
    gain = math.exp(-math.pi * 2000.0 / fs)
    weight = 1.0 - gain
    assert gain + weight == 1.0, "1-g must be exact for the oracle's inputs to be exact"

    x = torch.clamp(torch.randn(6, 4000, dtype=torch.float64,
                                generator=torch.Generator().manual_seed(11)), min=0.0)
    rectified = x.numpy()

    # Double-double state (hi, lo) carrying ~106 bits through the recurrence.
    splitter = 134217729.0  # 2**27 + 1

    def two_prod(a, b):
        product = a * b
        a_c = splitter * a
        a_hi = a_c - (a_c - a)
        b_c = splitter * b
        b_hi = b_c - (b_c - b)
        a_lo, b_lo = a - a_hi, b - b_hi
        return product, ((a_hi * b_hi - product) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo

    hi = np.zeros(rectified.shape[0])
    lo = np.zeros(rectified.shape[0])
    oracle = np.empty_like(rectified)
    for n in range(rectified.shape[1]):
        scaled_hi, scaled_lo = two_prod(hi, gain)
        scaled_lo = scaled_lo + lo * gain
        carried = scaled_hi + scaled_lo
        scaled_hi, scaled_lo = carried, scaled_lo - (carried - scaled_hi)

        term_hi, term_lo = two_prod(weight, rectified[:, n])

        total = scaled_hi + term_hi
        bb = total - scaled_hi
        error = (scaled_hi - (total - bb)) + (term_hi - bb) + (scaled_lo + term_lo)
        hi = total + error
        lo = error - (hi - total)
        oracle[:, n] = hi

    fft_error = np.abs(simulate_inner_haircell_transduction(x, fs).numpy() - oracle)
    iir_error = np.abs(_reference_haircell_iir(x, fs).numpy() - oracle)

    assert iir_error.max() <= fft_error.max()
    assert (iir_error ** 2).mean() <= (fft_error ** 2).mean()


@pytest.mark.skipif(not auditory_model._HAS_NUMBA, reason="numba is an optional extra")
def test_fused_fast_path_predicate_rejects_what_the_kernel_cannot_serve():
    """The fused kernel is a fourth branch, not a replacement.

    This pins the *predicate*, on tensors handed to it directly. What the predicate
    rejects is not the same as what the pipeline never produces -- see
    `test_a_float32_caller_still_reaches_the_fused_kernel` for the float32 case, which
    is dead in practice because the analyzer promotes before this is ever consulted.
    """
    subbands = torch.rand(4, 500, dtype=torch.float64)

    assert auditory_model._can_fuse_haircell_adaptation(subbands)
    assert not auditory_model._can_fuse_haircell_adaptation(subbands.float())
    assert not auditory_model._can_fuse_haircell_adaptation(subbands.clone().requires_grad_(True))
    with torch.no_grad():
        assert auditory_model._can_fuse_haircell_adaptation(subbands.clone().requires_grad_(True))


@pytest.mark.skipif(not auditory_model._HAS_NUMBA, reason="numba is an optional extra")
def test_a_float32_caller_still_reaches_the_fused_kernel():
    """`float32` in does NOT mean the torch fallback -- the gammatone promotes first.

    `GammatoneAnalyzerTorch.process`/`process_real` cast their rows to
    `complex128`/`float64` unconditionally, so the subbands this predicate sees are
    float64 whatever dtype the caller passed. A float32 caller of
    `calculate_auditory_quality_features` therefore *does* get the fused kernel, and
    its answer moves by the haircell-roundoff amount like everyone else's -- measured
    2.4e-12 (2.3e-15 of peak) on the representation below, and <=4.4e-16 absolute on
    the four features themselves.

    Recorded as a test because the obvious reading of the predicate is the opposite,
    and an earlier draft of this module, of the kernel's docstring and of the README
    all claimed float32 callers were excluded. The fix is documentation: adding a
    float32 branch to the kernel would be a real behaviour change, and the promotion
    is deliberate (`backend_torch/predictor.py` notes the pipeline always promotes).
    """
    seen = []
    real_gate = auditory_model._can_fuse_haircell_adaptation

    def spy(subbands):
        verdict = real_gate(subbands)
        seen.append((subbands.dtype, subbands.device.type, verdict))
        return verdict

    signal_data = (torch.rand(2, 6000, dtype=torch.float64,
                              generator=torch.Generator().manual_seed(5)) - 0.5).float()
    auditory_model._can_fuse_haircell_adaptation = spy
    try:
        with torch.no_grad():
            out, _ = auditory_model.generate_auditory_internal_representation_torch(signal_data, 16000.0)
    finally:
        auditory_model._can_fuse_haircell_adaptation = real_gate

    assert signal_data.dtype == torch.float32
    assert seen, "the gate was never consulted"
    assert all(dtype == torch.float64 and device == "cpu" and verdict for dtype, device, verdict in seen)
    assert out.dtype == torch.float64


@pytest.mark.skipif(not auditory_model._HAS_NUMBA, reason="numba is an optional extra")
def test_internal_representation_agrees_across_fused_and_unfused_paths(monkeypatch):
    """The public entry point must not depend on whether the fused path is taken."""
    signal_data = torch.rand(2, 6000, dtype=torch.float64,
                             generator=torch.Generator().manual_seed(5)) - 0.5

    fused, fs_fused = auditory_model.generate_auditory_internal_representation_torch(signal_data, 16000.0)
    monkeypatch.setattr(auditory_model, "_HAS_NUMBA", False)
    unfused, fs_unfused = auditory_model.generate_auditory_internal_representation_torch(signal_data, 16000.0)

    assert fs_fused == fs_unfused
    peak = float(unfused.abs().max())
    deviation = float((fused - unfused).abs().max())
    assert deviation <= _FUSED_PEAK_RELATIVE_TOL * peak, (
        f"representation drift {deviation:.3e} against peak {peak:.3e}"
    )


def test_adaptation_gradient_path_is_unaffected_by_the_fast_forward():
    """The fast paths are forward-only; a grad-requiring input must still backprop."""
    subbands = (torch.rand(4, 400, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 1e-3)
    subbands.requires_grad_(True)

    simulate_auditory_nerve_adaptation(subbands, 24000.0).sum().backward()

    assert subbands.grad is not None
    assert torch.isfinite(subbands.grad).all()
    assert subbands.grad.abs().max() > 0
