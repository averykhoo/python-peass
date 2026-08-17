"""
PEASS Test Suite - Differential Parity Tests (NumPy vs PyTorch)
File path: tests/regression/test_differential_numpy_vs_torch.py
"""

import math

import numpy as np
import pytest
import scipy.signal as signal
import soundfile as sf
import torch

import peass.backend_torch.decomposition as decomp_module
from peass import DecompositionConfiguration
from peass.backend_numpy.decomposition import decompose_distortion_components as decomp_np
from peass.backend_numpy.decomposition import extract_target_spatial_distortion_interference_artifacts as ext_np
from peass.backend_numpy.decomposition import perform_least_squares_projection
from peass.backend_numpy.decomposition import run_auditory_analysis_filterbank as run_analysis_np
from peass.backend_numpy.decomposition import run_auditory_synthesis_filterbank as run_synth_np
from peass.backend_numpy.gammatone import GammatoneAnalyzer
from peass.backend_numpy.gammatone import GammatoneDelay
from peass.backend_numpy.gammatone import GammatoneSynthesizer
from peass.backend_numpy.gammatone import fast_resample_poly
from peass.backend_numpy.gammatone import get_equivalent_rectangular_bandwidth_center_frequencies
from peass.backend_numpy.gammatone import get_mixer_gains
from peass.backend_torch.decomposition import decompose_distortion_components as decomp_th
from peass.backend_torch.decomposition import decompose_distortion_components as decomp_torch
from peass.backend_torch.decomposition import extract_target_spatial_distortion_interference_artifacts_torch as ext_th
from peass.backend_torch.decomposition import perform_least_squares_projection_torch
from peass.backend_torch.decomposition import run_auditory_analysis_filterbank_torch as run_analysis_th
from peass.backend_torch.decomposition import run_auditory_synthesis_filterbank_torch as run_synth_th
from peass.backend_torch.gammatone import GammatoneAnalyzerTorch
from peass.backend_torch.gammatone import GammatoneSynthesizerTorch
from peass.backend_torch.gammatone import get_erb_center_frequencies
from peass.backend_torch.utils import fast_resample_poly_torch
from tests.conftest import to_backend_format
from tests.conftest import to_numpy_format

# ---------------------------------------------------------------------------
# Cross-backend parity bars
# ---------------------------------------------------------------------------
#
# Every bar in this file compares NumPy against torch-on-CPU on *the same input*,
# both in float64. That is not a fidelity question -- the two are meant to compute
# the same number -- so these bars sit just above the arithmetic noise rather than at
# the 0.95..0.99 "the signals look similar" level they were originally written at.
#
# Two instruments are used. Whole-waveform *shape* is bounded by a correlation
# deficit `1 - corr`, asserted as the deficit rather than as the correlation because
# at 0.999999999+ an f"{corr:.4f}" failure message renders every one of these as
# "1.0000" and says nothing about how far off the run actually was. Elementwise
# agreement is bounded by `assert_allclose`, as a fraction of the NumPy reference's
# peak.
#
# ---------------------------------------------------------------------------
# The scaling rule for the elementwise bars
# ---------------------------------------------------------------------------
#
# Settled 2026-08-17, and applied to the three filterbank `assert_allclose` bars below
# plus the auditory site, i.e. to every elementwise bar that pass re-derived:
#
#     the bar is always `k * max|reference|`, passed as `atol` with `rtol=0.0`,
#     and `reference` is always the NumPy array.
#
# It is written that way to dodge three separate traps at once:
#
#   1. `atol` alone encodes the amplitude of whatever signal the test happened to
#      use. These differences are genuinely *relative*: scaling the input by 1000x
#      scales max|diff| by 1000x and leaves max|diff|/peak constant to two figures
#      (measured 2026-08-17 at both waveform sites below). A bare absolute number
#      therefore re-tunes itself silently the day someone edits the test input, and
#      that units mismatch -- not a too-tight number -- was the recorded defect in
#      these three bars.
#   2. `rtol` alone would let the value under test set its own bar. `assert_allclose`
#      compares against `atol + rtol * |desired|`, and `desired` is its SECOND
#      positional argument, which at every call site in this file is the torch
#      output; a torch-side blow-up would inflate its own tolerance. Pinning
#      `rtol=0.0` and deriving the scale from a named NumPy-side local also makes the
#      bar independent of argument order, which any use of `rtol` never is.
#   3. A per-element relative bar is the wrong instrument downstream of a subtraction
#      of comparable quantities (ARCHIVE.md, 2026-08-09). The synthesizer output
#      below holds 9 exact zeros in 200 samples, and the analyzer output's smallest
#      element is 4.3e-08 against a peak of 0.72, so a per-element bar there is set
#      by cancellation residue rather than by anything a regression would move. One
#      peak per array is not.
#
# What scaling by the peak costs is recorded too: it is looser for a quiet band than
# a per-band bar would be. At the analyzer site the per-band deviations span
# 1.3e-15..6.4e-14 band-relative (49x) and one global-peak number covers all of them.
# Global peak is still the right call -- a per-band bar would have to be set by the
# quietest band, i.e. at the thinnest margin available -- but the looseness is real.
#
# The rule does NOT yet cover the file's four other `assert_allclose` calls, and is
# written here rather than as a helper so that is visible: the resampler, Toeplitz
# projection, Hann window and ERB grid bars still carry bare absolute tolerances.
# Those were out of scope for the 2026-08-17 pass and are not fragile -- measured the
# same day, they deviate by 0.0, 8.9e-16, 1.1e-16 and 0.0 against bars of 1e-5, 1e-6,
# 1e-7 and 1e-7, i.e. bit-identical or one ULP -- but they are slack, not derived.
#
# ---------------------------------------------------------------------------
# The elementwise bars
# ---------------------------------------------------------------------------
#
# Measured 2026-08-17 on the reference platform (Windows 11, numpy 2.2.5,
# torch 2.12.1+cpu, scipy 1.15.3), as max|diff| / max|reference|:
#
#     site                      at the test's input   spread over inputs       allowed
#     FFT-vs-IIR analyzer                   8.8e-15   8.4e-15..1.7e-14  (8 seeds)  1e-12
#     synthesizer delay/phase               2.0e-14   1.4e-14..2.2e-14  (8 seeds)  1e-12
#     mixer gains                           4.1e-12   9.3e-13..3.1e-11 (6 configs) 1e-09
#
# i.e. 114x / 50x / 247x of margin at the inputs the tests use, and 59x / 44x / 32x
# against the worst input probed. The mixer gains get their own decade for the reason
# test_matlab_regression.py keeps its floors per component: they sit two to three
# decades above the two waveform sites, and a single bar set for the worst of the
# three would let the other two degrade by that ratio without failing.
#
# These three were `atol=1e-6`, `atol=1e-6` and `atol=5e-6`, running at 1.40x, 1.32x
# and 1.63x of slack -- and none of them was one number, because each also silently
# carried `assert_allclose`'s default `rtol=1e-7`; on `atol` alone the ratios were
# 1.36x, 1.20x and 1.59x, which is what their comments quoted. Those comments blamed
# "FFT-domain vs recursive time-domain float limits", "the iterative gain solve
# converging through different library math" and "cumulative numerical phase offsets"
# respectively. All three explanations were wrong: reverting one `dtype=` reproduces
# all three deviations, so there was one cause, not three. It was `torch.fft.fftfreq`
# inheriting torch's default float32 for the filter frequency grid; with
# `dtype=torch.float64` passed explicitly (see `backend_torch/gammatone.py`) the
# deviations drop by 5 to 8 orders of magnitude. This is the first re-derivation
# downward.
#
# Regression probes, run 2026-08-17, because tightening a bar is only worth anything
# if it still separates right from wrong -- all as a fraction of the reference peak:
#
#     probe                                     analyzer   gains     synth
#     baseline, float64 grid                     8.8e-15   4.1e-12   2.0e-14
#     grid reverted to torch's default float32   1.0e-06   7.3e-07   6.5e-07
#     1-sample time shift of the output          1.2e+00      -         -
#     1e-11 relative gain error on the output    1.0e-11      -         -
#
# So the float32-grid bug is now caught by 1e6x / 7e2x / 6e5x. Note what the
# tightening actually buys: at `atol=1e-6` that bug *was* the baseline and the bars
# could not have caught the defect that was in fact present. The analyzer bar now also
# fails a systematic 1e-11 relative gain error by 10x; the next decade down, 1e-12,
# measures 9.96e-13 and passes, so ~1.1e-12 relative is where its sensitivity ends.
_MAX_FILTERBANK_PARITY_PEAK_FRACTION = 1e-12
_MAX_MIXER_GAIN_PARITY_PEAK_FRACTION = 1e-9

# ---------------------------------------------------------------------------
# The correlation-deficit bars
# ---------------------------------------------------------------------------
#
# Re-measured 2026-08-17 on the same platform. All of them have collapsed onto
# `np.corrcoef`'s own resolution: the float64 frequency grid above and the restored
# MATLAB solver ridge in `_solve_hermitian_batch` moved torch onto numpy, and
# `1 - corr` cannot resolve below ~1.1e-16 (one eps at 1.0) however close the two
# arrays get.
#
#     site                                     measured 1-corr        allowed
#     gammatone analysis, per band                     3.3e-16          1e-09
#     gammatone synthesis reconstruction               0.0 (exact)      1e-11
#     decomposition true_target (synthetic)            4.4e-16          1e-09
#     analysis min band (MATLAB WAV)                   6.7e-16          1e-07
#     least-squares projection (MATLAB WAV)            4.4e-16          1e-10
#     synthesis filterbank (MATLAB WAV)                2.2e-16          1e-10
#     decomposition components (MATLAB WAV)            0.0..6.7e-16   dict below
#     auditory internal representation                 0.0 (exact)      1e-12
#
# Only the auditory floor is re-derived here (from 1e-5; see that test). The other
# seven are left exactly where they were, deliberately: a measurement pinned at the
# instrument's resolution floor cannot justify a new number, and quoting a "margin"
# for a 1e-16 measurement would be quoting the resolution, not the agreement. The
# successor instrument is the peak-fraction bar used above, which still has six
# decades of resolution at these levels -- which is why the auditory site now carries
# one *in addition to* its deficit floor. Re-deriving the other seven that way is a
# separate change and is not done here.
#
# The projection floor keeps its enormous slack on its own merits: it solves the same
# float64 system through two different LAPACK wrappers, and while that is exact to
# the last ULP on this input, the Gram can be badly conditioned on other inputs and
# other LAPACK builds reassociate freely.
#
# These are one machine's numbers, CI also runs ubuntu-latest, and ground rule 3 in
# TODO.md exists because platform-specific observations have been written down here as
# universal invariants before. The deficit floors were tightened from 0.95/0.98/0.99
# on 2026-08-15.
_MAX_ANALYSIS_BAND_DEFICIT = 1e-9
_MAX_SYNTHESIS_DEFICIT = 1e-11
_MAX_DECOMPOSITION_DEFICIT = 1e-9
_MAX_REFERENCE_ANALYSIS_BAND_DEFICIT = 1e-7
_MAX_PROJECTION_DEFICIT = 1e-10
_MAX_REFERENCE_SYNTHESIS_DEFICIT = 1e-10

# Tightened 2026-08-17 from 1e-5, four decades, and paired with a max-deviation bar in
# the same test. Measured `1 - corr` is 0.0 exactly on this file's fixture and
# 0.0..2.2e-16 across eight input variations, i.e. at `np.corrcoef`'s resolution, so no
# measurement can place this bar precisely; 1e-12 is ~1e4 above that resolution, so ULP
# jitter in the correlation can never trip it. A deficit goes as the square of the
# relative error, so 1e-12 only catches a divergence above ~1.4e-6 relative -- which is
# exactly why the finer bar below it exists.
#
# The tightening is not cosmetic. With the frequency grid on torch's default float32
# this site measured 4.258e-09, which the old 1e-5 floor passed by 2350x while
# describing that same 4.3e-09 in its comment as the expected value. The new floor
# fails it by 4258x -- verified by re-injecting the float32 grid.
_MAX_AUDITORY_REPRESENTATION_DEFICIT = 1e-12

# Max-deviation bar for the same site, as a fraction of the NumPy representation's
# peak; see the scaling rule above. Measured 2026-08-17 on this file's fixture:
# worst single element 6.02e-10 against a peak of 1546, i.e. 3.9e-13 of peak
# (2.6e-12 of the representation's RMS, which is how the previous comment reported
# it). 1e-10 is 257x above that. The spread it has to cover is documented at the test.
_MAX_AUDITORY_PEAK_FRACTION = 1e-10

# Per component, on the MATLAB reference WAVs. Kept per-component rather than
# global for the same reason as in test_matlab_regression.py: a single bar set for
# the worst component lets the best degrade by that ratio without failing.
#
# Re-measured 2026-08-17; these four collapsed onto `np.corrcoef`'s resolution along
# with the rest, and are left un-re-derived for the reason given above.
#
#     component            measured 1-corr     allowed
#     true_target                  0.0 (exact)   1e-09
#     target_distortion            6.7e-16       1e-08
#     interference                 5.6e-16       1e-08
#     artifacts                    0.0 (exact)   1e-07
#
# `artifacts` gets its own decade because it is the smallest-peak residual and the
# least-squares Gram's minimum eigenvalue is ~3.4e-10, so a 1-ULP perturbation
# upstream arrives ~1e6 larger -- see ARCHIVE.md.
_MAX_COMPONENT_DEFICIT = {
    "true_target": 1e-9,
    "target_distortion": 1e-8,
    "interference": 1e-8,
    "artifacts": 1e-7,
}


@pytest.fixture(scope="module")
def baseline_signals():
    """Generates clean, matching target and estimate signals for parity testing."""
    fs = 16000.0
    duration = 0.5
    num_samples = int(duration * fs)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # 440 Hz target sine, 1200 Hz interference noise
    target = np.sin(2.0 * np.pi * 440.0 * t)[:, np.newaxis]
    interf = np.sin(2.0 * np.pi * 1200.0 * t)[:, np.newaxis]
    estimate = target + 0.1 * interf + 0.01 * np.random.default_rng(seed=7).standard_normal((num_samples, 1))

    return target, interf, estimate, fs


def test_differential_gammatone_analysis(baseline_signals):
    """Verifies bit-level subband parity between the FFT and IIR filterbanks."""
    target, _, _, fs = baseline_signals
    device = torch.device("cpu")

    # 1. Run legacy NumPy filterbank
    analyzer_np = GammatoneAnalyzer(fs, 80.0, 1000.0, 4000.0, 1.0)
    subbands_np = analyzer_np.process(target.ravel())

    # 2. Run new PyTorch filterbank
    target_torch = to_backend_format(target.ravel(), "torch", device)
    analyzer_torch = GammatoneAnalyzerTorch(fs, 80.0, 1000.0, 4000.0, 1.0, device, torch.float64)
    subbands_torch = analyzer_torch.process(target_torch)

    subbands_torch_np = to_numpy_format(subbands_torch)

    # Assert high-fidelity matching across all bands. Worst band measured 9.1e-12
    # (band 1); the deficit falls to ~1e-13 for the mid bands, so the floor is set
    # by the low-frequency end where the IIR cascade has the longest memory.
    for b in range(analyzer_np.center_frequencies.shape[0]):
        corr = np.corrcoef(subbands_np[b].real, subbands_torch_np[b].real)[0, 1]
        deficit = 1.0 - corr
        assert deficit < _MAX_ANALYSIS_BAND_DEFICIT, (
            f"Gammatone analysis band {b} parity deficit is {deficit:.3e}, above the "
            f"{_MAX_ANALYSIS_BAND_DEFICIT:.0e} floor (correlation {corr:.15f})"
        )


def test_differential_gammatone_synthesis(baseline_signals):
    """Verifies reconstruction parity between the NumPy and PyTorch synthesizers."""
    target, _, _, fs = baseline_signals
    device = torch.device("cpu")

    analyzer_np = GammatoneAnalyzer(fs, 80.0, 1000.0, 4000.0, 1.0)
    subbands_np = analyzer_np.process(target.ravel())

    synth_np = GammatoneSynthesizer(analyzer_np, 0.004)
    reconstructed_np = synth_np.process(subbands_np)

    # Run PyTorch counterparts
    target_torch = to_backend_format(target.ravel(), "torch", device)
    analyzer_torch = GammatoneAnalyzerTorch(fs, 80.0, 1000.0, 4000.0, 1.0, device, torch.float64)
    subbands_torch = analyzer_torch.process(target_torch)

    synth_torch = GammatoneSynthesizerTorch(analyzer_torch, 0.004)
    reconstructed_torch = synth_torch.process(subbands_torch)

    recon_torch_np = to_numpy_format(reconstructed_torch)

    # Measured 3.8e-14. The full reconstruction is dominated by the strong mid
    # bands, so it is two decades tighter than the worst individual band above.
    corr = np.corrcoef(reconstructed_np, recon_torch_np)[0, 1]
    deficit = 1.0 - corr
    assert deficit < _MAX_SYNTHESIS_DEFICIT, (
        f"Reconstruction synthesis parity deficit is {deficit:.3e}, above the "
        f"{_MAX_SYNTHESIS_DEFICIT:.0e} floor (correlation {corr:.15f})"
    )


def test_differential_decomposition_pipeline(baseline_signals):
    """Verifies exact output waveform parity of the entire decomposition block."""
    target, interf, estimate, fs = baseline_signals
    device = torch.device("cpu")

    # NumPy Decomposition
    result_np = decomp_np([target, interf], estimate, sampling_frequency_hz=fs)
    wf_np = result_np.waveforms

    # PyTorch Decomposition
    target_torch = to_backend_format(target, "torch", device)
    interf_torch = to_backend_format(interf, "torch", device)
    estimate_torch = to_backend_format(estimate, "torch", device)

    result_torch = decomp_torch([target_torch, interf_torch], estimate_torch, sampling_frequency_hz=fs)
    wf_torch = result_torch.waveforms

    true_target_torch_np = to_numpy_format(wf_torch.true_target)

    # Measured 4.4e-16 on this synthetic pair, 2026-08-17 -- at `np.corrcoef`'s own
    # resolution, so the 1e-9 floor is over-wide; see the header note.
    #
    # Only `true_target` is asserted here, and that is deliberate rather than an
    # oversight. This comment used to record a large backend-dependent onset transient
    # in the other three components (torch peaking at ~1e3, correlations ~0.12) and
    # asked for it to be root-caused before widening the test. It has been: it was the
    # missing MATLAB solver ridge in `_solve_hermitian_batch`, restored in the torch
    # backend, which that function's docstring documents against these very signals.
    #
    # The other three are still not asserted, now for a measured reason: they come out
    # at deficits of 9.1e-06 (`target_distortion`), 5.5e-06 (`interference`) and
    # 1.3e-03 (`artifacts`) here, ten decades worse than the same components on the
    # MATLAB reference WAVs (0.0..6.7e-16, `test_print_decomposition_correlations`
    # below). That is the ridge doing its job rather than a backend divergence: one
    # gammatone band of a pure tone is a single complex exponential, so its Toeplitz
    # block is numerically rank ~1 and the 1e-15 regularization dominates the solution
    # there, which amplifies any last-ULP difference upstream. Adding those three would
    # mean pinning that sensitivity, which is a different test from this one.
    corr = np.corrcoef(wf_np.true_target.ravel(), true_target_torch_np.ravel())[0, 1]
    deficit = 1.0 - corr
    assert deficit < _MAX_DECOMPOSITION_DEFICIT, (
        f"Full decomposition true_target parity deficit is {deficit:.3e}, above the "
        f"{_MAX_DECOMPOSITION_DEFICIT:.0e} floor (correlation {corr:.15f})"
    )


def test_fast_resample_poly_parity():
    """
    Verifies that the PyTorch polyphase resampler mathematically matches the NumPy baseline.
    Converts via `to_numpy_format` to guarantee NumPy 2.x environment compatibility.
    """

    # 1. Generate an impulse test signal
    x_np = np.zeros((2, 100))
    x_np[0, 10] = 1.0
    x_np[1, 50] = 1.0

    up, down = 2, 3

    # 2. Run NumPy resample
    y_np = fast_resample_poly(x_np, up, down, axis=-1)

    # 3. Run PyTorch resample
    x_th = torch.tensor(x_np)
    y_th = to_numpy_format(fast_resample_poly_torch(x_th, up, down, axis=-1))

    # 4. Compare
    np.testing.assert_allclose(
        y_np, y_th, atol=1e-5,
        err_msg="PyTorch resample_poly does not match NumPy."
    )


def test_least_squares_projection_toeplitz_parity():
    """
    Verifies the core Toeplitz matrix construction and least squares projection in PyTorch.
    Converts via `to_numpy_format` to guarantee NumPy 2.x environment compatibility.
    """

    np.random.seed(42)
    num_samples = 20
    filter_half_length = 2
    filter_length = 2 * filter_half_length + 1
    num_sources = 2

    true_sources_np = np.random.randn(num_samples + filter_length - 1, num_sources)
    source_estimates_np = np.random.randn(num_samples, 1)
    analysis_window_np = np.random.rand(num_samples)

    # Run NumPy
    proj_np = perform_least_squares_projection(
        source_estimates_np, true_sources_np, filter_half_length, analysis_window_np
    )

    # Run PyTorch
    proj_th = to_numpy_format(
        perform_least_squares_projection_torch(
            torch.tensor(source_estimates_np),
            torch.tensor(true_sources_np),
            filter_half_length,
            torch.tensor(analysis_window_np)
        )
    )

    # Compare
    np.testing.assert_allclose(
        proj_np, proj_th, atol=1e-6,
        err_msg="PyTorch least squares projection mismatched."
    )


def test_hann_window_overlap_parity():
    """
    Verifies that the Hann window generated for PyTorch overlap-add matches SciPy.
    Converts via `to_numpy_format` to guarantee NumPy 2.x environment compatibility.
    """

    window_length = 50
    win_np = signal.windows.hann(window_length, sym=False)
    win_th = to_numpy_format(torch.hann_window(window_length, periodic=True, dtype=torch.float64))

    np.testing.assert_allclose(
        win_np, win_th, atol=1e-7,
        err_msg="PyTorch Hann window does not match SciPy."
    )


def test_gammatone_analyzer_fft_vs_iir_parity():
    """
    Verifies that the PyTorch FFT-based Gammatone filterbank perfectly matches
    the NumPy IIR-based filterbank down to float precision.
    """
    fs = 16000.0
    np.random.seed(42)
    sig_np = np.random.randn(1000)

    analyzer_np = GammatoneAnalyzer(fs, 80.0, 1000.0, 4000.0, 1.0)
    out_np = analyzer_np.process(sig_np)

    analyzer_th = GammatoneAnalyzerTorch(fs, 80.0, 1000.0, 4000.0, 1.0, torch.device("cpu"), torch.float64)
    out_th = to_numpy_format(analyzer_th.process(torch.tensor(sig_np)))

    # A fraction of the NumPy reference's peak, with `rtol` pinned to 0 -- the peak is
    # read off the reference array by name so the value under test cannot scale its own
    # bar. See the scaling rule at the top of this file for why not a bare `atol` and
    # why not `rtol`; 13 of these 24000 elements sit below 1e-6 of the peak, so a
    # per-element relative bar here would be reporting cancellation residue.
    #
    # Measured 2026-08-17: max|diff| 6.35e-15 against a peak of 0.7238, i.e. 8.8e-15 of
    # peak, so this runs at 114x of margin -- 59x against the worst of 8 input seeds
    # (1.7e-14). Was `atol=1e-6` at 1.4x, blamed on "FFT-domain vs recursive
    # time-domain float limits"; the real cause was the float32 frequency grid, and
    # reverting that grid measures 1.02e-06 of peak, which this bar fails by 1e6x.
    reference_peak = float(np.abs(out_np).max())
    np.testing.assert_allclose(
        out_np, out_th, rtol=0.0,
        atol=_MAX_FILTERBANK_PARITY_PEAK_FRACTION * reference_peak,
        err_msg="GammatoneAnalyzerTorch FFT output does not match NumPy IIR output."
    )


def test_gammatone_mixer_gains_parity():
    """
    Verifies that the iterative synthesis gains calculation perfectly matches NumPy.
    """

    fs = 16000.0
    analyzer_np = GammatoneAnalyzer(fs, 80.0, 1000.0, 4000.0, 1.0)
    delay_np = GammatoneDelay(analyzer_np, int(round(0.004 * fs)))
    gains_np = get_mixer_gains(analyzer_np, delay_np, 100)

    analyzer_th = GammatoneAnalyzerTorch(fs, 80.0, 1000.0, 4000.0, 1.0, torch.device("cpu"), torch.float64)
    synth_th = GammatoneSynthesizerTorch(analyzer_th, 0.004)
    gains_th = to_numpy_format(synth_th.gains)

    # Same form as the analyzer bar above: a fraction of the NumPy reference's peak,
    # `rtol` pinned to 0. A literal `rtol` would be safe against near-zeros *here*
    # specifically -- 24 gains spanning 0.615..1.146, so peak-scaled and
    # per-element-relative differ by at most 1.86x -- but `rtol` scales by `desired`,
    # which is the torch gains under test, so this site follows the same rule as the
    # rest of the file rather than inventing a second one.
    #
    # Measured 2026-08-17: max|diff| 4.64e-12 against a peak gain of 1.1459, i.e.
    # 4.1e-12 of peak, so this runs at 247x of margin -- and 32x against the worst of
    # six other filterbank/delay configurations probed (3.1e-11, at fs=44100 with 30
    # bands). It carries its own constant, a decade above the two waveform bars, for
    # the reason stated at the top of this file.
    #
    # Was `atol=1e-6` at 1.3x, the narrowest bar in the file, blamed on "the iterative
    # gain solve converging through different library math". That was wrong on both
    # counts. The difference is not the solve: measured 2026-08-17, the two backends'
    # `delays` are bit-identical int64 and their unit-modulus phase alignment factors
    # already differ by 4.5e-12, the same order as the gains, so the 100-iteration fixed
    # point inherits the difference rather than creating it. And it is not library math:
    # it entered through the float32 frequency grid in `_get_synthesizer_params_torch`,
    # which measures 7.3e-07 of peak when reverted, i.e. 725x over this bar.
    reference_peak = float(np.abs(gains_np).max())
    np.testing.assert_allclose(
        gains_np, gains_th, rtol=0.0,
        atol=_MAX_MIXER_GAIN_PARITY_PEAK_FRACTION * reference_peak,
        err_msg="GammatoneSynthesizerTorch gains do not match NumPy."
    )


def test_gammatone_synthesizer_delay_parity():
    """
    Verifies the delay line logic and phase alignments between PyTorch and NumPy.
    """
    fs = 16000.0
    np.random.seed(42)
    # Using small number of bands and samples to isolate delay logic
    analyzer_np = GammatoneAnalyzer(fs, 1000.0, 1000.0, 2000.0, 1.0)
    synth_np = GammatoneSynthesizer(analyzer_np, 0.004)

    analyzer_th = GammatoneAnalyzerTorch(fs, 1000.0, 1000.0, 2000.0, 1.0, torch.device("cpu"), torch.float64)
    synth_th = GammatoneSynthesizerTorch(analyzer_th, 0.004)

    subbands = np.random.randn(len(analyzer_np.center_frequencies), 200) + 1j * np.random.randn(
        len(analyzer_np.center_frequencies), 200)

    out_np = synth_np.process(subbands)
    out_th = to_numpy_format(synth_th.process(torch.tensor(subbands)))

    # A fraction of the NumPy reference's peak, `rtol` pinned to 0. `rtol` would be
    # doubly wrong here: it scales by the torch output, and this output contains 9
    # exact zeros in 200 samples, where a relative bar is not defined at all.
    #
    # Measured 2026-08-17: max|diff| 9.68e-14 against a peak of 4.8588, i.e. 2.0e-14 of
    # peak, so this runs at 50x of margin -- 44x against the worst of 8 input seeds
    # (2.2e-14). Was `atol=5e-6` at 1.6x, blamed on "cumulative numerical phase
    # offsets"; the real cause was the float32 frequency grid, which measures 6.5e-07
    # of peak when reverted, i.e. 6e5x over this bar. The group-delay drift this test
    # is named for is far louder still: a 1-sample shift of the output measures 1.29 of
    # peak, and test_synthesis_temporal_alignment_parity catches that class exactly.
    reference_peak = float(np.abs(out_np).max())
    np.testing.assert_allclose(
        out_np, out_th, rtol=0.0,
        atol=_MAX_FILTERBANK_PARITY_PEAK_FRACTION * reference_peak,
        err_msg="GammatoneSynthesizerTorch delay/phase alignment does not match NumPy."
    )


def test_get_erb_center_frequencies_parity():
    """
    Verifies that the ERB center frequencies generated by torch.arange perfectly match np.arange.
    """

    cfs_np = get_equivalent_rectangular_bandwidth_center_frequencies(1.0, 20.0, 1000.0, 8000.0)
    cfs_th = to_numpy_format(
        get_erb_center_frequencies(1.0, 20.0, 1000.0, 8000.0, device=torch.device("cpu"), dtype=torch.float64))

    np.testing.assert_allclose(
        cfs_np, cfs_th, atol=1e-7,
        err_msg="PyTorch ERB center frequencies do not match NumPy."
    )


@pytest.mark.regression
def test_pipeline_step_1_analysis_parity(matlab_ref_resources):
    """
    Diagnostic Test - Step 1: Subband Analysis
    Loads 'targetSrc.wav', decomposes it into Gammatone subbands using both backends,
    and checks the minimum cross-correlation across all subbands.
    """

    # 1. Load target audio
    target_src, fs = sf.read(matlab_ref_resources / "targetSrc.wav")
    ch0 = target_src[:, 0]  # Focus on Channel 0

    # 2. Run NumPy Analysis
    subbands_np, analyzer_np, _ = run_analysis_np(ch0, float(fs))

    # 3. Run PyTorch Analysis
    ch0_th = torch.tensor(ch0, dtype=torch.float64)
    subbands_th, analyzer_th, _ = run_analysis_th(ch0_th, float(fs))

    # 4. Compare bands
    assert len(subbands_np) == len(subbands_th), "Mismatch in total number of subbands."

    correlations = []
    for b in range(len(subbands_np)):
        np_band = subbands_np[b]
        th_band = to_numpy_format(subbands_th[b])

        # Take real and imaginary correlations separately
        corr_r = np.corrcoef(np_band.real, th_band.real)[0, 1]
        corr_i = np.corrcoef(np_band.imag, th_band.imag)[0, 1]

        correlations.append(min(corr_r, corr_i))

    # Measured worst band 2.8e-10; the next-worst is 2.3e-11, so one band sits 12x
    # above the rest of the distribution. The floor is set with that outlier's
    # spread in mind rather than against the bulk.
    min_corr = min(correlations)
    deficit = 1.0 - min_corr
    print(f"\n---> ANALYSIS STEP MIN CORRELATION DEFICIT: {deficit:.3e}")
    assert deficit < _MAX_REFERENCE_ANALYSIS_BAND_DEFICIT, (
        f"Analysis phase introduced mismatch. Worst band parity deficit is "
        f"{deficit:.3e}, above the {_MAX_REFERENCE_ANALYSIS_BAND_DEFICIT:.0e} floor "
        f"(correlation {min_corr:.15f})"
    )


@pytest.mark.regression
def test_pipeline_step_2_projection_parity(matlab_ref_resources):
    """
    Diagnostic Test - Step 2: Least-Squares Subspace Projection
    Bypasses PyTorch's subband analysis by feeding matching NumPy subbands directly
    into PyTorch's projection solver to check if projection introduces the discrepancy.
    """

    # 1. Load baseline inputs
    target, fs = sf.read(matlab_ref_resources / "targetSrc.wav")
    estimate, _ = sf.read(matlab_ref_resources / "targetEstimate.wav")

    # Group into multi-source structure matching internal format
    # numpy expects (samples, channels, sources) -> (N, 2, 2)
    true_sources_np = np.stack([target, np.zeros_like(target)], axis=2)
    source_estimates_np = estimate[:, :, np.newaxis]

    # Use standard default parameters for band-10 to test the solver specifically
    filter_length = 3
    window_length = 10
    hop_size = 5

    # 2. Run NumPy Projection
    _, np_spat, _, _ = ext_np(true_sources_np, source_estimates_np, filter_length, window_length, hop_size)

    # 3. Run PyTorch Projection using matching NumPy subband arrays
    true_sources_th = torch.tensor(true_sources_np, dtype=torch.float64)
    source_estimates_th = torch.tensor(source_estimates_np, dtype=torch.float64)

    _, th_spat, _, _ = ext_th(true_sources_th, source_estimates_th, filter_length, window_length, hop_size)
    th_spat_np = to_numpy_format(th_spat)

    # Measured 2.1e-15 -- the two solvers agree to the last few ULP on this input.
    # The floor keeps five decades of slack anyway; see the header note on LAPACK.
    corr = np.corrcoef(np_spat.ravel(), th_spat_np.ravel())[0, 1]
    deficit = 1.0 - corr
    print(f"\n---> PROJECTION STEP CORRELATION DEFICIT: {deficit:.3e}")
    assert deficit < _MAX_PROJECTION_DEFICIT, (
        f"Projection solver introduced mismatch. Parity deficit is {deficit:.3e}, "
        f"above the {_MAX_PROJECTION_DEFICIT:.0e} floor (correlation {corr:.15f})"
    )


@pytest.mark.regression
def test_pipeline_step_3_synthesis_parity(matlab_ref_resources):
    """
    Diagnostic Test - Step 3: Synthesis Reconstruction
    Bypasses both Analysis and Projection by feeding identical NumPy subband
    projections directly into PyTorch's synthesis filterbank.
    """

    # 1. Run NumPy Analysis to get subbands and a matched analyzer instance
    target, fs = sf.read(matlab_ref_resources / "targetSrc.wav")
    ch0 = target[:, 0]
    subbands_np, analyzer_np, _ = run_analysis_np(ch0, float(fs))

    # 2. Run NumPy Synthesis
    synth_np, _ = run_synth_np(subbands_np, analyzer_np)

    # 3. Build matching PyTorch Analyzer and Synthesizer

    # Reconstruct the upsampled analyzer with EXACT matching parameters
    fs_upsampled = analyzer_np.sampling_frequency_hz
    analyzer_th = GammatoneAnalyzerTorch(
        fs_upsampled, 20.0, 1000.0, analyzer_np.upper_cutoff_frequency_hz, 1.0,
        torch.device("cpu"), torch.float64
    )
    # Ensure properties match
    analyzer_th.original_sampling_frequency_hz = analyzer_np.original_sampling_frequency_hz

    # Wrap subbands
    subbands_th = [torch.tensor(b, dtype=torch.complex128) for b in subbands_np]

    synth_th = to_numpy_format(run_synth_th(subbands_th, analyzer_th))

    # Sync lengths
    min_len = min(len(synth_np), len(synth_th))
    corr = np.corrcoef(synth_np[:min_len], synth_th[:min_len])[0, 1]

    # Measured 1.2e-13. Both backends are fed identical NumPy subbands here, so the
    # only difference is the synthesis path itself.
    deficit = 1.0 - corr
    print(f"\n---> SYNTHESIS STEP CORRELATION DEFICIT: {deficit:.3e}")
    assert deficit < _MAX_REFERENCE_SYNTHESIS_DEFICIT, (
        f"Synthesis filterbank introduced mismatch. Parity deficit is {deficit:.3e}, "
        f"above the {_MAX_REFERENCE_SYNTHESIS_DEFICIT:.0e} floor "
        f"(correlation {corr:.15f})"
    )


@pytest.mark.regression
def test_print_decomposition_correlations(matlab_ref_resources):
    """
    Diagnostic Test: Waveform Correlation Inspector
    Runs both Numpy and PyTorch decompositions on real files and prints the exact
    correlation matrices to isolate which components differ and by how much.
    """

    # 1. Load baseline inputs
    target, fs = sf.read(matlab_ref_resources / "targetSrc.wav")
    interf1, _ = sf.read(matlab_ref_resources / "interfSrc1.wav")
    interf2, _ = sf.read(matlab_ref_resources / "interfSrc2.wav")
    estimate, _ = sf.read(matlab_ref_resources / "targetEstimate.wav")

    # 2. Run NumPy Decomposition
    config = DecompositionConfiguration(shade_in_milliseconds=10.0, shade_out_milliseconds=10.0)
    res_np = decomp_np([target, interf1, interf2], estimate, config, float(fs))
    wf_np = res_np.waveforms

    # 3. Run PyTorch Decomposition
    target_th = torch.tensor(target, dtype=torch.float64)
    interf1_th = torch.tensor(interf1, dtype=torch.float64)
    interf2_th = torch.tensor(interf2, dtype=torch.float64)
    estimate_th = torch.tensor(estimate, dtype=torch.float64)
    res_th = decomp_th([target_th, interf1_th, interf2_th], estimate_th, config, float(fs))
    wf_th = res_th.waveforms

    # Convert PyTorch to NumPy
    py_target = to_numpy_format(wf_th.true_target)
    py_distortion = to_numpy_format(wf_th.target_distortion)
    py_interf = to_numpy_format(wf_th.interference)
    py_artif = to_numpy_format(wf_th.artifacts)

    # 4. Compare PyTorch vs NumPy
    corr_true_pt_vs_np = np.corrcoef(wf_np.true_target.ravel(), py_target.ravel())[0, 1]
    corr_dist_pt_vs_np = np.corrcoef(wf_np.target_distortion.ravel(), py_distortion.ravel())[0, 1]
    corr_interf_pt_vs_np = np.corrcoef(wf_np.interference.ravel(), py_interf.ravel())[0, 1]
    corr_artif_pt_vs_np = np.corrcoef(wf_np.artifacts.ravel(), py_artif.ravel())[0, 1]

    # 5. Compare PyTorch vs MATLAB Reference Files
    mat_true, _ = sf.read(matlab_ref_resources / "targetEstimate_true.wav")
    mat_dist, _ = sf.read(matlab_ref_resources / "targetEstimate_eTarget.wav")
    mat_interf, _ = sf.read(matlab_ref_resources / "targetEstimate_eInterf.wav")
    mat_artif, _ = sf.read(matlab_ref_resources / "targetEstimate_eArtif.wav")

    min_len = min(len(py_target), len(mat_true))
    corr_true_pt_vs_mat = np.corrcoef(py_target[:min_len].ravel(), mat_true[:min_len].ravel())[0, 1]
    corr_dist_pt_vs_mat = np.corrcoef(py_distortion[:min_len].ravel(), mat_dist[:min_len].ravel())[0, 1]
    corr_interf_pt_vs_mat = np.corrcoef(py_interf[:min_len].ravel(), mat_interf[:min_len].ravel())[0, 1]
    corr_artif_pt_vs_mat = np.corrcoef(py_artif[:min_len].ravel(), mat_artif[:min_len].ravel())[0, 1]

    # 6. Compare NumPy vs MATLAB Reference Files (Control)
    corr_true_np_vs_mat = np.corrcoef(wf_np.true_target[:min_len].ravel(), mat_true[:min_len].ravel())[0, 1]
    corr_dist_np_vs_mat = np.corrcoef(wf_np.target_distortion[:min_len].ravel(), mat_dist[:min_len].ravel())[0, 1]
    corr_interf_np_vs_mat = np.corrcoef(wf_np.interference[:min_len].ravel(), mat_interf[:min_len].ravel())[0, 1]
    corr_artif_np_vs_mat = np.corrcoef(wf_np.artifacts[:min_len].ravel(), mat_artif[:min_len].ravel())[0, 1]

    print("\n" + "=" * 50)
    print("      DECOMPOSITION CORRELATION REPORT")
    print("=" * 50)
    print("  Component         | PyTorch vs NumPy | PyTorch vs MATLAB | NumPy vs MATLAB")
    print("  ------------------|------------------|-------------------|-----------------")
    print(
        f"  true_target       | {corr_true_pt_vs_np:.6f}         | {corr_true_pt_vs_mat:.6f}          | {corr_true_np_vs_mat:.6f}")
    print(
        f"  target_distortion | {corr_dist_pt_vs_np:.6f}         | {corr_dist_pt_vs_mat:.6f}          | {corr_dist_np_vs_mat:.6f}")
    print(
        f"  interference      | {corr_interf_pt_vs_np:.6f}         | {corr_interf_pt_vs_mat:.6f}          | {corr_interf_np_vs_mat:.6f}")
    print(
        f"  artifacts         | {corr_artif_pt_vs_np:.6f}         | {corr_artif_pt_vs_mat:.6f}          | {corr_artif_np_vs_mat:.6f}")
    print("=" * 50 + "\n")

    # Assert the PyTorch-vs-NumPy parity deficit per component, against the floors
    # documented at the top of this file. Note that only the first column of the
    # table above is asserted: the *-vs-MATLAB columns are fidelity, which is what
    # tests/regression/test_matlab_regression.py exists to bound.
    measured = {
        "true_target": corr_true_pt_vs_np,
        "target_distortion": corr_dist_pt_vs_np,
        "interference": corr_interf_pt_vs_np,
        "artifacts": corr_artif_pt_vs_np,
    }
    for label, corr in measured.items():
        max_deficit = _MAX_COMPONENT_DEFICIT[label]
        deficit = 1.0 - corr
        assert deficit < max_deficit, (
            f"{label} torch-vs-numpy parity deficit is {deficit:.3e}, above the "
            f"{max_deficit:.0e} floor (correlation {corr:.15f})"
        )


def test_synthesis_output_length_parity(matlab_ref_resources):
    """
    Verifies if PyTorch and NumPy synthesis filterbank outputs have identical lengths.
    If they differ, it proves that PyTorch uses an incorrect group delay (e.g. 4ms instead of 41.67ms),
    causing severe temporal offsets in the reconstructed target waveform.
    """

    target, fs = sf.read(matlab_ref_resources / "targetSrc.wav")
    ch0 = target[:, 0]

    # 1. Run NumPy
    subbands_np, analyzer_np, _ = run_analysis_np(ch0, float(fs))
    synth_np, _ = run_synth_np(subbands_np, analyzer_np)

    # 2. Run PyTorch
    fs_upsampled = analyzer_np.sampling_frequency_hz
    analyzer_th = GammatoneAnalyzerTorch(
        fs_upsampled, 20.0, 1000.0, analyzer_np.upper_cutoff_frequency_hz, 1.0,
        torch.device("cpu"), torch.float64
    )
    analyzer_th.original_sampling_frequency_hz = analyzer_np.original_sampling_frequency_hz
    subbands_th = [torch.tensor(b, dtype=torch.complex128) for b in subbands_np]

    synth_th = run_synth_th(subbands_th, analyzer_th)

    # 3. Assert matching length
    assert len(synth_np) == len(synth_th), (
        f"Length mismatch: NumPy output has {len(synth_np)} samples, "
        f"but PyTorch output has {len(synth_th)} samples. "
        f"This indicates a massive group delay mismatch in PyTorch."
    )


def test_synthesis_temporal_alignment_parity(matlab_ref_resources):
    """
    Finds the exact cross-correlation lag between PyTorch and NumPy synthesis outputs.
    If the lag is non-zero, it mathematically proves a temporal shift in the PyTorch output.
    """

    target, fs = sf.read(matlab_ref_resources / "targetSrc.wav")
    ch0 = target[:, 0]

    # 1. Run NumPy
    subbands_np, analyzer_np, _ = run_analysis_np(ch0, float(fs))
    synth_np, _ = run_synth_np(subbands_np, analyzer_np)

    # 2. Run PyTorch
    fs_upsampled = analyzer_np.sampling_frequency_hz
    analyzer_th = GammatoneAnalyzerTorch(
        fs_upsampled, 20.0, 1000.0, analyzer_np.upper_cutoff_frequency_hz, 1.0,
        torch.device("cpu"), torch.float64
    )
    analyzer_th.original_sampling_frequency_hz = analyzer_np.original_sampling_frequency_hz
    subbands_th = [torch.tensor(b, dtype=torch.complex128) for b in subbands_np]

    synth_th = to_numpy_format(run_synth_th(subbands_th, analyzer_th))

    # Pad shorter signal to compare correlation
    L = max(len(synth_np), len(synth_th))
    s_np = np.pad(synth_np, (0, L - len(synth_np)))
    s_th = np.pad(synth_th, (0, L - len(synth_th)))

    # 3. Find cross-correlation lag
    correlation = signal.correlate(s_np, s_th, mode='full')
    lags = signal.correlation_lags(len(s_np), len(s_th), mode='full')
    best_lag = lags[np.argmax(np.abs(correlation))]

    # Regression guard: both backends must use the same synthesis group delay
    # (1000/fs). A non-zero lag means the torch synthesis delay has drifted.
    assert best_lag == 0, (
        f"Temporal shift detected: PyTorch synthesis is shifted by {best_lag} "
        f"samples relative to NumPy (synthesis group-delay mismatch)."
    )


@pytest.mark.regression
def test_differential_auditory_internal_representation(baseline_signals):
    """
    Parity check for the auditory model (haircell transduction + nerve adaptation
    + modulation filtering) between the NumPy and torch backends. The torch forward
    now tracks numpy to float64 arithmetic noise: measured 2026-08-17, `1 - corr` is
    0.0 exactly and the largest single-sample difference is 6.02e-10, 3.9e-13 of the
    representation's peak.

    This used to be documented as the one site where the backends are *not* trying to
    be bit-identical -- 4.3e-9 of deficit and a worst sample ~0.2% of RMS, attributed
    to torch's differentiable surrogates. That attribution was wrong. The forward pass
    contains no surrogate: `_straight_through_max` returns
    `hard.detach() + (soft - soft.detach())`, which is exactly `torch.maximum`, and on
    this path (CPU, float64, no gradient) the fused Numba kernel runs instead and the
    softplus is not evaluated at all. Measured across all three execution paths -- fused
    Numba kernel, scripted torch loop, and the straight-through loop under
    `requires_grad` -- the deviation is 3.80e-13..3.89e-13 of peak, a 2% spread, and
    3.68e-13..3.89e-13 across three noise seeds. What was actually there was the float32
    frequency grid, as at the filterbank sites: reverting it reproduces 2.195e-03 of RMS
    exactly.
    """
    from peass.backend_numpy.auditory_model import generate_auditory_internal_representation as air_np
    from peass.backend_torch.auditory_model import generate_auditory_internal_representation_torch as air_th

    _, _, estimate_np, fs = baseline_signals
    sig_np = estimate_np[:, 0]

    rep_np, fr_np = air_np(sig_np, fs)
    rep_th, fr_th = air_th(torch.tensor(sig_np, dtype=torch.float64), fs)
    rep_th_np = to_numpy_format(rep_th)

    # torch returns a batched representation (B, bands, samples, mods); take batch 0
    if rep_th_np.ndim == rep_np.ndim + 1:
        rep_th_np = rep_th_np[0]

    assert fr_np == fr_th, f"Representation sampling rate mismatch: {fr_np} vs {fr_th}"

    min_len = min(rep_np.shape[1], rep_th_np.shape[1])
    a = rep_np[:, :min_len, :].ravel()
    b = rep_th_np[:, :min_len, :].ravel()

    # Two bars, coarse then fine. The correlation deficit measures 0.0 exactly here
    # and 0.0..2.2e-16 across eight input variations, i.e. it has run out of
    # resolution, so it is now only the gross-shape guard (see its constant); the
    # max-deviation bar below is what actually bounds the agreement.
    corr = np.corrcoef(a, b)[0, 1]
    deficit = 1.0 - corr
    assert deficit < _MAX_AUDITORY_REPRESENTATION_DEFICIT, (
        f"Auditory internal representation parity deficit is {deficit:.3e}, above "
        f"the {_MAX_AUDITORY_REPRESENTATION_DEFICIT:.0e} floor "
        f"(correlation {corr:.15f})"
    )

    # Fraction of the NumPy representation's peak, per the scaling rule at the top of
    # this file. Measured 2026-08-17: worst element 6.02e-10 against a peak of 1546,
    # i.e. 3.9e-13 of peak, so this runs at 257x of margin. The deviation distribution
    # is smooth rather than one outlier (the max is 1.1x the 99.99th percentile), so
    # this is a stable statistic.
    #
    # The spread the bar has to cover is dominated by signal *length*, not by the
    # platform: the deviation accumulates through the 5-stage adaptation recurrence, so
    # the same fixture at 1.0 s or 2.0 s measures 1.4e-11 of peak, 35x this 0.5 s
    # figure, which 1e-10 still covers by 7x. Seed and execution path move it by under
    # 10%. If this fixture is ever lengthened past ~2 s, re-derive rather than assume.
    #
    # It fails the float32-grid regression by 3e6x: that measures 3.23e-04 of peak here,
    # equivalently the 2.195e-03 of RMS this test's comment used to quote as normal.
    reference_peak = float(np.abs(a).max())
    np.testing.assert_allclose(
        a, b, rtol=0.0, atol=_MAX_AUDITORY_PEAK_FRACTION * reference_peak,
        err_msg=(
            "Auditory internal representation deviates from NumPy by more than "
            f"{_MAX_AUDITORY_PEAK_FRACTION:.0e} of its peak ({reference_peak:.4e})"
        )
    )
