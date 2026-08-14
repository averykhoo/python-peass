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
# Cross-backend parity floors
# ---------------------------------------------------------------------------
#
# Every correlation bar in this file compares NumPy against torch-on-CPU on *the
# same input*, both in float64. That is not a fidelity question -- the two are
# meant to compute the same number -- so these floors should sit just above the
# arithmetic noise, not at the 0.95..0.99 "the signals look similar" level they
# were written at. Measured 2026-08-15 on the reference platform:
#
#     site                                   measured 1-corr   allowed   margin
#     gammatone analysis, per band                   9.1e-12     1e-09     110x
#     gammatone synthesis reconstruction             3.8e-14     1e-11     263x
#     decomposition true_target (synthetic)          1.8e-12     1e-09     547x
#     analysis min band (MATLAB WAV)                 2.8e-10     1e-07     363x
#     least-squares projection (MATLAB WAV)          2.1e-15     1e-10   47000x
#     synthesis filterbank (MATLAB WAV)              1.2e-13     1e-10     843x
#     decomposition components (MATLAB WAV)   see the dict below
#     auditory internal representation               4.3e-09     1e-05    2350x
#
# All are expressed as the largest tolerated `1 - corr` and asserted as the
# deficit rather than the correlation. At 0.999999999+ an f"{corr:.4f}" failure
# message renders every one of these as "1.0000" and tells you nothing about how
# far off the run actually was.
#
# Margins are set by how the underlying difference arises, not uniformly:
#
#   * The filterbank sites compare an FFT implementation against a recursive IIR
#     one, so they carry a genuine ~1e-6 relative difference (see the atol=1e-6
#     `assert_allclose` tests below). A correlation deficit goes as the *square*
#     of that relative error, so a 10x platform-to-platform swing in the error
#     moves the deficit 100x -- hence ~1e2..1e3 margins there.
#   * The projection site solves the same float64 system through two different
#     LAPACK wrappers. It is exact to 2e-15 here, but the Gram can be badly
#     conditioned on other inputs and other LAPACK builds reassociate freely, so
#     it keeps a deliberately enormous margin rather than a snug one.
#   * The auditory model is the one site where the backends are *not* trying to
#     be bit-identical: torch uses differentiable surrogates (softplus, straight-
#     through max, FIR-truncated IIRs), documented in README as matching by high
#     correlation rather than to floating-point precision. It still measures
#     4.3e-9, but its worst-case sample difference is ~0.2% of RMS -- a
#     straight-through branch flipping on another platform is a real possibility,
#     so it gets the loosest floor of the file by three decades.
#
# These are one machine's numbers, CI also runs ubuntu-latest, and ground rule 3
# in TODO.md exists because platform-specific observations have been written down
# here as universal invariants before. Tightened from 0.95/0.98/0.99 on
# 2026-08-15; the loosest of those had ~5e7 of unused headroom.
_MAX_ANALYSIS_BAND_DEFICIT = 1e-9
_MAX_SYNTHESIS_DEFICIT = 1e-11
_MAX_DECOMPOSITION_DEFICIT = 1e-9
_MAX_REFERENCE_ANALYSIS_BAND_DEFICIT = 1e-7
_MAX_PROJECTION_DEFICIT = 1e-10
_MAX_REFERENCE_SYNTHESIS_DEFICIT = 1e-10
_MAX_AUDITORY_REPRESENTATION_DEFICIT = 1e-5

# Per component, on the MATLAB reference WAVs. Kept per-component rather than
# global for the same reason as in test_matlab_regression.py: a single bar set for
# the worst component lets the best degrade by that ratio without failing.
#
#     component            measured 1-corr     allowed     margin
#     true_target                  6.8e-12       1e-09       146x
#     target_distortion            7.5e-11       1e-08       133x
#     interference                 3.0e-11       1e-08       334x
#     artifacts                    2.0e-10       1e-07       498x
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

    # Measured 1.8e-12 on this synthetic pair.
    #
    # Only `true_target` is asserted here, and that is deliberate rather than an
    # oversight: on this particular input the other three components carry a large
    # backend-dependent onset transient in the first ~60 samples (torch peaks at
    # ~1e3 against a numpy RMS of ~1e-1), which cancels between `target_distortion`
    # and `interference` -- both backends reconstruct the estimate to the same
    # 1.1 max error -- but leaves their individual correlations at ~0.12. The same
    # components measure 1e-11 parity on the MATLAB reference WAVs
    # (`test_print_decomposition_correlations` below), so this is specific to the
    # short synthetic pair, not a general backend divergence. Do not widen this
    # test to the other components without first root-causing that transient.
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

    # FRAGILE -- do not tighten. 1e-6 accounts for FFT-domain vs recursive
    # time-domain float limits, and the measured max|diff| is 7.4e-7 against a peak
    # of 0.72, i.e. only 1.36x of margin. This is the tightest-running bar in the
    # file. If it ever fails on another platform the correct response is to raise
    # the tolerance to ~1e-5, not to hunt for a regression; a real regression here
    # shows up orders of magnitude larger.
    np.testing.assert_allclose(
        out_np, out_th, atol=1e-6,
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

    # FRAGILE -- do not tighten. Measured max|diff| 8.3e-7 against a peak gain of
    # 1.15, i.e. 1.2x of margin, the narrowest in the file. The difference is the
    # iterative gain solve converging through different library math, not a
    # regression signal; raise to ~1e-5 if a platform trips it.
    np.testing.assert_allclose(
        gains_np, gains_th, atol=1e-6,
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

    # FRAGILE -- do not tighten. 5e-6 accounts for cumulative numerical phase
    # offsets; measured max|diff| is 3.2e-6 against a peak of 4.86, i.e. 1.59x of
    # margin. Same reasoning as the two bars above.
    np.testing.assert_allclose(
        out_np, out_th, atol=5e-6,
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
    + modulation filtering) between the NumPy and torch backends. With the FFT
    haircell and the straight-through max in the adaptation loop, the torch forward
    tracks numpy to 4.3e-9 correlation deficit -- close, but not to floating-point
    precision: the largest single-sample difference is ~0.2% of the representation
    RMS. (A regression to the plain softplus max drops this to ~0.9 on longer
    signals.)
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
    # Measured 4.3e-9 -- three decades looser than anything else in this file, and
    # deliberately floored looser still. This is the one path where the backends do
    # not attempt bit-identity: the worst single-sample difference is ~0.2% of the
    # representation RMS, which is what a straight-through max taking the other
    # branch looks like, and that branch choice is platform-sensitive in a way the
    # pure filterbank sites are not. A regression to the plain softplus max drops
    # this to ~0.9 on longer signals, so the floor still catches the failure it was
    # written for with ~1e5 to spare.
    corr = np.corrcoef(a, b)[0, 1]
    deficit = 1.0 - corr
    assert deficit < _MAX_AUDITORY_REPRESENTATION_DEFICIT, (
        f"Auditory internal representation parity deficit is {deficit:.3e}, above "
        f"the {_MAX_AUDITORY_REPRESENTATION_DEFICIT:.0e} floor "
        f"(correlation {corr:.15f})"
    )
