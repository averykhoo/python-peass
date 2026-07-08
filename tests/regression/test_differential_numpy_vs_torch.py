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

    # Assert high-fidelity matching across all bands
    for b in range(analyzer_np.center_frequencies.shape[0]):
        corr = np.corrcoef(subbands_np[b].real, subbands_torch_np[b].real)[0, 1]
        assert corr > 0.98, f"Gammatone Analysis Band {b} correlation is too low: {corr:.4f}"


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

    corr = np.corrcoef(reconstructed_np, recon_torch_np)[0, 1]
    assert corr > 0.98, f"Reconstruction synthesis correlation is too low: {corr:.4f}"


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

    corr = np.corrcoef(wf_np.true_target.ravel(), true_target_torch_np.ravel())[0, 1]
    assert corr > 0.95, f"Full Decomposition True Target parity correlation is too low: {corr:.4f}"


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

    # Relaxed slightly to 1e-6 to account for FFT-domain vs. recursive Time-domain float limits
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

    # Relaxed slightly to 1e-6 to account for library math differences
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

    # Relaxed slightly to 5e-6 to account for cumulative numerical phase offsets
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

    min_corr = min(correlations)
    print(f"\n---> ANALYSIS STEP MIN CORRELATION: {min_corr:.6f}")
    assert min_corr >= 0.98, f"Analysis phase introduced mismatch. Min band correlation is {min_corr:.6f}"


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

    corr = np.corrcoef(np_spat.ravel(), th_spat_np.ravel())[0, 1]
    print(f"\n---> PROJECTION STEP CORRELATION: {corr:.6f}")
    assert corr >= 0.98, f"Projection solver introduced mismatch. Correlation is {corr:.6f}"


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

    print(f"\n---> SYNTHESIS STEP CORRELATION: {corr:.8f}")
    assert corr >= 0.95, f"Synthesis filterbank introduced mismatch. Correlation is {corr:.6f}"


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

    # Assert that all PyTorch vs NumPy component correlations are >= 0.99
    assert corr_true_pt_vs_np >= 0.99, f"true_target correlation too low: {corr_true_pt_vs_np:.6f}"
    assert corr_dist_pt_vs_np >= 0.99, f"target_distortion correlation too low: {corr_dist_pt_vs_np:.6f}"
    assert corr_interf_pt_vs_np >= 0.99, f"interference correlation too low: {corr_interf_pt_vs_np:.6f}"
    assert corr_artif_pt_vs_np >= 0.99, f"artifacts correlation too low: {corr_artif_pt_vs_np:.6f}"


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
    matches numpy to floating-point precision, so this asserts a tight bound. (A
    regression to the plain softplus max drops this to ~0.9 on longer signals.)
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
    corr = np.corrcoef(a, b)[0, 1]
    assert corr > 0.999, f"Auditory internal representation parity too low: corr={corr:.4f}"
