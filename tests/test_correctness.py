"""
PEASS Test Suite - Advanced Physical and Psychoacoustic Correctness Tests
File path: tests/test_correctness.py
"""

import pathlib
from typing import Tuple

import numpy as np
import pytest
import scipy.signal as signal
import soundfile as sf

from peass.auditory_model import simulate_auditory_nerve_adaptation
from peass.auditory_model import simulate_inner_haircell_transduction
from peass.decomposition import DecompositionConfiguration
from peass.decomposition import decompose_distortion_components
from peass.decomposition import run_auditory_analysis_filterbank
from peass.decomposition import run_auditory_synthesis_filterbank
from peass.gammatone import GammatoneAnalyzer
from peass.gammatone import GammatoneSynthesizer
from peass.metrics import (
    calculate_auditory_similarity_metric,
)
from peass.predictor import predict_perceptual_evaluation_scores

# -----------------------------------------------------------------------------
# ABSOLUTE FILE PATH RESOLUTION
# -----------------------------------------------------------------------------
TESTS_DIR = pathlib.Path(__file__).parent.resolve()
REPO_ROOT = TESTS_DIR.parent
PEASS_DATA_DIR = REPO_ROOT / "references" / "peass_master_22c7fc4e" / "data"

TARGET_WAV_PATH = PEASS_DATA_DIR / "exp01_target.wav"
INTERFERER_WAV_PATH = PEASS_DATA_DIR / "exp01_InterfSrc1.wav"


# -----------------------------------------------------------------------------
# PYTEST FIXTURES (PARAMETERIZED WAVEFORMS & SEEDED NOISE)
# -----------------------------------------------------------------------------
@pytest.fixture(params=["sine", "square", "triangle", "chirp", "noise"])
def synthetic_signal(request) -> Tuple[np.ndarray, float]:
    """
    Generates parameterized waveforms to verify mathematical robustness across
    periodic, discontinuous, and stochastic signal profiles.
    """
    signal_type = request.param
    sampling_frequency = 16000.0
    duration = 1.5
    num_samples = int(duration * sampling_frequency)
    time_steps = np.linspace(0.0, duration, num_samples, endpoint=False)

    rng = np.random.default_rng(seed=42)

    if signal_type == "sine":
        data = np.sin(2.0 * np.pi * 440.0 * time_steps)
    elif signal_type == "square":
        data = signal.square(2.0 * np.pi * 200.0 * time_steps)
    elif signal_type == "triangle":
        data = signal.sawtooth(2.0 * np.pi * 200.0 * time_steps, width=0.5)
    elif signal_type == "chirp":
        data = signal.chirp(time_steps, f0=50.0, t1=duration, f1=8000.0, method="linear")
    elif signal_type == "noise":
        data = rng.normal(0.0, 0.2, num_samples)
    else:
        raise ValueError("Invalid signal type requested.")

    # Normalize to prevent digital clipping
    data = data / (np.max(np.abs(data)) + 1e-9) * 0.9
    return data[:, np.newaxis], sampling_frequency


# -----------------------------------------------------------------------------
# TEST SUITE 1: FILTERBANK & DECOMPOSITION SUBSPACES
# -----------------------------------------------------------------------------
def test_gammatone_reconstruction_parameterized(synthetic_signal):
    """
    Verifies that the Gammatone analysis-synthesis filterbank correctly
    reconstructs signals across diverse parameterized waveforms.
    """
    waveform, fs = synthetic_signal
    sig_1d = waveform.ravel()

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=fs,
        lower_cutoff_frequency_hz=80.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=6000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    subbands = analyzer.process(sig_1d)
    desired_delay_sec = 0.004
    synthesizer = GammatoneSynthesizer(analyzer, desired_delay_seconds=desired_delay_sec)
    reconstructed = synthesizer.process(subbands)

    delay_samples = int(round(desired_delay_sec * fs))

    # Slice to discard transient filter-delay periods
    original_sliced = sig_1d[delay_samples:-delay_samples]
    reconstructed_sliced = reconstructed[2 * delay_samples:]

    min_len = min(len(original_sliced), len(reconstructed_sliced))
    original_sliced = original_sliced[:min_len]
    reconstructed_sliced = reconstructed_sliced[:min_len]

    # Verify high phase and shape matching through cross-correlation
    corr = np.corrcoef(original_sliced, reconstructed_sliced)[0, 1]
    assert corr > 0.85


def test_decomposition_gain_invariance_parameterized(synthetic_signal):
    """
    Verifies that a constant scaling (gain modification) of the target
    is mapped entirely to Target Distortion, leaving Interference/Artifacts at zero.
    """
    waveform, fs = synthetic_signal
    target = waveform
    silent_interferer = np.zeros_like(target)

    # 30% reduction in level
    estimate = 0.7 * target

    config = DecompositionConfiguration()
    result = decompose_distortion_components(
        source_files=[target, silent_interferer],
        estimate_file=estimate,
        configuration=config,
        sampling_frequency_hz=fs
    )
    waveforms = result.waveforms

    # Compare the synthesized target distortion against the synthesized true target.
    # Both have passed through the identical filterbank, ensuring exact mathematical parity.
    np.testing.assert_allclose(waveforms.target_distortion, -0.3 * waveforms.true_target, atol=1e-4, rtol=1e-4)

    # Interferences and Artifacts remain mathematically negligible
    assert np.max(np.abs(waveforms.interference)) < 1e-4
    assert np.max(np.abs(waveforms.artifacts)) < 1e-4


def test_decomposition_in_bounds_delay_parameterized(synthetic_signal):
    """
    Verifies that temporal delays within the window boundaries are absorbed
    exclusively as Target Distortion.
    """
    waveform, fs = synthetic_signal
    target = waveform
    silent_interferer = np.zeros_like(target)

    # Small delay (3 samples, well within the 40ms solver limit)
    shift = 3
    estimate = np.roll(target, shift, axis=0)
    estimate[:shift, :] = 0.0

    config = DecompositionConfiguration()
    result = decompose_distortion_components(
        source_files=[target, silent_interferer],
        estimate_file=estimate,
        configuration=config,
        sampling_frequency_hz=fs
    )
    waveforms = result.waveforms

    # Shift is in-bounds. Subband least-squares modeling allows small numerical leakage
    # on transient edges, which is physically correct.
    assert np.max(np.abs(waveforms.interference)) < 5e-2
    assert np.max(np.abs(waveforms.artifacts)) < 5e-2


def test_correctness_on_real_audio_db():
    """
    Verifies subspace decomposition on actual speech files extracted from
    the reference PEASS subjective evaluation database directory.
    """
    assert TARGET_WAV_PATH.is_file(), f"Database reference file missing at {TARGET_WAV_PATH}"
    assert INTERFERER_WAV_PATH.is_file(), f"Database reference file missing at {INTERFERER_WAV_PATH}"

    target, fs1 = sf.read(TARGET_WAV_PATH)
    interferer, fs2 = sf.read(INTERFERER_WAV_PATH)
    assert fs1 == fs2

    if target.ndim == 1:
        target = target[:, np.newaxis]
    if interferer.ndim == 1:
        interferer = interferer[:, np.newaxis]

    # Create estimated mixture with known leakage
    estimate = target + 0.1 * interferer

    config = DecompositionConfiguration()
    result = decompose_distortion_components(
        source_files=[target, interferer],
        estimate_file=estimate,
        configuration=config,
        sampling_frequency_hz=fs1
    )
    waveforms = result.waveforms

    # Process interferer through equivalent filterbank to verify exact mathematical reconstruction parity
    subbands_int, analyzer, mod_mat = run_auditory_analysis_filterbank(interferer[:, 0], fs1)
    interferer_synth, _ = run_auditory_synthesis_filterbank(subbands_int, analyzer)

    if interferer_synth.ndim == 1:
        interferer_synth = interferer_synth[:, np.newaxis]

    # Handle slight truncation or OLA discrepancies by slicing both to the minimum length
    min_len = min(len(waveforms.interference), len(interferer_synth))
    waveforms_interf_sliced = waveforms.interference[:min_len]
    interferer_synth_sliced = interferer_synth[:min_len]

    # 1. Verification of Target matching (on aligned slices)
    np.testing.assert_allclose(waveforms_interf_sliced, 0.1 * interferer_synth_sliced, atol=1e-3, rtol=1e-3)

    # 2. Verification of Interference matching target interferer correlation
    corr_interf = np.corrcoef(waveforms.interference.ravel(), interferer.ravel())[0, 1]
    assert corr_interf > 0.85


# -----------------------------------------------------------------------------
# TEST SUITE 2: PHYSIOLOGICAL ERGONOMICS & AUDITORY MODEL
# -----------------------------------------------------------------------------
def test_haircell_frequency_selectivity():
    """
    Verifies that the Inner Hair Cell model attenuates high frequencies
    consistent with its 1 kHz lowpass membrane shear limit.
    """
    fs = 16000.0
    time_steps = np.linspace(0.0, 0.5, int(0.5 * fs), endpoint=False)

    # 100 Hz (Passband) vs 8000 Hz (Severe attenuation)
    low_sine = np.sin(2.0 * np.pi * 100.0 * time_steps)
    high_sine = np.sin(2.0 * np.pi * 8000.0 * time_steps)

    low_ihc = simulate_inner_haircell_transduction(low_sine[np.newaxis, :], fs)
    high_ihc = simulate_inner_haircell_transduction(high_sine[np.newaxis, :], fs)

    # Half-wave rectification constraint
    assert np.all(low_ihc >= -1e-15)
    assert np.all(high_ihc >= -1e-15)

    rms_low = np.sqrt(np.mean(low_ihc ** 2))
    rms_high = np.sqrt(np.mean(high_ihc ** 2))

    # Expect strong attenuation at 8 kHz
    attenuation_db = 20.0 * np.log10(rms_low / (rms_high + 1e-15))
    assert attenuation_db > 15.0


def test_auditory_nerve_adaptation_dynamics():
    """
    Verifies that the non-linear adaptation loops correctly model onset-overshoot
    and slow synaptic masking decay curves.
    """
    fs = 16000.0
    num_samples = int(0.6 * fs)

    pulse = np.zeros(num_samples)
    pulse[:int(0.2 * fs)] = 1.0  # On for 200ms

    adapted = simulate_auditory_nerve_adaptation(pulse[np.newaxis, :], fs)
    adapted_1d = adapted[0, :]

    # 1. Onset overshoot: First 20ms amplitude must be larger than standard middle pulse
    onset_peak = np.max(adapted_1d[:int(0.02 * fs)])
    steady_state = np.mean(adapted_1d[int(0.1 * fs):int(0.18 * fs)])
    assert onset_peak > steady_state

    # 2. Slow decay: Immediately after the offset, the signal undershoots (adaptation suppression)
    # and slowly recovers back up toward the resting threshold.
    offset_sample = int(0.2 * fs)
    immediate_after = adapted_1d[offset_sample + int(0.005 * fs)]
    later_after = adapted_1d[offset_sample + int(0.1 * fs)]

    assert immediate_after < later_after
    assert np.all(adapted_1d >= -300.0)


def test_perceptual_assimilation_behavior():
    """
    Verifies that the assimilation masking rules correctly pull attenuated signals
    closer to reference thresholds, increasing perceptual similarity.
    """
    fs = 100.0
    num_bands, num_samples, num_modulations = 4, 200, 1

    rng = np.random.default_rng(seed=42)
    ref_rep = np.abs(rng.normal(1.0, 0.2, (num_bands, num_samples, num_modulations)))

    # Introduce non-uniform noise to test representation
    test_rep = np.abs(ref_rep + rng.normal(0.0, 0.1, ref_rep.shape))
    test_rep_scaled = 0.5 * test_rep

    # Standard unassimilated Pearson correlation
    lref = ref_rep.ravel() - np.mean(ref_rep)
    ltest = test_rep_scaled.ravel() - np.mean(test_rep_scaled)
    standard_corr = np.sum(lref * ltest) / np.sqrt(np.sum(lref ** 2) * np.sum(ltest ** 2))

    # Model assimilated metric (assimilation pulls the noisy signal closer to ref, masking the noise)
    perceptual_metric = calculate_auditory_similarity_metric(ref_rep, test_rep_scaled, fs)

    # Perceptual metric must be higher than standard correlation due to threshold masking
    assert perceptual_metric > standard_corr

    # Perfect identity constraint
    identity_metric = calculate_auditory_similarity_metric(ref_rep, ref_rep.copy(), fs)
    np.testing.assert_allclose(identity_metric, 1.0, atol=1e-5)


# -----------------------------------------------------------------------------
# TEST SUITE 3: NUMERICAL BOUNDS & STRESS STABILITY
# -----------------------------------------------------------------------------
def test_filterbank_dc_offset_rejection():
    """
    Verifies that the auditory filterbank correctly blocks constant DC bias offsets.
    """
    fs = 16000.0
    time_steps = np.linspace(0.0, 0.5, int(0.5 * fs), endpoint=False)

    dc_bias = 2.5
    signal_with_dc = np.sin(2.0 * np.pi * 500.0 * time_steps) + dc_bias

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=fs,
        lower_cutoff_frequency_hz=235.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=6000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    subbands = np.real(analyzer.process(signal_with_dc))

    # All subband outputs must have DC offset rejected (attenuation > 50 dB)
    for band_idx in range(subbands.shape[0]):
        assert np.abs(np.mean(subbands[band_idx, :])) < 1e-2


def test_stress_and_empty_edge_cases():
    """
    Tests edge cases (complete silence, very short inputs) to verify
    zero-division and dimensionality crash resistance.
    """
    fs = 16000.0
    num_samples = int(1.0 * fs)

    silent_target = np.zeros((num_samples, 1))
    silent_noise = np.zeros((num_samples, 1))
    silent_estimate = np.zeros((num_samples, 1))

    # 1. Silent signals should evaluate cleanly without division-by-zero crashes
    scores = predict_perceptual_evaluation_scores(
        original_files=[silent_target, silent_noise],
        estimate_file=silent_estimate,
        sampling_frequency_hz=fs
    )

    assert 0.0 <= scores.overall_perceptual_score <= 100.0
    assert 0.0 <= scores.target_perceptual_score <= 100.0

    # 2. Short signals should raise appropriate value errors rather than unhandled array exceptions
    short_target = np.random.randn(10, 1)
    short_noise = np.zeros_like(short_target)
    short_estimate = short_target.copy()

    with pytest.raises(ValueError):
        decompose_distortion_components(
            source_files=[short_target, short_noise],
            estimate_file=short_estimate,
            sampling_frequency_hz=fs
        )


def test_regression_against_matlab_references():
    """
    Validates Python-decomposed waveforms directly against the original
    reference WAV files generated by the official MATLAB PEASS v2.0.1 toolbox.
    """
    import numpy as np
    import pathlib
    import soundfile as sf
    from peass.decomposition import decompose_distortion_components, DecompositionConfiguration

    # Locate reference paths relative to test file directory
    ref_dir = pathlib.Path(__file__).parent.parent / "references" / "peass_master_22c7fc4e" / "v2.0.1" / "example"

    target_src_path = ref_dir / "targetSrc.wav"
    interf1_path = ref_dir / "interfSrc1.wav"
    interf2_path = ref_dir / "interfSrc2.wav"
    estimate_path = ref_dir / "targetEstimate.wav"

    # Reference outputs generated by MATLAB
    ref_true_path = ref_dir / "targetEstimate_true.wav"
    ref_target_path = ref_dir / "targetEstimate_eTarget.wav"
    ref_interf_path = ref_dir / "targetEstimate_eInterf.wav"
    ref_artif_path = ref_dir / "targetEstimate_eArtif.wav"

    # Verify MATLAB references exist
    assert target_src_path.is_file(), "MATLAB reference files not found."

    # Load inputs
    target_src, fs = sf.read(target_src_path)
    interf1, _ = sf.read(interf1_path)
    interf2, _ = sf.read(interf2_path)
    estimate, _ = sf.read(estimate_path)

    # Decompose using the Python port with identical window configurations
    config = DecompositionConfiguration(shade_in_milliseconds=10.0, shade_out_milliseconds=10.0)
    result = decompose_distortion_components(
        source_files=[target_src, interf1, interf2],
        estimate_file=estimate,
        configuration=config,
        sampling_frequency_hz=fs
    )
    waveforms = result.waveforms

    # Load MATLAB gold standard waveforms
    gold_true, _ = sf.read(ref_true_path)
    gold_target, _ = sf.read(ref_target_path)
    gold_interf, _ = sf.read(ref_interf_path)
    gold_artif, _ = sf.read(ref_artif_path)

    # Handle tiny sampling-rate alignment differences (decimation alignment)
    min_len = min(len(waveforms.true_target), len(gold_true))

    # Verify high cross-correlation (> 0.95) with the MATLAB-generated gold standard
    components_to_verify = [
        (waveforms.true_target[:min_len], gold_true[:min_len], "true_target"),
        (waveforms.target_distortion[:min_len], gold_target[:min_len], "target_distortion"),
        (waveforms.interference[:min_len], gold_interf[:min_len], "interference"),
        (waveforms.artifacts[:min_len], gold_artif[:min_len], "artifacts")
    ]

    for py_val, mat_val, name in components_to_verify:
        for ch in range(py_val.shape[1]):
            # Quiet/silent segments can have unstable correlation; verify by variance in those cases
            if np.std(mat_val[:, ch]) < 1e-4:
                assert np.std(py_val[:, ch]) < 1e-4, f"Energy mismatch in {name} channel {ch}"
            else:
                corr = np.corrcoef(py_val[:, ch], mat_val[:, ch])[0, 1]
                assert corr > 0.95, f"Correlation too low for {name} channel {ch}: {corr:.4f}"


def test_gammatone_fallback_vs_jit_equivalence():
    """
    Verifies that disabling Numba JIT acceleration forces the filterbank
    to run on the fallback path and produces identical outputs down to float precision.
    """
    import peass.gammatone as gammatone
    from peass.gammatone import GammatoneAnalyzer

    # Generate a random signal
    rng = np.random.default_rng(seed=123)
    signal_input = rng.normal(0.0, 0.5, 1000)
    fs = 16000.0

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=fs,
        lower_cutoff_frequency_hz=80.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=4000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    # 1. Run with JIT active
    original_has_numba = gammatone._HAS_NUMBA
    try:
        gammatone._HAS_NUMBA = True
        analyzer.clear_state()
        jit_output = analyzer.process(signal_input)

        # 2. Force fallback path by mocking _HAS_NUMBA as False
        gammatone._HAS_NUMBA = False
        analyzer.clear_state()
        fallback_output = analyzer.process(signal_input)

        # Assert mathematical parity down to machine precision
        np.testing.assert_allclose(jit_output, fallback_output, rtol=1e-12, atol=1e-12)
        
    finally:
        # Restore original setting
        gammatone._HAS_NUMBA = original_has_numba
