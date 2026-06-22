"""
PEASS Test Suite - Pipeline Integration Tests
File path: tests/integration/test_pipeline.py
"""

import numpy as np
import pytest

from peass import DecompositionConfiguration
from peass import calculate_auditory_quality_features
from peass import calculate_bss_eval_energy_ratios
from peass import decompose_distortion_components
from peass import predict_perceptual_evaluation_scores
from tests.conftest import to_backend_format
from tests.conftest import to_numpy_format


@pytest.mark.integration
def test_end_to_end_stereo_separation(peass_backend):
    """
    Integration test using multi-channel (Stereo) inputs.
    Verifies that the entire PEASS pipeline (decomposition, metrics, and prediction)
    runs with 2-channel stereo arrays, handling the collapsed worst-channel scores.
    """
    backend_type, device = peass_backend
    sampling_frequency = 16000.0
    duration_seconds = 1.0
    num_samples = int(duration_seconds * sampling_frequency)
    time_steps = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)

    # 1. Create stereo clean target (different frequencies for Left & Right channels)
    target_left = np.sin(2.0 * np.pi * 300.0 * time_steps)
    target_right = np.sin(2.0 * np.pi * 400.0 * time_steps)
    target = np.stack([target_left, target_right], axis=1)

    # 2. Create stereo interference
    interf_left = np.sin(2.0 * np.pi * 1000.0 * time_steps)
    interf_right = np.sin(2.0 * np.pi * 1200.0 * time_steps)
    interferer = np.stack([interf_left, interf_right], axis=1)

    # 3. Create estimate with slight leakage and artifact noise
    estimate = target + 0.1 * interferer + 0.02 * np.random.randn(num_samples, 2)

    # 2. Format inputs to active backend
    target_in = to_backend_format(target, backend_type, device)
    interferer_in = to_backend_format(interferer, backend_type, device)
    estimate_in = to_backend_format(estimate, backend_type, device)

    # 3. Run prediction
    results = predict_perceptual_evaluation_scores(
        original_files=[target_in, interferer_in],
        estimate_file=estimate_in,
        sampling_frequency_hz=sampling_frequency,
        return_decomposition=True
    )

    # Verify scores are generated and stay bounded
    assert 0.0 <= float(to_numpy_format(results.overall_perceptual_score)) <= 100.0
    assert 0.0 <= float(to_numpy_format(results.target_perceptual_score)) <= 100.0
    assert 0.0 <= float(to_numpy_format(results.interference_perceptual_score)) <= 100.0
    assert 0.0 <= float(to_numpy_format(results.artifact_perceptual_score)) <= 100.0

    # Ensure returned decomposition arrays are present and have stereo dimensions
    decomp_waveforms = results.decomposition_waveforms
    assert decomp_waveforms is not None
    assert to_numpy_format(decomp_waveforms.true_target).shape == (num_samples, 2)
    assert to_numpy_format(decomp_waveforms.target_distortion).shape == (num_samples, 2)
    assert to_numpy_format(decomp_waveforms.interference).shape == (num_samples, 2)
    assert to_numpy_format(decomp_waveforms.artifacts).shape == (num_samples, 2)


@pytest.mark.integration
def test_silent_reference_robustness():
    """
    Tests edge-case where one of the interfering sources is completely silent.
    Verifies that the decomposition and scoring models do not crash or trigger division-by-zero.
    """
    sampling_frequency = 16000.0
    duration_seconds = 0.5
    num_samples = int(duration_seconds * sampling_frequency)
    time_steps = np.linspace(0, duration_seconds, num_samples)

    target = np.sin(2.0 * np.pi * 440.0 * time_steps)[:, np.newaxis]
    silent_noise = np.zeros_like(target)  # Completely silent interferer
    estimate = target + 0.01 * np.random.randn(num_samples, 1)

    # Execute full scoring run
    results = predict_perceptual_evaluation_scores(
        original_files=[target, silent_noise],
        estimate_file=estimate,
        sampling_frequency_hz=sampling_frequency
    )

    assert 0.0 <= results.overall_perceptual_score <= 100.0


@pytest.mark.integration
def test_sequential_component_execution_pipeline():
    """
    Integration test directly chaining manual outputs of each component:
    Decomposition -> Metrics -> Neural Net Scoring.
    """
    sampling_frequency = 16000.0
    duration_seconds = 1.0
    num_samples = int(duration_seconds * sampling_frequency)
    time_steps = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)

    target = np.sin(2.0 * np.pi * 350.0 * time_steps)[:, np.newaxis]
    noise = np.sin(2.0 * np.pi * 950.0 * time_steps)[:, np.newaxis]
    estimate = target + 0.05 * noise

    # Step 1: Decomposition
    decomposition_result = decompose_distortion_components(
        source_files=[target, noise],
        estimate_file=estimate,
        sampling_frequency_hz=sampling_frequency
    )
    waveforms = decomposition_result.waveforms

    # Step 2: Energy ratios
    ISR, SIR, SAR, SDR = calculate_bss_eval_energy_ratios(
        waveforms.true_target,
        waveforms.target_distortion,
        waveforms.interference,
        waveforms.artifacts
    )
    assert ISR > 0
    assert SIR > 0

    # Step 3: Auditory Quality Features
    decomposition_tuple = (
        waveforms.true_target,
        waveforms.target_distortion,
        waveforms.interference,
        waveforms.artifacts
    )
    q_target, q_interf, q_artif, q_global = calculate_auditory_quality_features(
        decomposition_tuple, sampling_frequency
    )
    assert -1.0 <= q_target <= 1.0
    assert -1.0 <= q_interf <= 1.0
    assert -1.0 <= q_artif <= 1.0
    assert -1.0 <= q_global <= 1.0


@pytest.mark.integration
def test_predictor_pipeline_on_database_assets(database_audio_pair):
    """Tests the entire neural-network scoring pipeline on real files."""
    target, interferer, fs, _ = database_audio_pair

    # Simulate a realistic separation estimate (Target + low-level leakage + small noise)
    estimate = target + 0.05 * interferer + 0.01 * np.random.randn(*target.shape)

    # Execute full scoring pipeline
    results = predict_perceptual_evaluation_scores(
        original_files=[target, interferer],
        estimate_file=estimate,
        sampling_frequency_hz=fs
    )

    # Assert physical output bounds are maintained
    assert 0.0 <= results.overall_perceptual_score <= 100.0
    assert 0.0 <= results.target_perceptual_score <= 100.0
    assert 0.0 <= results.interference_perceptual_score <= 100.0
    assert 0.0 <= results.artifact_perceptual_score <= 100.0


@pytest.mark.integration
def test_decomposition_correctness_on_database_assets(database_audio_pair):
    """Tests the physical least-squares decomposition on real files."""
    target, interferer, fs, _ = database_audio_pair
    estimate = target + 0.1 * interferer

    config = DecompositionConfiguration()
    result = decompose_distortion_components(
        source_files=[target, interferer],
        estimate_file=estimate,
        configuration=config,
        sampling_frequency_hz=fs
    )

    # Assert that the extracted physical interference correlates with the actual source
    corr_interf = np.corrcoef(result.waveforms.interference.ravel(), interferer.ravel())[0, 1]
    assert corr_interf > 0.85
