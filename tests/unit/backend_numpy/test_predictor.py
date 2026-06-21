"""
PEASS Test Suite - Predictor End-to-End Regressor Unit Tests
File path: tests/unit/test_predictor.py
"""

import pathlib

import numpy as np
import pytest

from peass.backend_numpy.decomposition import DecompositionConfiguration
from peass.backend_numpy.decomposition import decompose_distortion_components
from peass.backend_numpy.predictor import predict_perceptual_evaluation_scores


@pytest.mark.unit
@pytest.mark.parametrize("estimate_scale", [1.0, 0.5, 0.1, 0.01])
def test_predictor_score_range_constraints(synthetic_audio_data, estimate_scale):
    """Verifies score scaling holds across different degradation ranges."""
    target, interferer, estimate, fs = synthetic_audio_data

    # Scale estimate to simulate varying degrees of signal degradation
    scaled_estimate = estimate * estimate_scale

    results = predict_perceptual_evaluation_scores(
        original_files=[target, interferer],
        estimate_file=scaled_estimate,
        sampling_frequency_hz=fs,
        return_decomposition=True
    )

    assert isinstance(results.source_to_distortion_ratio, float)
    assert isinstance(results.overall_perceptual_score, float)
    assert 0.0 <= results.overall_perceptual_score <= 100.0
    assert 0.0 <= results.target_perceptual_score <= 100.0
    assert 0.0 <= results.interference_perceptual_score <= 100.0
    assert 0.0 <= results.artifact_perceptual_score <= 100.0


@pytest.mark.unit
def test_predictor_pristine_audio_conditions():
    """Verifies perfect scores are assigned to pristine identical tracks."""
    sampling_frequency_hz = 16000.0
    duration_seconds = 1.5
    num_samples = int(duration_seconds * sampling_frequency_hz)
    time_steps = np.linspace(0, duration_seconds, num_samples)

    target = np.sin(2.0 * np.pi * 300.0 * time_steps)[:, np.newaxis]
    noise = 0.0 * np.sin(2.0 * np.pi * 1000.0 * time_steps)[:, np.newaxis]

    estimate = target.copy()

    results = predict_perceptual_evaluation_scores(
        original_files=[target, noise],
        estimate_file=estimate,
        sampling_frequency_hz=sampling_frequency_hz
    )

    assert results.overall_perceptual_score > 90.0
    assert results.target_perceptual_score > 90.0
    assert results.interference_perceptual_score > 90.0


@pytest.mark.unit
def test_predictor_file_based_execution(audio_files_fixture):
    """
    Verifies that the end-to-end predictor runs successfully in file-based mode.
    """
    target_path, interferer_path, estimate_path = audio_files_fixture

    configuration = DecompositionConfiguration(
        destination_directory=str(target_path.parent)
    )

    results = predict_perceptual_evaluation_scores(
        original_files=[str(target_path), str(interferer_path)],
        estimate_file=str(estimate_path),
        configuration=configuration,
        return_decomposition=True
    )

    # Assert that score outputs are valid
    assert 0.0 <= results.overall_perceptual_score <= 100.0

    # Assert that file-path mappings are present
    assert results.decomposition_files is not None
    assert pathlib.Path(results.decomposition_files.true_target).is_file()


@pytest.mark.unit
def test_predictor_with_alternative_options(synthetic_audio_data):
    """
    Tests predictor behavior when invoking custom parameters like use_two_stage_projection.
    """
    target, interferer, estimate, fs = synthetic_audio_data

    configuration = DecompositionConfiguration(
        use_two_stage_projection=True,
        frame_length_seconds=0.4,
        filter_length_seconds=0.03,
        shade_in_milliseconds=5.0,
        shade_out_milliseconds=5.0
    )

    results = predict_perceptual_evaluation_scores(
        original_files=[target, interferer],
        estimate_file=estimate,
        configuration=configuration,
        sampling_frequency_hz=fs
    )

    assert 0.0 <= results.overall_perceptual_score <= 100.0
    assert 0.0 <= results.target_perceptual_score <= 100.0
    assert 0.0 <= results.interference_perceptual_score <= 100.0
    assert 0.0 <= results.artifact_perceptual_score <= 100.0


@pytest.mark.unit
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


@pytest.mark.unit
@pytest.mark.parametrize("gain_scaling", [2.0, 1.0, 0.1, 1e-4])  # Amplified, Standard, Quiet, Near-Silent
def test_predictor_amplitude_robustness(synthetic_audio_data, gain_scaling):
    """
    Verifies that scoring and mathematical mappings do not crash or produce NaN values
    when subjected to high or extremely low signal amplitudes.
    """
    target, interferer, estimate, fs = synthetic_audio_data

    scaled_target = target * gain_scaling
    scaled_interferer = interferer * gain_scaling
    scaled_estimate = estimate * gain_scaling

    results = predict_perceptual_evaluation_scores(
        original_files=[scaled_target, scaled_interferer],
        estimate_file=scaled_estimate,
        sampling_frequency_hz=fs
    )

    # Confirm that scores remain numbers rather than NaN or Inf
    assert not np.isnan(results.overall_perceptual_score)
    assert not np.isinf(results.overall_perceptual_score)
