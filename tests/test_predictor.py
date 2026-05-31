"""
PEASS Test Suite - Predictor End-to-End Regressor Tests
"""
import pathlib
from typing import Tuple

import numpy as np
import pytest

from peass.decomposition import DecompositionConfiguration
from peass.predictor import predict_perceptual_evaluation_scores


@pytest.mark.parametrize("estimate_scale", [1.0, 0.5, 0.1, 0.01])
def test_predictor_score_range_constraints(
        synthetic_audio_data: Tuple[np.ndarray, np.ndarray, np.ndarray, float],
        estimate_scale
):
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


def test_predictor_pristine_audio_conditions():
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


def test_predictor_file_based_execution(
        audio_files_fixture: Tuple[pathlib.Path, pathlib.Path, pathlib.Path]
):
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


def test_predictor_with_alternative_options(
        synthetic_audio_data: Tuple[np.ndarray, np.ndarray, np.ndarray, float]
):
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
