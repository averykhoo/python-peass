"""
PEASS Test Suite - Predictor End-to-End Regressor Tests
File path: tests/test_predictor.py
"""

import pathlib
from typing import Tuple

import numpy as np

from peass.predictor import predict_peass_scores


def test_predictor_score_range_constraints(
        synthetic_audio_data: Tuple[np.ndarray, np.ndarray, np.ndarray, float]
):
    """
    Performs an end-to-end in-memory execution run and verifies that all
    predicted perceptual scores fall within valid bounds [1].
    """
    target, interferer, estimate, fs = synthetic_audio_data

    # Execute predictor
    results = predict_peass_scores(
        original_files=[target, interferer],
        estimate_file=estimate,
        sampling_frequency=fs,
        return_decomposition=True
    )

    # 1. Assert traditional dB metrics computed successfully
    for metric_key in ["SDR", "ISR", "SIR", "SAR"]:
        assert metric_key in results
        assert isinstance(results[metric_key], float)

    # 2. Assert predicted perceptual scores are bounded within [0, 100]
    for score_key in ["OPS", "TPS", "IPS", "APS"]:
        assert score_key in results
        assert isinstance(results[score_key], float)
        assert 0.0 <= results[score_key] <= 100.0

    # 3. Verify returned decomposition arrays
    assert "decomposition_arrays" in results
    assert "true_target" in results["decomposition_arrays"]
    assert "target_distortion" in results["decomposition_arrays"]
    assert "interference" in results["decomposition_arrays"]
    assert "artifacts" in results["decomposition_arrays"]


def test_predictor_file_based_execution(
        audio_files_fixture: Tuple[pathlib.Path, pathlib.Path, pathlib.Path]
):
    """
    Verifies that the end-to-end predictor runs successfully in file-based mode.
    """
    target_path, interferer_path, estimate_path = audio_files_fixture

    results = predict_peass_scores(
        original_files=[str(target_path), str(interferer_path)],
        estimate_file=str(estimate_path),
        options={'destDir': str(target_path.parent)},
        return_decomposition=True
    )

    # Assert that score outputs are valid
    assert 0.0 <= results["OPS"] <= 100.0

    # Assert that file-path mappings are present
    assert "decomposition_files" in results
    assert pathlib.Path(results["decomposition_files"]["true_target"]).is_file()


def test_predictor_pristine_audio_conditions():
    """
    Verifies that evaluating a pristine, near-perfect separation yields
    predictably high perceptual scores.
    """
    sampling_frequency = 16000.0
    duration_seconds = 1.5
    num_samples = int(duration_seconds * sampling_frequency)
    time_steps = np.linspace(0, duration_seconds, num_samples)

    # Target is clean sine wave
    target = np.sin(2.0 * np.pi * 300.0 * time_steps)[:, np.newaxis]
    noise = 0.0 * np.sin(2.0 * np.pi * 1000.0 * time_steps)[:, np.newaxis]

    # Perfect Estimate (identical to target)
    estimate = target.copy()

    results = predict_peass_scores(
        original_files=[target, noise],
        estimate_file=estimate,
        sampling_frequency=sampling_frequency
    )

    # A perfect separation should result in scores close to 100
    assert results["OPS"] > 90.0
    assert results["TPS"] > 90.0
    assert results["IPS"] > 90.0
    assert results["APS"] > 90.0
