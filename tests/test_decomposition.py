"""
PEASS Test Suite - Subband Least-Squares Decomposition Tests
File path: tests/test_decomposition.py
"""

import pathlib
from typing import Tuple

import numpy as np

from peass.decomposition import extract_distortion_components


def test_decomposition_algebraic_reconstruction(
        synthetic_audio_data: Tuple[np.ndarray, np.ndarray, np.ndarray, float]
):
    """
    Mathematically verifies that the decomposition satisfies the fundamental
    separation error conservation identity:

        estimate - true_target = target_distortion + interference + artifacts

    This verifies that no signal energy is lost or created during the
    subband least-squares and filterbank synthesis stages [1].
    """
    target, interferer, estimate, fs = synthetic_audio_data

    # Execute physical decomposition in-memory
    _, decomposed_arrays = extract_distortion_components(
        src_files=[target, interferer],
        est_file=estimate,
        sampling_frequency=fs
    )
    true_target, target_distortion, interference, artifacts = decomposed_arrays

    # Assert shape alignment
    assert true_target.shape == estimate.shape
    assert target_distortion.shape == estimate.shape
    assert interference.shape == estimate.shape
    assert artifacts.shape == estimate.shape

    # Calculate actual total error
    actual_total_error = estimate - target

    # Sum of the decomposed sub-components
    summed_sub_components = target_distortion + interference + artifacts

    # Mathematically assert the reconstruction identity holds
    # Tolerances are set slightly loose to account for subband filterbank transition phases
    np.testing.assert_allclose(actual_total_error, summed_sub_components, atol=1e-3, rtol=1e-3)


def test_decomposition_file_generation(
        audio_files_fixture: Tuple[pathlib.Path, pathlib.Path, pathlib.Path]
):
    """
    Verifies that the decomposer successfully writes output WAV files to disk
    in file-based execution mode [1].
    """
    target_path, interferer_path, estimate_path = audio_files_fixture

    output_files, _ = extract_distortion_components(
        src_files=[str(target_path), str(interferer_path)],
        est_file=str(estimate_path),
        options={'destDir': str(target_path.parent)}
    )

    # Assert that all 4 expected wav files were successfully generated
    assert len(output_files) == 4
    for file_path in output_files:
        path_obj = pathlib.Path(file_path)
        assert path_obj.is_file()
        assert path_obj.stat().st_size > 0
