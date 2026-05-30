"""
PEASS Test Suite - Subband Least-Squares Decomposition Tests
File path: tests/test_decomposition.py
"""

import pathlib
from typing import Tuple

import numpy as np
import pytest
import scipy.signal as signal

from peass.decomposition import extract_distortion_components


def apply_shading(sig: np.ndarray, fs: float, shade_in: float = 10.0, shade_out: float = 10.0) -> np.ndarray:
    """Helper window shading mimicking extract_distortion_components boundaries."""
    sig_shaded = sig.copy()
    if shade_in > 0:
        win_len = 2 * int(round(shade_in / 1000.0 * fs + 1))
        wShadeIn = signal.windows.hann(win_len, sym=False)[:win_len // 2]
        for c in range(sig_shaded.shape[1]):
            sig_shaded[:len(wShadeIn), c] *= wShadeIn
    if shade_out > 0:
        win_len = 2 * int(round(shade_out / 1000.0 * fs + 1))
        wShadeOut = signal.windows.hann(win_len, sym=False)[:win_len // 2]
        wShadeOut = np.flip(wShadeOut)
        for c in range(sig_shaded.shape[1]):
            sig_shaded[-len(wShadeOut):, c] *= wShadeOut
    return sig_shaded


def test_decomposition_algebraic_reconstruction(
        synthetic_audio_data: Tuple[np.ndarray, np.ndarray, np.ndarray, float]
):
    """
    Mathematically verifies that the decomposition satisfies the fundamental
    separation error conservation identity:

        estimate_reconstructed = target_distortion + interference + artifacts + true_target

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

    # Apply 10ms Hann shading window matching internal processing bounds
    estimate_shaded = apply_shading(estimate, fs, 10.0, 10.0)

    # Reconstruct estimate_shaded using the same analysis-synthesis filterbank to get the expected sum
    from peass.decomposition import my_analysis_filter_bank, my_synthesis_filter_bank

    NChan = estimate_shaded.shape[1]
    est_reconstructed = np.zeros_like(estimate_shaded)
    Mmod = None

    def fit_to_length(sig, target_len):
        if len(sig) >= target_len:
            return sig[:target_len]
        return np.pad(sig, (0, target_len - len(sig)), mode='constant')

    for nChan in range(NChan):
        subbands, analyzer, Mmod = my_analysis_filter_bank(estimate_shaded[:, nChan], fs)
        synth_signal, _ = my_synthesis_filter_bank(subbands, analyzer)
        est_reconstructed[:, nChan] = fit_to_length(synth_signal, len(estimate_shaded))

    # Sum of the decomposed sub-components
    summed_sub_components = true_target + target_distortion + interference + artifacts

    # Mathematically assert the reconstruction identity holds precisely
    np.testing.assert_allclose(est_reconstructed, summed_sub_components, atol=1e-7, rtol=1e-7)


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


def test_decomposition_input_validation():
    """
    Verifies that extract_distortion_components raises ValueError when input
    dimensions or sampling frequencies are mismatched.
    """
    fs = 16000.0
    sig_len = 1000
    target = np.random.randn(sig_len, 1)
    interferer = np.random.randn(sig_len, 1)

    # Mismatched length estimate
    estimate_mismatched = np.random.randn(sig_len + 10, 1)

    with pytest.raises(ValueError, match="dimensions|size"):
        extract_distortion_components(
            src_files=[target, interferer],
            est_file=estimate_mismatched,
            sampling_frequency=fs
        )

    # Missing sampling rate in in-memory mode
    with pytest.raises(ValueError, match="requires explicit sampling rate"):
        extract_distortion_components(
            src_files=[target, interferer],
            est_file=target,
            sampling_frequency=None
        )
