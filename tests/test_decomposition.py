"""
PEASS Test Suite - Subband Least-Squares Decomposition Tests
"""

import pathlib
from typing import Tuple
import numpy as np
import pytest
import scipy.signal as signal

from peass.decomposition import (
    DecompositionConfiguration,
    decompose_distortion_components,
    run_auditory_analysis_filterbank,
    run_auditory_synthesis_filterbank
)

def apply_window_shading_helper(sig: np.ndarray, fs: float, shade_in: float = 10.0, shade_out: float = 10.0) -> np.ndarray:
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
    target, interferer, estimate, fs = synthetic_audio_data

    configuration = DecompositionConfiguration()
    decomposition_result = decompose_distortion_components(
        source_files=[target, interferer],
        estimate_file=estimate,
        configuration=configuration,
        sampling_frequency_hz=fs
    )
    waveforms = decomposition_result.waveforms

    assert waveforms.true_target.shape == estimate.shape

    estimate_shaded = apply_window_shading_helper(estimate, fs, 10.0, 10.0)
    number_of_channels = estimate_shaded.shape[1]
    est_reconstructed = np.zeros_like(estimate_shaded)

    def fit_to_length(sig, target_len):
        if len(sig) >= target_len:
            return sig[:target_len]
        return np.pad(sig, (0, target_len - len(sig)), mode='constant')

    for channel_idx in range(number_of_channels):
        subbands, analyzer, modulation_matrix = run_auditory_analysis_filterbank(estimate_shaded[:, channel_idx], fs)
        synth_signal, _ = run_auditory_synthesis_filterbank(subbands, analyzer)
        est_reconstructed[:, channel_idx] = fit_to_length(synth_signal, len(estimate_shaded))

    summed_sub_components = (
        waveforms.true_target + waveforms.target_distortion + waveforms.interference + waveforms.artifacts
    )
    np.testing.assert_allclose(est_reconstructed, summed_sub_components, atol=1e-7, rtol=1e-7)

def test_decomposition_input_validation():
    sampling_frequency_hz = 16000.0
    signal_length = 1000
    target = np.random.randn(signal_length, 1)
    interferer = np.random.randn(signal_length, 1)

    estimate_mismatched = np.random.randn(signal_length + 10, 1)
    with pytest.raises(ValueError, match="dimensions|size"):
        decompose_distortion_components(
            source_files=[target, interferer],
            estimate_file=estimate_mismatched,
            sampling_frequency_hz=sampling_frequency_hz
        )