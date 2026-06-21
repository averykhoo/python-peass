"""
PEASS Test Suite - Subband Least-Squares Decomposition Unit Tests
File path: tests/unit/test_decomposition.py
"""

import pathlib

import numpy as np
import pytest

from peass.backend_numpy.decomposition import DecompositionConfiguration
from peass.backend_numpy.decomposition import decompose_distortion_components
from peass.backend_numpy.decomposition import run_auditory_analysis_filterbank
from peass.backend_numpy.decomposition import run_auditory_synthesis_filterbank


def apply_window_shading_helper(sig: np.ndarray, fs: float, shade_in: float = 10.0,
                                shade_out: float = 10.0) -> np.ndarray:
    sig_shaded = sig.copy()
    num_samples = sig_shaded.shape[0]

    fade_in_samples = int(round(shade_in / 1000.0 * fs)) if shade_in > 0 else 0
    fade_out_samples = int(round(shade_out / 1000.0 * fs)) if shade_out > 0 else 0

    if fade_in_samples > 1 and fade_in_samples <= num_samples:
        time_steps = np.arange(fade_in_samples)
        shade_in_window = 0.5 - 0.5 * np.cos(np.pi * time_steps / (fade_in_samples - 1))
        for chan_idx in range(sig_shaded.shape[1]):
            sig_shaded[:fade_in_samples, chan_idx] *= shade_in_window

    if fade_out_samples > 1 and fade_out_samples <= num_samples:
        time_steps = np.arange(fade_out_samples)
        shade_out_window = 0.5 + 0.5 * np.cos(np.pi * time_steps / (fade_out_samples - 1))
        for chan_idx in range(sig_shaded.shape[1]):
            sig_shaded[-fade_out_samples:, chan_idx] *= shade_out_window

    return sig_shaded


@pytest.mark.unit
def test_decomposition_algebraic_reconstruction(synthetic_audio_data):
    """
    Mathematically verifies that the decomposition satisfies the fundamental
    separation error conservation identity:

        estimate_reconstructed = target_distortion + interference + artifacts + true_target

    This verifies that no signal energy is lost or created during the
    subband least-squares and filterbank synthesis stages [1].
    """
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


@pytest.mark.unit
def test_decomposition_file_generation(audio_files_fixture):
    """
    Verifies that the decomposer successfully writes output WAV files to disk
    in file-based execution mode.
    """
    target_path, interferer_path, estimate_path = audio_files_fixture

    # Define the configurations using the new dataclass structure
    configuration = DecompositionConfiguration(
        destination_directory=str(target_path.parent)
    )

    # Execute using the modern, JIT-friendly entry point
    result = decompose_distortion_components(
        source_files=[str(target_path), str(interferer_path)],
        estimate_file=str(estimate_path),
        configuration=configuration
    )

    # Verify that the generated file paths are present
    assert result.file_paths is not None
    output_files = [
        result.file_paths.true_target,
        result.file_paths.target_distortion,
        result.file_paths.interference,
        result.file_paths.artifacts
    ]

    assert len(output_files) == 4
    for file_path in output_files:
        path_obj = pathlib.Path(file_path)
        assert path_obj.is_file()
        assert path_obj.stat().st_size > 0


@pytest.mark.unit
def test_decomposition_input_validation():
    """Checks bounds validation during parameter instantiation."""
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

    # Missing sampling rate in in-memory mode
    with pytest.raises(ValueError, match="requires explicit sampling rate"):
        decompose_distortion_components(
            source_files=[target, interferer],
            estimate_file=target,
            sampling_frequency_hz=None
        )


@pytest.mark.unit
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


@pytest.mark.unit
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
