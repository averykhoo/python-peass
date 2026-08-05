"""
PEASS Test Suite - Subband Least-Squares Decomposition Unit Tests
File path: tests/unit/test_decomposition.py
"""

import pathlib

import numpy as np
import pytest

from peass.backend_numpy.decomposition import DecompositionConfiguration
from peass.backend_numpy.decomposition import decompose_distortion_components
from peass.backend_numpy.decomposition import matlab_shade_length
from peass.backend_numpy.decomposition import matlab_shade_window
from peass.backend_numpy.decomposition import run_auditory_analysis_filterbank
from peass.backend_numpy.decomposition import run_auditory_synthesis_filterbank

# (fs, shade_ms) pairs exercised by the MATLAB-fidelity shade window tests. The
# 44100/5.0, 44100/25.0 and 22050/10.0 entries land exactly on a .5 sample count,
# which pins MATLAB's round-half-away-from-zero tie rule.
SHADE_WINDOW_CASES = [
    (8000.0, 10.0), (8000.0, 5.0), (8000.0, 25.0),
    (16000.0, 10.0), (16000.0, 5.0), (16000.0, 25.0),
    (22050.0, 10.0),
    (44100.0, 10.0), (44100.0, 5.0), (44100.0, 25.0),
    (48000.0, 10.0), (48000.0, 5.0), (48000.0, 25.0),
    (8000.0, 0.125), (16000.0, 0.2),
]


def matlab_shade_window_reference(fs: float, shade_ms: float) -> np.ndarray:
    """
    Literal transcription of MATLAB `extractDistortionComponents.m` (v2.0.1)::

        wShadeIn = hann(2*round(shadeInMs/1000*fs+1),'periodic');
        wShadeIn = wShadeIn(2:end/2);

    with ``hann(N,'periodic')[k] = 0.5*(1 - cos(2*pi*(k-1)/N))`` for 1-based k=1..N and
    MATLAB's round-half-away-from-zero. Used as an independent oracle for the closed
    form used in production.
    """
    n_total = 2 * int(np.floor(shade_ms / 1000.0 * fs + 1.0 + 0.5))
    k = np.arange(1, n_total + 1)  # 1-based MATLAB index
    full_hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * (k - 1) / n_total))
    # MATLAB w(2:end/2) keeps 1-based k = 2 .. N/2 -> 0-based 1 .. N//2 - 1
    return full_hann[1:n_total // 2]


def apply_window_shading_helper(sig: np.ndarray, fs: float, shade_in: float = 10.0,
                                shade_out: float = 10.0) -> np.ndarray:
    sig_shaded = sig.copy()
    num_samples = sig_shaded.shape[0]

    fade_in_samples = matlab_shade_length(shade_in, fs) if shade_in > 0 else 0
    fade_out_samples = matlab_shade_length(shade_out, fs) if shade_out > 0 else 0

    if 0 < fade_in_samples <= num_samples:
        shade_in_window = matlab_shade_window(fade_in_samples)
        for chan_idx in range(sig_shaded.shape[1]):
            sig_shaded[:fade_in_samples, chan_idx] *= shade_in_window

    if 0 < fade_out_samples <= num_samples:
        shade_out_window = matlab_shade_window(fade_out_samples)[::-1]
        for chan_idx in range(sig_shaded.shape[1]):
            sig_shaded[-fade_out_samples:, chan_idx] *= shade_out_window

    return sig_shaded


@pytest.mark.unit
@pytest.mark.parametrize("fs, shade_ms", SHADE_WINDOW_CASES)
def test_shade_window_matches_matlab_hann_slice(fs, shade_ms):
    """
    The production shade window must reproduce MATLAB's
    `hann(2*round(ms/1000*fs+1),'periodic')` sliced with `(2:end/2)` exactly.
    """
    reference_window = matlab_shade_window_reference(fs, shade_ms)

    fade_samples = matlab_shade_length(shade_ms, fs)
    assert fade_samples == reference_window.size

    produced_window = matlab_shade_window(fade_samples)
    np.testing.assert_allclose(produced_window, reference_window, rtol=0.0, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("fs, shade_ms", SHADE_WINDOW_CASES)
def test_shade_window_is_strict_hann_interior(fs, shade_ms):
    """
    MATLAB drops both the leading zero and the unity midpoint of the Hann rise, so the
    window is strictly inside (0, 1). A plain 0->1 ramp would violate this.
    """
    window = matlab_shade_window(matlab_shade_length(shade_ms, fs))

    assert window.size > 0
    assert np.all(window > 0.0)
    assert np.all(window < 1.0)
    # Strictly increasing rise
    assert np.all(np.diff(window) > 0.0)


@pytest.mark.unit
def test_shade_out_window_is_exact_reverse_of_shade_in():
    """MATLAB derives wShadeOut from wShadeIn via flipud(), so the two must mirror exactly."""
    for fs, shade_ms in SHADE_WINDOW_CASES:
        shade_in_window = matlab_shade_window(matlab_shade_length(shade_ms, fs))
        shade_out_window = matlab_shade_window(matlab_shade_length(shade_ms, fs))[::-1]
        np.testing.assert_array_equal(shade_out_window, shade_in_window[::-1])


@pytest.mark.unit
def test_shade_window_length_uses_matlab_tie_breaking():
    """
    MATLAB rounds halves away from zero; Python's built-in round() rounds half to even.
    At 44.1 kHz a 5 ms shade is exactly 220.5 samples, which discriminates the two.
    """
    assert matlab_shade_length(5.0, 44100.0) == 221
    assert matlab_shade_length(25.0, 44100.0) == 1103
    assert matlab_shade_length(10.0, 22050.0) == 221
    assert matlab_shade_length(10.0, 44100.0) == 441
    # A single-sample window is well defined for the MATLAB formula (the old 0->1 ramp
    # divided by zero here), and evaluates to the Hann midpoint-adjacent value.
    assert matlab_shade_length(0.125, 8000.0) == 1
    np.testing.assert_allclose(matlab_shade_window(1), [0.5], rtol=0.0, atol=1e-15)


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
