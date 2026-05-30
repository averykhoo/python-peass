"""
PEASS Test Suite - Pytest Configuration and Fixtures
File path: tests/conftest.py
"""

import pathlib
import tempfile
from typing import Generator
from typing import Tuple

import numpy as np
import pytest
import scipy.signal as signal
import soundfile as sf


@pytest.fixture(scope="module")
def synthetic_audio_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generates synthetic target, interferer, and estimate signals for DSP testing.

    Target: 440 Hz Sine wave (representing clear speech)
    Interferer: 1200 Hz Sine wave (representing background acoustic noise)
    Estimate: Distorted version with lowpass-filtered target (spatial/filtering distortion),
              remaining interference leakage, and additive white noise (artifacts).
    """
    sampling_frequency = 16000.0  # 16 kHz
    duration_seconds = 2.0
    num_samples = int(duration_seconds * sampling_frequency)
    time_steps = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)

    # 1. Generate clean target (440 Hz)
    target_waveform = np.sin(2.0 * np.pi * 440.0 * time_steps)[:, np.newaxis]

    # 2. Generate interferer (1200 Hz)
    interferer_waveform = np.sin(2.0 * np.pi * 1200.0 * time_steps)[:, np.newaxis]

    # 3. Simulate Target/Spatial Distortion by lowpassing the target
    # Recreates a typical acoustic channel filtering alteration
    butterworth_b, butterworth_a = signal.butter(4, 800.0 / (sampling_frequency / 2.0), btype='low')
    target_distorted = signal.lfilter(butterworth_b, butterworth_a, target_waveform, axis=0)

    # 4. Generate Artifacts (low level white Gaussian noise)
    artifact_noise = 0.01 * np.random.randn(num_samples, 1)

    # 5. Compile simulated Separation Estimate
    # Recreates: s_estimate = s_distorted_target + e_interference + e_artifacts
    estimate_waveform = target_distorted + 0.15 * interferer_waveform + artifact_noise

    return target_waveform, interferer_waveform, estimate_waveform, sampling_frequency


@pytest.fixture(scope="module")
def audio_files_fixture(
        synthetic_audio_data: Tuple[np.ndarray, np.ndarray, np.ndarray, float]
) -> Generator[Tuple[pathlib.Path, pathlib.Path, pathlib.Path], None, None]:
    """
    Writes synthetic waveforms to temporary WAV files on disk to test file-based modes.
    """
    target, interferer, estimate, fs = synthetic_audio_data

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        target_path = temp_path / "target_source.wav"
        interferer_path = temp_path / "interferer.wav"
        estimate_path = temp_path / "estimate.wav"

        sf.write(target_path, target, int(fs))
        sf.write(interferer_path, interferer, int(fs))
        sf.write(estimate_path, estimate, int(fs))

        yield target_path, interferer_path, estimate_path
