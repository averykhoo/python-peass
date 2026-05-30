"""
PEASS Test Suite - Auditory Physiological Model Tests
"""

from typing import Tuple
import numpy as np

from peass.auditory_model import (
    ModulationProcessingType,
    generate_auditory_internal_representation,
    simulate_auditory_nerve_adaptation,
    simulate_inner_haircell_transduction
)
from peass.gammatone import GammatoneAnalyzer, GammatoneSynthesizer

def test_gammatone_analysis_reconstruction():
    sampling_frequency_hz = 16000.0
    duration_seconds = 0.5
    num_samples = int(duration_seconds * sampling_frequency_hz)
    signal_input = np.sin(2.0 * np.pi * 500.0 * np.linspace(0, duration_seconds, num_samples))

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=sampling_frequency_hz,
        lower_cutoff_frequency_hz=80.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=8000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    subbands = analyzer.process(signal_input)
    assert subbands.ndim == 2

    synthesizer = GammatoneSynthesizer(analyzer, desired_delay_seconds=0.004)
    reconstructed = synthesizer.process(subbands)
    assert reconstructed.ndim == 1
    assert reconstructed.shape[0] == num_samples


def test_haircell_and_adaptation_properties(
    synthetic_audio_data: Tuple[np.ndarray, np.ndarray, np.ndarray, float]
):
    target, _, _, sampling_frequency_hz = synthetic_audio_data
    subband_signals = target.T

    transduced = simulate_inner_haircell_transduction(subband_signals, sampling_frequency_hz)
    assert transduced.shape == subband_signals.shape
    assert np.all(transduced >= 0.0)

    adapted = simulate_auditory_nerve_adaptation(transduced, sampling_frequency_hz)
    assert adapted.shape == subband_signals.shape
    assert np.max(adapted) > 0.0
    assert np.max(adapted) <= 1e7


def test_internal_auditory_representation(
    synthetic_audio_data: Tuple[np.ndarray, np.ndarray, np.ndarray, float]
):
    target, _, _, sampling_frequency_hz = synthetic_audio_data

    representation, processed_fs = generate_auditory_internal_representation(
        target, sampling_frequency_hz, modulation_processing_type=ModulationProcessingType.LOWPASS
    )
    assert representation.ndim == 3
    assert processed_fs == 100.0
    assert representation.shape[2] == 1

    representation_fb, processed_fs_fb = generate_auditory_internal_representation(
        target, sampling_frequency_hz, modulation_processing_type=ModulationProcessingType.FILTERBANK
    )
    assert representation_fb.ndim == 3
    assert processed_fs_fb == 800.0
    assert representation_fb.shape[2] == 8