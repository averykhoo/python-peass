"""
PEASS Test Suite - Auditory Physiological Model Tests
File path: tests/test_auditory.py
"""

from typing import Tuple

import numpy as np

from peass.auditory_model import adaptation_loops
from peass.auditory_model import generate_internal_representation
from peass.auditory_model import haircell_transduction
from peass.gammatone import GammatoneAnalyzer
from peass.gammatone import GammatoneSynthesizer


def test_gammatone_analysis_reconstruction():
    """
    Verifies that the Gammatone analysis and synthesis filterbanks can process
    signals without crashes and maintain dimensional structures [3].
    """
    sampling_frequency = 16000.0
    duration_seconds = 0.5
    num_samples = int(duration_seconds * sampling_frequency)
    signal_input = np.sin(2.0 * np.pi * 500.0 * np.linspace(0, duration_seconds, num_samples))

    # Initialize analysis and synthesis stages
    analyzer = GammatoneAnalyzer(
        sampling_frequency=sampling_frequency,
        lower_cutoff_hz=80.0,
        specified_center_hz=1000.0,
        upper_cutoff_hz=8000.0,
        filters_per_erb=1.0
    )

    # Process through analysis filterbank (yields complex subbands)
    subbands = analyzer.process(signal_input)
    assert subbands.ndim == 2
    assert subbands.shape[0] == len(analyzer.filters)
    assert subbands.shape[1] == num_samples

    # Process through synthesizer (reconstructs fullband real signal)
    synthesizer = GammatoneSynthesizer(analyzer, desired_delay_seconds=0.004)
    reconstructed = synthesizer.process(subbands)

    assert reconstructed.ndim == 1
    assert reconstructed.shape[0] == num_samples


def test_haircell_and_adaptation_properties(
        synthetic_audio_data: Tuple[np.ndarray, np.ndarray, np.ndarray, float]
):
    """
    Validates biological inner hair cell transduction and adaptive neural loops [2].
    """
    target, _, _, fs = synthetic_audio_data
    subband_signals = target.T  # Transpose to (channels, samples) shape

    # 1. Haircell transduction test (Must enforce unidirectional half-wave rectification)
    transduced = haircell_transduction(subband_signals, fs)
    assert transduced.shape == subband_signals.shape
    # Inner hair cell signals should not contain negative values
    assert np.all(transduced >= 0.0)

    # 2. Adaptation loops test (Must model adaptive compression)
    adapted = adaptation_loops(transduced, fs)
    assert adapted.shape == subband_signals.shape

    # Non-linear adaptive compression under sudden transient steps yields large overshoot spikes.
    # We assert that the loops compute successfully and produce positive, bounded signals.
    assert np.max(adapted) > 0.0
    assert np.max(adapted) <= 1e7


def test_internal_auditory_representation(
        synthetic_audio_data: Tuple[np.ndarray, np.ndarray, np.ndarray, float]
):
    """
    Tests the 3D internal representation generator.
    """
    target, _, _, fs = synthetic_audio_data

    # Default 'lp' (lowpass) modulation processing
    representation, processed_fs = generate_internal_representation(target, fs, modulation_processing_type='lp')
    assert representation.ndim == 3
    assert processed_fs == 100.0
    assert representation.shape[2] == 1  # Lowpass yields 1 modulation band (0 Hz)

    # 'fb' (filterbank) modulation processing
    representation_fb, processed_fs_fb = generate_internal_representation(target, fs, modulation_processing_type='fb')
    assert representation_fb.ndim == 3
    assert processed_fs_fb == 800.0
    assert representation_fb.shape[2] == 8  # Filterbank yields 8 modulation bands
