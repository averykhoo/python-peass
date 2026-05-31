"""
PEASS Test Suite - Auditory Physiological Model Tests
"""

from typing import Tuple

import numpy as np

from peass.auditory_model import ModulationProcessingType
from peass.auditory_model import _fallback_adaptation_loops
from peass.auditory_model import _fallback_fused_auditory_kernel
from peass.auditory_model import generate_auditory_internal_representation
from peass.auditory_model import simulate_auditory_nerve_adaptation
from peass.auditory_model import simulate_inner_haircell_transduction
from peass.gammatone import GammatoneAnalyzer
from peass.gammatone import GammatoneSynthesizer


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


def test_auditory_fallback_kernels():
    """Explicitly tests the fallback implementations to guarantee coverage on JIT environments."""
    subband_signals = np.random.randn(4, 100)
    adaptation_loop_bandwidths = 1.0 / (np.pi * np.array([0.005, 0.05, 0.129, 0.253, 0.5]))

    # 1. Test fallback adaptation loops directly
    res_loops = _fallback_adaptation_loops(
        subband_signals=subband_signals,
        sampling_frequency_hz=16000.0,
        adaptation_bandwidths=adaptation_loop_bandwidths,
        absolute_hearing_threshold=1e-5
    )
    assert res_loops.shape == subband_signals.shape

    # 2. Test fallback fused auditory kernel directly
    res_fused = _fallback_fused_auditory_kernel(
        subband_signals=subband_signals,
        sampling_frequency_hz=16000.0,
        haircell_filter_gain=np.exp(-np.pi * 2000.0 / 16000.0),
        adaptation_bandwidths=adaptation_loop_bandwidths,
        absolute_hearing_threshold=1e-5
    )
    assert res_fused.shape == subband_signals.shape


def test_generate_representation_input_shapes_and_resampling():
    """Tests generate_auditory_internal_representation with alternative shapes and high sample rates."""
    # Row vector shape (1, N) to test transpose branch
    signal_row = np.random.randn(1, 1000)
    rep_row, fs_row = generate_auditory_internal_representation(signal_row, 16000.0)
    assert rep_row.ndim == 3

    # Column vector shape (N, 1)
    signal_col = np.random.randn(1000, 1)
    rep_col, fs_col = generate_auditory_internal_representation(signal_col, 16000.0)
    assert rep_col.ndim == 3

    # High sampling rate (e.g., 48000 Hz) to test skipped resampling branch
    signal_high = np.random.randn(1000)
    rep_high, fs_high = generate_auditory_internal_representation(signal_high, 48000.0)
    assert rep_high.ndim == 3
