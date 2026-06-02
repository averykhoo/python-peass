"""
PEASS Test Suite - Auditory Physiological Model Tests
"""


import numpy as np
import pytest
import scipy.signal as signal

from peass.auditory_model import ModulationProcessingType
from peass.auditory_model import _HAS_NUMBA
from peass.auditory_model import _fallback_adaptation_loops
from peass.auditory_model import _fallback_fused_auditory_kernel
from peass.auditory_model import generate_auditory_internal_representation
from peass.auditory_model import simulate_auditory_nerve_adaptation
from peass.auditory_model import simulate_inner_haircell_transduction

if _HAS_NUMBA:
    from peass.auditory_model import (
        _numba_adaptation_loops_kernel,
        _numba_fused_auditory_kernel,
    )
from peass.gammatone import GammatoneAnalyzer, GammatoneSynthesizer


@pytest.mark.parametrize("sampling_frequency_hz", [8000.0, 16000.0, 44100.0])
def test_gammatone_analysis_reconstruction_fidelity(sampling_frequency_hz):
    """
    Verifies that the Gammatone analysis and synthesis filterbank reconstruct
    the input signal with high fidelity (cross-correlation close to 1.0)
    after accounting for group delay.
    """
    duration_seconds = 0.5
    num_samples = int(duration_seconds * sampling_frequency_hz)
    time_steps = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    signal_input = np.sin(2.0 * np.pi * 440.0 * time_steps)

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=sampling_frequency_hz,
        lower_cutoff_frequency_hz=80.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=3000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    subbands = analyzer.process(signal_input)
    assert subbands.ndim == 2

    desired_delay_seconds = 0.004
    synthesizer = GammatoneSynthesizer(analyzer, desired_delay_seconds=desired_delay_seconds)
    reconstructed = synthesizer.process(subbands)

    # Slice arrays to account for the synthesis delay buffer
    delay_samples = int(round(desired_delay_seconds * sampling_frequency_hz))
    original_slice = signal_input[delay_samples:-delay_samples]
    reconstructed_slice = reconstructed[2 * delay_samples: len(original_slice) + 2 * delay_samples]

    min_len = min(len(original_slice), len(reconstructed_slice))
    original_slice = original_slice[:min_len]
    reconstructed_slice = reconstructed_slice[:min_len]

    # Assert high-fidelity signal reconstruction
    corr = np.corrcoef(original_slice, reconstructed_slice)[0, 1]
    assert corr > 0.90, f"Reconstruction fidelity too low: {corr:.4f}"


def test_haircell_and_adaptation_properties(
        synthetic_audio_data: tuple[np.ndarray, np.ndarray, np.ndarray, float]
):
    """
    Verifies physiological properties of hair cell transduction and adaptation,
    including non-negativity and compression.
    """
    target, _, _, sampling_frequency_hz = synthetic_audio_data
    subband_signals = target.T

    # 1. Hair cell halfway-rectification constraint
    transduced = simulate_inner_haircell_transduction(subband_signals, sampling_frequency_hz)
    assert transduced.shape == subband_signals.shape
    assert np.all(transduced >= -1e-15)

    # 2. Auditory nerve adaptation compression and boundary scaling
    adapted = simulate_auditory_nerve_adaptation(transduced, sampling_frequency_hz)
    assert adapted.shape == subband_signals.shape
    assert np.max(adapted) > 0.0
    assert np.max(adapted) <= 1e7


def test_auditory_numba_fallback_equivalence():
    """
    Mathematically verifies that JIT-compiled Numba kernels and pure Python/SciPy fallbacks
    yield identical numerical results down to machine precision.
    """
    if not _HAS_NUMBA:
        pytest.skip("Numba is not installed or enabled in this environment.")

    rng = np.random.default_rng(seed=123)
    subbands = rng.normal(0.0, 0.5, (4, 1000))
    fs = 16000.0

    # 1. Compare Inner Hair Cell Transduction
    numba_ihc = simulate_inner_haircell_transduction(subbands, fs)

    haircell_filter_gain = np.exp(-np.pi * 2000.0 / fs)
    fallback_ihc = np.maximum(subbands, 0.0)
    fallback_ihc = signal.lfilter([1.0 - haircell_filter_gain], [1.0, -haircell_filter_gain], fallback_ihc, axis=-1)

    np.testing.assert_allclose(numba_ihc, fallback_ihc, rtol=1e-12, atol=1e-12)

    # 2. Compare Adaptation Loops
    adaptation_loop_bandwidths = 1.0 / (np.pi * np.array([0.005, 0.05, 0.129, 0.253, 0.5]))
    thresh = 10.0 ** -5.0  # (100 dB SPL threshold)

    numba_adapt = _numba_adaptation_loops_kernel(numba_ihc, fs, adaptation_loop_bandwidths, thresh)
    fallback_adapt = _fallback_adaptation_loops(numba_ihc, fs, adaptation_loop_bandwidths, thresh)
    np.testing.assert_allclose(numba_adapt, fallback_adapt, rtol=1e-12, atol=1e-12)

    # 3. Compare Fused Kernels
    numba_fused = _numba_fused_auditory_kernel(subbands, fs, haircell_filter_gain, adaptation_loop_bandwidths, thresh)
    fallback_fused = _fallback_fused_auditory_kernel(subbands, fs, haircell_filter_gain, adaptation_loop_bandwidths,
                                                     thresh)
    np.testing.assert_allclose(numba_fused, fallback_fused, rtol=1e-12, atol=1e-12)


def test_internal_auditory_representation_modulation_properties(
        synthetic_audio_data: tuple[np.ndarray, np.ndarray, np.ndarray, float]
):
    """
    Verifies the real/complex modulation property mapping of the internal auditory representation.
    """
    target, _, _, sampling_frequency_hz = synthetic_audio_data

    # LOWPASS mode (modulation frequency <= 10 Hz) must yield real-valued outputs
    representation_lp, processed_fs = generate_auditory_internal_representation(
        target, sampling_frequency_hz, modulation_processing_type=ModulationProcessingType.LOWPASS
    )
    assert representation_lp.ndim == 3
    assert processed_fs == 100.0
    assert np.allclose(np.imag(representation_lp), 0.0, atol=1e-15)

    # FILTERBANK mode (modulation frequencies > 10 Hz) converts complex envelopes to real magnitudes
    representation_fb, processed_fs_fb = generate_auditory_internal_representation(
        target, sampling_frequency_hz, modulation_processing_type=ModulationProcessingType.FILTERBANK
    )
    assert representation_fb.ndim == 3
    assert processed_fs_fb == 800.0
    assert np.allclose(np.imag(representation_fb), 0.0, atol=1e-15)


def test_generate_representation_input_shapes_and_resampling():
    """
    Tests Auditory Representation with alternative input formats and resampling conditions.
    """
    # 1. Row vector shape (1, N) to verify transposing logic
    signal_row = np.random.randn(1, 1000)
    rep_row, fs_row = generate_auditory_internal_representation(signal_row, 16000.0)
    assert rep_row.ndim == 3

    # 2. Column vector shape (N, 1)
    signal_col = np.random.randn(1000, 1)
    rep_col, fs_col = generate_auditory_internal_representation(signal_col, 16000.0)
    assert rep_col.ndim == 3

    # 3. High sampling rate condition
    signal_high = np.random.randn(1000)
    rep_high, fs_high = generate_auditory_internal_representation(signal_high, 48000.0)
    assert rep_high.ndim == 3
