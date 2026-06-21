"""
PEASS Test Suite - Auditory Physiological Model Unit Tests
"""

import numpy as np
import pytest
import scipy.signal as signal

from peass.backend_numpy.auditory_model import ModulationProcessingType
from peass.backend_numpy.auditory_model import _HAS_NUMBA
from peass.backend_numpy.auditory_model import _fallback_adaptation_loops
from peass.backend_numpy.auditory_model import _fallback_fused_auditory_kernel
from peass.backend_numpy.auditory_model import generate_auditory_internal_representation
from peass.backend_numpy.auditory_model import simulate_auditory_nerve_adaptation
from peass.backend_numpy.auditory_model import simulate_inner_haircell_transduction

if _HAS_NUMBA:
    from peass.backend_numpy.auditory_model import (
        _numba_adaptation_loops_kernel,
        _numba_fused_auditory_kernel,
    )


@pytest.mark.unit
def test_internal_auditory_representation(synthetic_audio_data):
    """
    Verifies the dimensions and channel counts of the internal auditory representation.
    """
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_haircell_frequency_selectivity():
    """
    Verifies that the Inner Hair Cell model attenuates high frequencies
    consistent with its 1 kHz lowpass membrane shear limit.
    """
    fs = 16000.0
    time_steps = np.linspace(0.0, 0.5, int(0.5 * fs), endpoint=False)

    # 100 Hz (Passband) vs 8000 Hz (Severe attenuation)
    low_sine = np.sin(2.0 * np.pi * 100.0 * time_steps)
    high_sine = np.sin(2.0 * np.pi * 8000.0 * time_steps)

    low_ihc = simulate_inner_haircell_transduction(low_sine[np.newaxis, :], fs)
    high_ihc = simulate_inner_haircell_transduction(high_sine[np.newaxis, :], fs)

    # Half-wave rectification constraint
    assert np.all(low_ihc >= -1e-15)
    assert np.all(high_ihc >= -1e-15)

    rms_low = np.sqrt(np.mean(low_ihc ** 2))
    rms_high = np.sqrt(np.mean(high_ihc ** 2))

    # Expect strong attenuation at 8 kHz
    attenuation_db = 20.0 * np.log10(rms_low / (rms_high + 1e-15))
    assert attenuation_db > 15.0


@pytest.mark.unit
def test_auditory_nerve_adaptation_dynamics():
    """
    Verifies that the non-linear adaptation loops correctly model onset-overshoot
    and slow synaptic masking decay curves.
    """
    fs = 16000.0
    num_samples = int(0.6 * fs)

    pulse = np.zeros(num_samples)
    pulse[:int(0.2 * fs)] = 1.0  # On for 200ms

    adapted = simulate_auditory_nerve_adaptation(pulse[np.newaxis, :], fs)
    adapted_1d = adapted[0, :]

    # 1. Onset overshoot: First 20ms amplitude must be larger than standard middle pulse
    onset_peak = np.max(adapted_1d[:int(0.02 * fs)])
    steady_state = np.mean(adapted_1d[int(0.1 * fs):int(0.18 * fs)])
    assert onset_peak > steady_state

    # 2. Slow decay: Immediately after the offset, the signal undershoots (adaptation suppression)
    # and slowly recovers back up toward the resting threshold.
    offset_sample = int(0.2 * fs)
    immediate_after = adapted_1d[offset_sample + int(0.005 * fs)]
    later_after = adapted_1d[offset_sample + int(0.1 * fs)]

    assert immediate_after < later_after
    assert np.all(adapted_1d >= -300.0)


@pytest.mark.unit
def test_haircell_and_adaptation_properties(synthetic_audio_data):
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_internal_auditory_representation_modulation_properties(synthetic_audio_data):
    """
    Verifies the real/complex modulation property mapping of the internal auditory representation,
    along with target sub-dimension shape constraints across modulation modes.
    """
    target, _, _, sampling_frequency_hz = synthetic_audio_data

    # 1. LOWPASS mode (modulation frequency <= 10 Hz) must yield real-valued outputs
    representation_lp, processed_fs = generate_auditory_internal_representation(
        target, sampling_frequency_hz, modulation_processing_type=ModulationProcessingType.LOWPASS
    )
    assert representation_lp.ndim == 3
    assert processed_fs == 100.0
    assert np.allclose(np.imag(representation_lp), 0.0, atol=1e-15)
    assert representation_lp.shape[2] == 1  # Verify single subband channel constraint

    # 2. FILTERBANK mode (modulation frequencies > 10 Hz) converts complex envelopes to real magnitudes
    representation_fb, processed_fs_fb = generate_auditory_internal_representation(
        target, sampling_frequency_hz, modulation_processing_type=ModulationProcessingType.FILTERBANK
    )
    assert representation_fb.ndim == 3
    assert processed_fs_fb == 800.0
    assert np.allclose(np.imag(representation_fb), 0.0, atol=1e-15)
    assert representation_fb.shape[2] == 8  # Verify eight subband channel constraint


@pytest.mark.unit
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
