"""
PEASS Test Suite - Gammatone Filterbank Mathematics Unit Tests
File path: tests/unit/test_gammatone.py
"""

import numpy as np
import pytest

from peass.gammatone import GammatoneAnalyzer
from peass.gammatone import GammatoneFilter
from peass.gammatone import GammatoneSynthesizer
from peass.gammatone import calculate_equivalent_rectangular_bandwidth
from peass.gammatone import convert_equivalent_rectangular_bandwidth_scale_to_frequency
from peass.gammatone import convert_frequency_to_equivalent_rectangular_bandwidth_scale
from peass.gammatone import get_equivalent_rectangular_bandwidth_center_frequencies


@pytest.mark.unit
@pytest.mark.parametrize("frequency_hz", [50.0, 100.0, 440.0, 1000.0, 5000.0, 12000.0])
def test_erb_scale_inversion(frequency_hz):
    """
    Tests that frequency_to_erb_scale and erb_scale_to_frequency are mathematical inverses.
    """
    erb = convert_frequency_to_equivalent_rectangular_bandwidth_scale(frequency_hz)
    f_recon = convert_equivalent_rectangular_bandwidth_scale_to_frequency(erb)
    assert np.isclose(frequency_hz, f_recon)


@pytest.mark.unit
def test_calculate_erb_bandwidth_values():
    """
    Checks calculation of ERB bandwidths against expected values.
    Formula: ERB = 24.7 * (0.00437 * fc + 1.0)
    """
    # At fc = 0: ERB = 24.7
    assert np.isclose(calculate_equivalent_rectangular_bandwidth(0.0), 24.7)

    # At fc = 1000: ERB = 24.7 * (4.37 + 1.0) = 132.639
    assert np.isclose(calculate_equivalent_rectangular_bandwidth(1000.0), 132.639)


@pytest.mark.unit
def test_get_center_frequencies_range():
    """
    Verifies that get_center_frequencies produces frequencies within bounds and contains the base frequency.
    """
    lower = 100.0
    base = 1000.0
    upper = 8000.0
    filters_per_erb = 1.0

    cfs = get_equivalent_rectangular_bandwidth_center_frequencies(filters_per_erb, lower, base, upper)

    assert cfs[0] >= lower
    assert cfs[-1] <= upper
    # There should be an exact match for the base frequency
    assert np.any(np.isclose(cfs, base))


@pytest.mark.unit
def test_gammatone_filter_state_clearing():
    """
    Verifies state setting and clearing inside GammatoneFilter.
    """
    filt = GammatoneFilter(sampling_frequency_hz=16000.0, center_frequency_hz=1000.0)

    # Assert state is initially zeros
    assert np.all(filt.state == 0.0)

    # Process some noise to populate the state
    input_noise = np.random.randn(100)
    filt.process(input_noise)

    # State should now be populated with non-zero values
    assert np.any(filt.state != 0.0)

    # Clear the state
    filt.clear_state()
    assert np.all(filt.state == 0.0)


@pytest.mark.unit
def test_gammatone_analyzer_state_clearing():
    """
    Verifies state setting and clearing inside GammatoneAnalyzer.
    """
    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=16000.0,
        lower_cutoff_frequency_hz=100.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=4000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    # All filter states should start at zero
    for filt in analyzer.filters:
        assert np.all(filt.state == 0.0)

    # Process signal
    analyzer.process(np.random.randn(50))

    # States should be non-zero
    for filt in analyzer.filters:
        assert np.any(filt.state != 0.0)

    # Clear states
    analyzer.clear_state()
    for filt in analyzer.filters:
        assert np.all(filt.state == 0.0)


@pytest.mark.unit
def test_gammatone_reconstruction_parameterized(synthetic_signal):
    """
    Verifies that the Gammatone analysis-synthesis filterbank correctly
    reconstructs signals across diverse parameterized waveforms.
    """
    waveform, fs = synthetic_signal
    sig_1d = waveform.ravel()

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=fs,
        lower_cutoff_frequency_hz=80.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=6000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    subbands = analyzer.process(sig_1d)
    desired_delay_sec = 0.004
    synthesizer = GammatoneSynthesizer(analyzer, desired_delay_seconds=desired_delay_sec)
    reconstructed = synthesizer.process(subbands)

    delay_samples = int(round(desired_delay_sec * fs))

    # Slice to discard transient filter-delay periods
    original_sliced = sig_1d[delay_samples:-delay_samples]
    reconstructed_sliced = reconstructed[2 * delay_samples:]

    min_len = min(len(original_sliced), len(reconstructed_sliced))
    original_sliced = original_sliced[:min_len]
    reconstructed_sliced = reconstructed_sliced[:min_len]

    # Verify high phase and shape matching through cross-correlation
    corr = np.corrcoef(original_sliced, reconstructed_sliced)[0, 1]
    assert corr > 0.85


@pytest.mark.unit
def test_gammatone_fallback_vs_jit_equivalence():
    """
    Verifies that disabling Numba JIT acceleration forces the filterbank
    to run on the fallback path and produces identical outputs down to float precision.
    """
    import peass.gammatone as gammatone
    if not gammatone._HAS_NUMBA:
        pytest.skip("Numba is not installed or enabled in this environment.")
    from peass.gammatone import GammatoneAnalyzer

    # Generate a random signal
    rng = np.random.default_rng(seed=123)
    signal_input = rng.normal(0.0, 0.5, 1000)
    fs = 16000.0

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=fs,
        lower_cutoff_frequency_hz=80.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=4000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    # 1. Run with JIT active
    original_has_numba = gammatone._HAS_NUMBA
    try:
        gammatone._HAS_NUMBA = True
        analyzer.clear_state()
        jit_output = analyzer.process(signal_input)

        # 2. Force fallback path by mocking _HAS_NUMBA as False
        gammatone._HAS_NUMBA = False
        analyzer.clear_state()
        fallback_output = analyzer.process(signal_input)

        # Assert mathematical parity down to machine precision
        np.testing.assert_allclose(jit_output, fallback_output, rtol=1e-12, atol=1e-12)

    finally:
        # Restore original setting
        gammatone._HAS_NUMBA = original_has_numba


@pytest.mark.unit
@pytest.mark.parametrize("sampling_frequency_hz", [8000.0, 16000.0, 44100.0])
def test_gammatone_analysis_reconstruction(sampling_frequency_hz):
    """
    Checks the basic array shape and dimensions of the reconstructed signal from
    the analysis-synthesis filterbank.
    """
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


@pytest.mark.unit
def test_filterbank_dc_offset_rejection():
    """
    Verifies that the auditory filterbank correctly blocks constant DC bias offsets.
    """
    fs = 16000.0
    time_steps = np.linspace(0.0, 0.5, int(0.5 * fs), endpoint=False)

    dc_bias = 2.5
    signal_with_dc = np.sin(2.0 * np.pi * 500.0 * time_steps) + dc_bias

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=fs,
        lower_cutoff_frequency_hz=235.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=6000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    subbands = np.real(analyzer.process(signal_with_dc))

    # All subband outputs must have DC offset rejected (attenuation > 50 dB)
    for band_idx in range(subbands.shape[0]):
        assert np.abs(np.mean(subbands[band_idx, :])) < 1e-2


@pytest.mark.unit
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
