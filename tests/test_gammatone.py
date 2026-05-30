"""
PEASS Test Suite - Gammatone Filterbank Mathematics Unit Tests
"""

import numpy as np

from peass.gammatone import (
    GammatoneAnalyzer,
    GammatoneFilter,
    calculate_erb_bandwidth,
    erb_scale_to_frequency,
    frequency_to_erb_scale,
    get_center_frequencies,
)


def test_erb_scale_inversion():
    """
    Tests that frequency_to_erb_scale and erb_scale_to_frequency are mathematical inverses.
    """
    frequencies = [50.0, 100.0, 440.0, 1000.0, 5000.0, 12000.0]
    for f in frequencies:
        erb = frequency_to_erb_scale(f)
        f_recon = erb_scale_to_frequency(erb)
        assert np.isclose(f, f_recon)


def test_calculate_erb_bandwidth_values():
    """
    Checks calculation of ERB bandwidths against expected values.
    Formula: ERB = 24.7 * (0.00437 * fc + 1.0)
    """
    # At fc = 0: ERB = 24.7
    assert np.isclose(calculate_erb_bandwidth(0.0), 24.7)

    # At fc = 1000: ERB = 24.7 * (4.37 + 1.0) = 132.639
    assert np.isclose(calculate_erb_bandwidth(1000.0), 132.639)


def test_get_center_frequencies_range():
    """
    Verifies that get_center_frequencies produces frequencies within bounds and contains the base frequency.
    """
    lower = 100.0
    base = 1000.0
    upper = 8000.0
    filters_per_erb = 1.0

    cfs = get_center_frequencies(filters_per_erb, lower, base, upper)

    assert cfs[0] >= lower
    assert cfs[-1] <= upper
    # There should be an exact match for the base frequency
    assert np.any(np.isclose(cfs, base))


def test_gammatone_filter_state_clearing():
    """
    Verifies state setting and clearing inside GammatoneFilter.
    """
    filt = GammatoneFilter(sampling_frequency=16000.0, center_frequency=1000.0)

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


def test_gammatone_analyzer_state_clearing():
    """
    Verifies state setting and clearing inside GammatoneAnalyzer.
    """
    analyzer = GammatoneAnalyzer(
        sampling_frequency=16000.0,
        lower_cutoff_hz=100.0,
        specified_center_hz=1000.0,
        upper_cutoff_hz=4000.0,
        filters_per_erb=1.0
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