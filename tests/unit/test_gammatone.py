"""
PEASS Test Suite - Gammatone Filterbank Mathematics Unit Tests
"""

import numpy as np
import pytest

from peass.gammatone import GammatoneAnalyzer
from peass.gammatone import GammatoneFilter
from peass.gammatone import calculate_equivalent_rectangular_bandwidth
from peass.gammatone import convert_equivalent_rectangular_bandwidth_scale_to_frequency
from peass.gammatone import convert_frequency_to_equivalent_rectangular_bandwidth_scale
from peass.gammatone import get_equivalent_rectangular_bandwidth_center_frequencies


@pytest.mark.parametrize("frequency_hz", [50.0, 100.0, 440.0, 1000.0, 5000.0, 12000.0])
def test_erb_scale_inversion(frequency_hz):
    """
    Tests that frequency_to_erb_scale and erb_scale_to_frequency are mathematical inverses.
    """
    erb = convert_frequency_to_equivalent_rectangular_bandwidth_scale(frequency_hz)
    f_recon = convert_equivalent_rectangular_bandwidth_scale_to_frequency(erb)
    assert np.isclose(frequency_hz, f_recon)


def test_calculate_erb_bandwidth_values():
    """
    Checks calculation of ERB bandwidths against expected values.
    Formula: ERB = 24.7 * (0.00437 * fc + 1.0)
    """
    # At fc = 0: ERB = 24.7
    assert np.isclose(calculate_equivalent_rectangular_bandwidth(0.0), 24.7)

    # At fc = 1000: ERB = 24.7 * (4.37 + 1.0) = 132.639
    assert np.isclose(calculate_equivalent_rectangular_bandwidth(1000.0), 132.639)


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
