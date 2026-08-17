"""
PEASS Test Suite - Configuration validation
File path: tests/unit/test_config.py
"""

import pytest

from peass.config import DecompositionConfiguration


@pytest.mark.unit
def test_default_configuration_is_constructible():
    """The documented defaults must not trip their own validation."""
    assert DecompositionConfiguration().segmentation_factor == 1


@pytest.mark.unit
@pytest.mark.parametrize("factor", [0, 2, 4])
def test_segmentation_factor_other_than_one_is_rejected(factor):
    """
    MATLAB's segmented decomposition path was never ported.

    The field is kept so `options.segmentationFactor` maps onto a recognizable name,
    but it must fail loudly rather than silently decompose in one piece: a user
    following MATLAB's advice to raise it after an out-of-memory error would
    otherwise get no relief and no warning.
    """
    with pytest.raises(NotImplementedError, match="segmentation_factor"):
        DecompositionConfiguration(segmentation_factor=factor)


@pytest.mark.unit
@pytest.mark.parametrize("factor", [0, -1, -10])
def test_resample_filter_half_length_factor_below_one_is_rejected(factor):
    """
    Below 1 the fast polyphase path is silently wrong rather than absent.

    It does not raise and it does not degrade gracefully: at hf = 0 it returns finite
    numbers that disagree with its own padded reference by O(1) (measured 0.39 to 2.13
    across 33 swept configurations, against 2.22e-16 for hf >= 1), because the grid
    algebra needs `right_pad >= (hf-1)*down + 1 > 0`. The gradient path routes to the
    padded reference, so the two paths disagree there as well.

    This is the only kind of input the dataclass rejects on numerical grounds rather
    than on unimplemented-feature grounds, hence `ValueError` rather than
    `NotImplementedError`.
    """
    with pytest.raises(ValueError, match="resample_filter_half_length_factor"):
        DecompositionConfiguration(resample_filter_half_length_factor=factor)


@pytest.mark.unit
@pytest.mark.parametrize("factor", [1, 3, 10])
def test_resample_filter_half_length_factor_at_or_above_one_is_accepted(factor):
    """The guard must not move the boundary: 1 is the first valid value, not 2."""
    assert DecompositionConfiguration(
        resample_filter_half_length_factor=factor
    ).resample_filter_half_length_factor == factor
