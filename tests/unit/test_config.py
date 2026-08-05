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
