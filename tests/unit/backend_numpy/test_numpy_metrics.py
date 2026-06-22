"""
PEASS Test Suite - Metrics Unit Tests
File path: tests/unit/test_metrics.py
"""

import numpy as np
import pytest

from peass.backend_numpy.metrics import calculate_auditory_similarity_metric
from peass.backend_numpy.metrics import calculate_bss_eval_energy_ratios


@pytest.mark.unit
def test_calculate_energy_ratios_analytical():
    """Tests the energy calculations using hardcoded, predictable signals."""
    true_source = np.array([2.0, 2.0, 2.0])
    target_distortion = np.array([0.2, 0.2, 0.2])
    interference = np.array([0.1, 0.1, 0.1])
    artifacts = np.array([0.05, 0.05, 0.05])

    (
        source_to_spatial_distortion_ratio,
        source_to_interference_ratio,
        source_to_artifacts_ratio,
        source_to_distortion_ratio
    ) = calculate_bss_eval_energy_ratios(true_source, target_distortion, interference, artifacts)

    assert np.isclose(source_to_spatial_distortion_ratio, 20.0)
    assert np.isclose(source_to_interference_ratio, 10.0 * np.log10(484.0))
    assert np.isclose(source_to_artifacts_ratio, 10.0 * np.log10(2116.0))
    assert np.isclose(source_to_distortion_ratio, 10.0 * np.log10(32.653061224))


@pytest.mark.unit
def test_pemo_similarity_identity():
    """Verifies that comparing an internal representation with itself yields high similarity."""
    bands, samples, modulations = 4, 200, 1
    fs = 100.0

    ref_rep = np.abs(np.random.randn(bands, samples, modulations))
    test_rep = ref_rep.copy()

    sim = calculate_auditory_similarity_metric(ref_rep, test_rep, fs)
    assert 0.9 <= sim <= 1.0


@pytest.mark.unit
def test_perceptual_assimilation_behavior():
    """
    Verifies that the assimilation masking rules correctly pull attenuated signals
    closer to reference thresholds, increasing perceptual similarity.
    """
    fs = 100.0
    num_bands, num_samples, num_modulations = 4, 200, 1

    rng = np.random.default_rng(seed=42)
    ref_rep = np.abs(rng.normal(1.0, 0.2, (num_bands, num_samples, num_modulations)))

    # Introduce non-uniform noise to test representation
    test_rep = np.abs(ref_rep + rng.normal(0.0, 0.1, ref_rep.shape))
    test_rep_scaled = 0.5 * test_rep

    # Standard unassimilated Pearson correlation
    lref = ref_rep.ravel() - np.mean(ref_rep)
    ltest = test_rep_scaled.ravel() - np.mean(test_rep_scaled)
    standard_corr = np.sum(lref * ltest) / np.sqrt(np.sum(lref ** 2) * np.sum(ltest ** 2))

    # Model assimilated metric (assimilation pulls the noisy signal closer to ref, masking the noise)
    perceptual_metric = calculate_auditory_similarity_metric(ref_rep, test_rep_scaled, fs)

    # Perceptual metric must be higher than standard correlation due to threshold masking
    assert perceptual_metric > standard_corr

    # Perfect identity constraint
    identity_metric = calculate_auditory_similarity_metric(ref_rep, ref_rep.copy(), fs)
    np.testing.assert_allclose(identity_metric, 1.0, atol=1e-5)
