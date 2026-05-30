"""
PEASS Test Suite - Metrics Unit Tests
"""
import numpy as np

from peass.metrics import calculate_bss_eval_energy_ratios, calculate_auditory_similarity_metric

def test_calculate_energy_ratios_analytical():
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

def test_pemo_similarity_identity():
    bands, samples, modulations = 4, 200, 1
    fs = 100.0

    ref_rep = np.abs(np.random.randn(bands, samples, modulations))
    test_rep = ref_rep.copy()

    sim = calculate_auditory_similarity_metric(ref_rep, test_rep, fs)
    assert 0.9 <= sim <= 1.0