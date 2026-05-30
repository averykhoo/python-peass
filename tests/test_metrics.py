"""
PEASS Test Suite - Metrics Unit Tests
File path: tests/test_metrics.py
"""

import numpy as np

from peass.metrics import calculate_energy_ratios
from peass.metrics import pemo_similarity_metric


def test_calculate_energy_ratios_analytical():
    """
    Verifies that the energy ratio math matches analytical expectations exactly.
    """
    s_true = np.array([2.0, 2.0, 2.0])
    e_target = np.array([0.2, 0.2, 0.2])
    e_interf = np.array([0.1, 0.1, 0.1])
    e_artif = np.array([0.05, 0.05, 0.05])

    # Expected values derived from BSS Eval energy criteria equations:
    # ISR = 10 * log10(sum(s_true**2) / sum(e_target**2)) = 10 * log10(12.0 / 0.12) = 20.0 dB
    # SIR = 10 * log10(sum((s_true + e_target)**2) / sum(e_interf**2)) = 10 * log10(14.52 / 0.03) = 26.848 dB
    # SAR = 10 * log10(sum((s_true + e_target + e_interf)**2) / sum(e_artif**2)) = 10 * log10(15.87 / 0.0075) = 33.255 dB
    # SDR = 10 * log10(sum(s_true**2) / sum((e_target + e_interf + e_artif)**2)) = 10 * log10(12.0 / 0.3675) = 15.139 dB

    ISR, SIR, SAR, SDR = calculate_energy_ratios(s_true, e_target, e_interf, e_artif)

    assert np.isclose(ISR, 20.0)
    assert np.isclose(SIR, 10.0 * np.log10(484.0))
    assert np.isclose(SAR, 10.0 * np.log10(2116.0))
    assert np.isclose(SDR, 10.0 * np.log10(32.653061224))


def test_pemo_similarity_identity():
    """
    Verifies that comparing two identical internal representations yields
    a similarity index approaching 1.0.
    """
    bands, samples, modulations = 4, 200, 1
    fs = 100.0  # Internal sample rate

    # Generate random positive values representing adapted subbands
    ref_rep = np.abs(np.random.randn(bands, samples, modulations))
    test_rep = ref_rep.copy()

    sim = pemo_similarity_metric(ref_rep, test_rep, fs)

    # Comparing identical representations should result in near perfect correlation
    assert 0.9 <= sim <= 1.0
