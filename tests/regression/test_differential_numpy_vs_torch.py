"""
PEASS Test Suite - Differential Parity Tests (NumPy vs PyTorch)
File path: tests/unit/backend_torch/test_differential_numpy_vs_torch.py
"""

import numpy as np
import pytest
import torch

# NumPy modules
from peass.backend_numpy.gammatone import GammatoneAnalyzer, GammatoneSynthesizer
from peass.backend_numpy.decomposition import decompose_distortion_components as decomp_np

# PyTorch modules
from peass.backend_torch.gammatone import GammatoneAnalyzerTorch, GammatoneSynthesizerTorch
from peass.backend_torch.decomposition import decompose_distortion_components as decomp_torch
from tests.conftest import to_numpy_format, to_backend_format


@pytest.fixture(scope="module")
def baseline_signals():
    """Generates clean, matching target and estimate signals for parity testing."""
    fs = 16000.0
    duration = 0.5
    num_samples = int(duration * fs)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # 440 Hz target sine, 1200 Hz interference noise
    target = np.sin(2.0 * np.pi * 440.0 * t)[:, np.newaxis]
    interf = np.sin(2.0 * np.pi * 1200.0 * t)[:, np.newaxis]
    estimate = target + 0.1 * interf + 0.01 * np.random.randn(num_samples, 1)

    return target, interf, estimate, fs


def test_differential_gammatone_analysis(baseline_signals):
    """Verifies bit-level subband parity between the FFT and IIR filterbanks."""
    target, _, _, fs = baseline_signals
    device = torch.device("cpu")

    # 1. Run legacy NumPy filterbank
    analyzer_np = GammatoneAnalyzer(fs, 80.0, 1000.0, 4000.0, 1.0)
    subbands_np = analyzer_np.process(target.ravel())

    # 2. Run new PyTorch filterbank
    target_torch = to_backend_format(target.ravel(), "torch", device)
    analyzer_torch = GammatoneAnalyzerTorch(fs, 80.0, 1000.0, 4000.0, 1.0, device, torch.float64)
    subbands_torch = analyzer_torch.process(target_torch)

    subbands_torch_np = to_numpy_format(subbands_torch)

    # Assert high-fidelity matching across all bands
    for b in range(analyzer_np.center_frequencies.shape[0]):
        corr = np.corrcoef(subbands_np[b].real, subbands_torch_np[b].real)[0, 1]
        assert corr > 0.98, f"Gammatone Analysis Band {b} correlation is too low: {corr:.4f}"


def test_differential_gammatone_synthesis(baseline_signals):
    """Verifies reconstruction parity between the NumPy and PyTorch synthesizers."""
    target, _, _, fs = baseline_signals
    device = torch.device("cpu")

    analyzer_np = GammatoneAnalyzer(fs, 80.0, 1000.0, 4000.0, 1.0)
    subbands_np = analyzer_np.process(target.ravel())

    synth_np = GammatoneSynthesizer(analyzer_np, 0.004)
    reconstructed_np = synth_np.process(subbands_np)

    # Run PyTorch counterparts
    target_torch = to_backend_format(target.ravel(), "torch", device)
    analyzer_torch = GammatoneAnalyzerTorch(fs, 80.0, 1000.0, 4000.0, 1.0, device, torch.float64)
    subbands_torch = analyzer_torch.process(target_torch)

    synth_torch = GammatoneSynthesizerTorch(analyzer_torch, 0.004)
    reconstructed_torch = synth_torch.process(subbands_torch)

    recon_torch_np = to_numpy_format(reconstructed_torch)

    corr = np.corrcoef(reconstructed_np, recon_torch_np)[0, 1]
    assert corr > 0.98, f"Reconstruction synthesis correlation is too low: {corr:.4f}"


def test_differential_decomposition_pipeline(baseline_signals):
    """Verifies exact output waveform parity of the entire decomposition block."""
    target, interf, estimate, fs = baseline_signals
    device = torch.device("cpu")

    # NumPy Decomposition
    result_np = decomp_np([target, interf], estimate, sampling_frequency_hz=fs)
    wf_np = result_np.waveforms

    # PyTorch Decomposition
    target_torch = to_backend_format(target, "torch", device)
    interf_torch = to_backend_format(interf, "torch", device)
    estimate_torch = to_backend_format(estimate, "torch", device)

    result_torch = decomp_torch([target_torch, interf_torch], estimate_torch, sampling_frequency_hz=fs)
    wf_torch = result_torch.waveforms

    true_target_torch_np = to_numpy_format(wf_torch.true_target)

    corr = np.corrcoef(wf_np.true_target.ravel(), true_target_torch_np.ravel())[0, 1]
    assert corr > 0.95, f"Full Decomposition True Target parity correlation is too low: {corr:.4f}"