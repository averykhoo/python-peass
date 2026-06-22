import math

import torch
import pytest
import numpy as np
from peass.backend_torch.gammatone import GammatoneAnalyzerTorch, GammatoneSynthesizerTorch
from tests.conftest import to_numpy_format


@pytest.mark.parametrize("device_str", ["cpu", "cuda", "mps"])
def test_torch_gammatone_filterbank(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if device_str == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available.")

    device = torch.device(device_str)

    # Use a predictable sine wave instead of random noise for strict reconstruction checks
    fs = 16000.0
    t = torch.linspace(0.0, 1.0, 16000, device=device, dtype=torch.float64)
    x = torch.sin(2.0 * math.pi * 440.0 * t)

    analyzer = GammatoneAnalyzerTorch(fs, 80.0, 1000.0, 6000.0, 1.0, device, torch.float64)
    subbands = analyzer.process(x)

    assert subbands.device == device
    assert subbands.shape[1] == 16000

    synth = GammatoneSynthesizerTorch(analyzer, 0.004)
    reconstructed = synth.process(subbands)

    assert reconstructed.device == device
    assert reconstructed.shape[0] == 16000

    # Convert to numpy and check that reconstruction is highly correlated (>0.90)
    x_np = to_numpy_format(x)
    recon_np = to_numpy_format(reconstructed)

    # Account for synthesizer delay offset (0.004 seconds * 16000 Hz = 64 samples)
    delay = int(round(0.004 * fs))
    orig_slice = x_np[delay: -delay]
    recon_slice = recon_np[2 * delay: len(orig_slice) + 2 * delay]

    corr = np.corrcoef(orig_slice, recon_slice)[0, 1]
    assert corr > 0.90, f"PyTorch reconstruction fidelity failed. Correlation is {corr:.4f}"