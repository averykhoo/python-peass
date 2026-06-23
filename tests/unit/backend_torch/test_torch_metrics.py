import numpy as np
import pytest
import torch

from peass.backend_torch.metrics import calculate_auditory_similarity_metric_torch
from peass.backend_torch.metrics import calculate_bss_eval_energy_ratios


@pytest.mark.unit
def test_torch_calculate_energy_ratios_analytical():
    """Tests energy ratios using hardcoded predictable tensor inputs."""
    true_source = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
    target_distortion = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float64)
    interference = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
    artifacts = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64)

    isr, sir, sar, sdr = calculate_bss_eval_energy_ratios(
        true_source, target_distortion, interference, artifacts
    )

    assert np.isclose(isr, 20.0)
    assert np.isclose(sir, 10.0 * np.log10(484.0))
    assert np.isclose(sar, 10.0 * np.log10(2116.0))
    assert np.isclose(sdr, 10.0 * np.log10(32.653061224))


@pytest.mark.unit
@pytest.mark.parametrize("device_str", ["cpu", "cuda", "mps"])
def test_torch_pemo_similarity_identity(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if device_str == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available.")
    device = torch.device(device_str)

    bands, samples, modulations = 4, 200, 1
    fs = 100.0

    ref_rep = torch.abs(torch.randn(bands, samples, modulations, device=device, dtype=torch.float64))
    test_rep = ref_rep.clone()

    sim = calculate_auditory_similarity_metric_torch(ref_rep, test_rep, fs)
    assert 0.9 <= sim <= 1.0


@pytest.mark.unit
@pytest.mark.parametrize("device_str", ["cpu", "cuda", "mps"])
def test_torch_perceptual_assimilation_behavior(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if device_str == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available.")
    device = torch.device(device_str)

    fs = 100.0
    num_bands, num_samples, num_modulations = 4, 200, 1

    # Generates non-zero variance vectors to prevent division-by-zero NaNs in standard Pearson correlation
    ref_rep = torch.abs(
        1.0 + 0.2 * torch.randn(num_bands, num_samples, num_modulations, device=device, dtype=torch.float64))
    # Noisy estimate scaled down
    test_rep = 0.5 * (ref_rep + 0.1 * torch.randn_like(ref_rep))

    # Standard Pearson Correlation
    lref = ref_rep.view(-1) - torch.mean(ref_rep)
    ltest = test_rep.view(-1) - torch.mean(test_rep)
    standard_corr = (torch.sum(lref * ltest) / torch.sqrt(torch.sum(lref ** 2) * torch.sum(ltest ** 2))).item()

    # Perceptual metric should select higher similarity score due to masking assimilation
    perceptual_metric = calculate_auditory_similarity_metric_torch(ref_rep, test_rep, fs)
    assert perceptual_metric > standard_corr
