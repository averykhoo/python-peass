import math

import pytest
import torch

from peass import DecompositionConfiguration
from peass.backend_torch.decomposition import apply_window_shading_torch
from peass.backend_torch.decomposition import decompose_distortion_components


@pytest.mark.unit
@pytest.mark.parametrize("device_str", ["cpu", "cuda", "mps"])
def test_torch_decomposition_algebraic_reconstruction(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if device_str == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available.")
    device = torch.device(device_str)

    fs = 16000.0
    duration = 0.5
    num_samples = int(duration * fs)
    t = torch.linspace(0.0, duration, num_samples, device=device, dtype=torch.float64)

    # Generate target, interferer, and estimate signals
    target = torch.sin(2.0 * math.pi * 440.0 * t).unsqueeze(1)
    interferer = torch.sin(2.0 * math.pi * 1000.0 * t).unsqueeze(1)
    estimate = target + 0.1 * interferer

    config = DecompositionConfiguration()
    result = decompose_distortion_components(
        source_files=[target, interferer],
        estimate_file=estimate,
        configuration=config,
        sampling_frequency_hz=fs
    )
    wf = result.waveforms

    # Verify matching dimensions
    assert wf.true_target.shape == estimate.shape

    # The sum of sub-components must mathematically reconstruct the shaded estimate signal
    estimate_shaded = apply_window_shading_torch(estimate, fs, 10.0, 10.0)
    summed_components = wf.true_target + wf.target_distortion + wf.interference + wf.artifacts

    torch.testing.assert_close(summed_components, estimate_shaded, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_torch_decomposition_input_validation():
    fs = 16000.0
    target = torch.randn(1000, 1, dtype=torch.float64)
    interferer = torch.randn(1000, 1, dtype=torch.float64)
    estimate_mismatched = torch.randn(1010, 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="dimensions|size"):
        decompose_distortion_components(
            source_files=[target, interferer],
            estimate_file=estimate_mismatched,
            sampling_frequency_hz=fs
        )

    with pytest.raises(ValueError, match="requires explicit sampling rate"):
        decompose_distortion_components(
            source_files=[target, interferer],
            estimate_file=target,
            sampling_frequency_hz=None
        )


@pytest.mark.unit
@pytest.mark.parametrize("device_str", ["cpu", "cuda", "mps"])
def test_torch_decomposition_gain_invariance(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if device_str == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available.")
    device = torch.device(device_str)

    fs = 16000.0
    t = torch.linspace(0.0, 0.5, 8000, device=device, dtype=torch.float64)
    target = torch.sin(2.0 * math.pi * 440.0 * t).unsqueeze(1)
    silent_interferer = torch.zeros_like(target)

    # 30% amplitude reduction
    estimate = 0.7 * target

    result = decompose_distortion_components(
        source_files=[target, silent_interferer],
        estimate_file=estimate,
        sampling_frequency_hz=fs
    )
    wf = result.waveforms

    # Gain change should map perfectly to target_distortion
    torch.testing.assert_close(wf.target_distortion, -0.3 * wf.true_target, rtol=1e-5, atol=1e-5)
    assert torch.max(torch.abs(wf.interference)).item() < 1e-4
    assert torch.max(torch.abs(wf.artifacts)).item() < 1e-4
