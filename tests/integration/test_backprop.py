import math

import pytest

from peass import DecompositionConfiguration
from peass import decompose_distortion_components


@pytest.mark.integration
def test_pytorch_backpropagation_through_decomposition():
    """
    Verifies that gradients can flow backward through the entire PyTorch
    decomposition pipeline (Analysis Filterbank -> Least-Squares Projection -> Synthesis Filterbank).
    This is crucial for using the decomposition components as a neural network loss function.
    """
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch is not installed in this environment.")

    sampling_frequency = 16000.0
    duration_seconds = 0.5
    num_samples = int(duration_seconds * sampling_frequency)

    # We use double precision to match the regression test suites
    device = torch.device("cpu")
    time_steps = torch.linspace(0.0, duration_seconds, num_samples, device=device, dtype=torch.float64)

    # 1. Create source signals (Target + Interferer) as standard constants (no gradients needed)
    target = torch.sin(2.0 * math.pi * 440.0 * time_steps).unsqueeze(1)
    interferer = torch.sin(2.0 * math.pi * 1000.0 * time_steps).unsqueeze(1)

    # 2. Create the estimate waveform with requires_grad=True (simulating a denoiser network output)
    estimate = (target + 0.15 * interferer + 0.01 * torch.randn_like(target)).clone().detach().requires_grad_(True)

    # 3. Run PyTorch decomposition
    config = DecompositionConfiguration(
        shade_in_milliseconds=10.0,
        shade_out_milliseconds=10.0,
        use_two_stage_projection=False
    )
    result = decompose_distortion_components(
        source_files=[target, interferer],
        estimate_file=estimate,
        configuration=config,
        sampling_frequency_hz=sampling_frequency
    )
    waveforms = result.waveforms

    # 4. Define a loss function on the decomposed waveforms (e.g. minimizing target distortion and artifacts)
    loss = torch.mean(waveforms.target_distortion ** 2) + torch.mean(waveforms.artifacts ** 2)

    # 5. Execute backward pass
    loss.backward()

    # 6. Assertions to verify the autograd graph was preserved
    assert estimate.grad is not None, "Autograd graph disconnected! Gradients did not flow back to the estimate tensor."
    assert not torch.isnan(estimate.grad).any(), "Gradient calculations produced NaN values."
    assert not torch.isinf(estimate.grad).any(), "Gradient calculations produced infinite values."

    # Ensure that the gradient contains actual active values (not just zeros)
    max_gradient = torch.max(torch.abs(estimate.grad)).item()
    assert max_gradient > 0.0, "Gradients are zero, meaning the graph is dead."

    print(f"\n---> BACKPROP SUCCESS. Max gradient on input estimate: {max_gradient:.6e}")
