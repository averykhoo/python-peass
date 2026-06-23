import math

import pytest
import torch

from peass.backend_torch.predictor import predict_perceptual_evaluation_scores


@pytest.mark.unit
@pytest.mark.parametrize("device_str", ["cpu", "cuda", "mps"])
def test_torch_predictor_score_range_constraints(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if device_str == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available.")
    device = torch.device(device_str)

    fs = 16000.0
    t = torch.linspace(0.0, 0.5, 8000, device=device, dtype=torch.float64)
    target = torch.sin(2.0 * math.pi * 440.0 * t).unsqueeze(1)
    interferer = torch.sin(2.0 * math.pi * 1000.0 * t).unsqueeze(1)

    # Run scores on degraded estimate (50% signal reduction + heavy leakage)
    estimate = 0.5 * target + 0.3 * interferer

    results = predict_perceptual_evaluation_scores(
        original_files=[target, interferer],
        estimate_file=estimate,
        sampling_frequency_hz=fs
    )

    assert 0.0 <= results.overall_perceptual_score <= 100.0
    assert 0.0 <= results.target_perceptual_score <= 100.0
    assert 0.0 <= results.interference_perceptual_score <= 100.0
    assert 0.0 <= results.artifact_perceptual_score <= 100.0


@pytest.mark.unit
@pytest.mark.parametrize("device_str", ["cpu", "cuda", "mps"])
def test_torch_predictor_pristine_audio_conditions(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if device_str == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available.")
    device = torch.device(device_str)

    fs = 16000.0
    t = torch.linspace(0.0, 0.5, 8000, device=device, dtype=torch.float64)
    target = torch.sin(2.0 * math.pi * 300.0 * t).unsqueeze(1)
    noise = torch.zeros_like(target)

    # Perfect estimate
    estimate = target.clone()

    results = predict_perceptual_evaluation_scores(
        original_files=[target, noise],
        estimate_file=estimate,
        sampling_frequency_hz=fs
    )

    assert results.overall_perceptual_score > 90.0
    assert results.target_perceptual_score > 90.0
    assert results.interference_perceptual_score > 90.0


@pytest.mark.unit
@pytest.mark.parametrize("device_str", ["cpu", "cuda", "mps"])
def test_torch_predictor_amplitude_robustness(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if device_str == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available.")
    device = torch.device(device_str)

    fs = 16000.0
    t = torch.linspace(0.0, 0.5, 8000, device=device, dtype=torch.float64)
    target = torch.sin(2.0 * math.pi * 440.0 * t).unsqueeze(1)
    interferer = torch.sin(2.0 * math.pi * 1000.0 * t).unsqueeze(1)
    estimate = target + 0.1 * interferer

    # Test extremely low amplitude limits to verify numerical stability
    scaled_target = target * 1e-5
    scaled_interferer = interferer * 1e-5
    scaled_estimate = estimate * 1e-5

    results = predict_perceptual_evaluation_scores(
        original_files=[scaled_target, scaled_interferer],
        estimate_file=scaled_estimate,
        sampling_frequency_hz=fs
    )

    assert not torch.isnan(results.overall_perceptual_score).any()
    assert not torch.isinf(results.overall_perceptual_score).any()


@pytest.mark.unit
def test_torch_predictor_silent_edge_cases():
    fs = 16000.0
    silent_target = torch.zeros((4000, 1), dtype=torch.float64)
    silent_noise = torch.zeros_like(silent_target)
    silent_estimate = torch.zeros_like(silent_target)

    scores = predict_perceptual_evaluation_scores(
        original_files=[silent_target, silent_noise],
        estimate_file=silent_estimate,
        sampling_frequency_hz=fs
    )

    assert 0.0 <= scores.overall_perceptual_score <= 100.0
    assert 0.0 <= scores.target_perceptual_score <= 100.0
