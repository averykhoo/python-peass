"""
PEASS Test Suite - Pipeline Integration Tests
File path: tests/test_integration.py
"""

import numpy as np

from peass.decomposition import extract_distortion_components
from peass.metrics import audio_quality_features
from peass.metrics import calculate_energy_ratios
from peass.predictor import predict_peass_scores


def test_end_to_end_stereo_separation():
    """
    Integration test using multi-channel (Stereo) inputs.
    Verifies that the entire PEASS pipeline (decomposition, metrics, and prediction)
    runs with 2-channel stereo arrays, handling the collapsed worst-channel scores correctly.
    """
    sampling_frequency = 16000.0
    duration_seconds = 1.0
    num_samples = int(duration_seconds * sampling_frequency)
    time_steps = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)

    # 1. Create stereo clean target (different frequencies for Left & Right channels)
    target_left = np.sin(2.0 * np.pi * 300.0 * time_steps)
    target_right = np.sin(2.0 * np.pi * 400.0 * time_steps)
    target = np.stack([target_left, target_right], axis=1)  # Shape (16000, 2)

    # 2. Create stereo interference
    interf_left = np.sin(2.0 * np.pi * 1000.0 * time_steps)
    interf_right = np.sin(2.0 * np.pi * 1200.0 * time_steps)
    interferer = np.stack([interf_left, interf_right], axis=1)

    # 3. Create estimate with slight leakage and artifact noise
    estimate = target + 0.1 * interferer + 0.02 * np.random.randn(num_samples, 2)

    # 4. Run end-to-end prediction
    results = predict_peass_scores(
        original_files=[target, interferer],
        estimate_file=estimate,
        sampling_frequency=sampling_frequency,
        return_decomposition=True
    )

    # Verify scores are generated and stay bounded
    for key in ["OPS", "TPS", "IPS", "APS"]:
        assert 0.0 <= results[key] <= 100.0

    # Ensure returned decomposition arrays are present and have stereo dimensions
    decomp_arrays = results["decomposition_arrays"]
    assert decomp_arrays["true_target"].shape == (num_samples, 2)
    assert decomp_arrays["target_distortion"].shape == (num_samples, 2)
    assert decomp_arrays["interference"].shape == (num_samples, 2)
    assert decomp_arrays["artifacts"].shape == (num_samples, 2)


def test_silent_reference_robustness():
    """
    Tests edge-case where one of the interfering sources is completely silent (all zeros).
    Verifies that the decomposition and scoring models do not crash or trigger division-by-zero errors.
    """
    sampling_frequency = 16000.0
    duration_seconds = 0.5
    num_samples = int(duration_seconds * sampling_frequency)
    time_steps = np.linspace(0, duration_seconds, num_samples)

    target = np.sin(2.0 * np.pi * 440.0 * time_steps)[:, np.newaxis]
    silent_noise = np.zeros_like(target)  # Completely silent interferer
    estimate = target + 0.01 * np.random.randn(num_samples, 1)

    # Execute full scoring run
    results = predict_peass_scores(
        original_files=[target, silent_noise],
        estimate_file=estimate,
        sampling_frequency=sampling_frequency
    )

    assert "OPS" in results
    assert 0.0 <= results["OPS"] <= 100.0


def test_sequential_component_execution_pipeline():
    """
    Integration test directly chaining manual outputs of each component.
    Decomposition -> Metrics -> Neural Net Scoring.
    """
    sampling_frequency = 16000.0
    duration_seconds = 1.0
    num_samples = int(duration_seconds * sampling_frequency)
    time_steps = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)

    target = np.sin(2.0 * np.pi * 350.0 * time_steps)[:, np.newaxis]
    noise = np.sin(2.0 * np.pi * 950.0 * time_steps)[:, np.newaxis]
    estimate = target + 0.05 * noise

    # Step 1: Decomposition
    _, decomposed_arrays = extract_distortion_components(
        src_files=[target, noise],
        est_file=estimate,
        sampling_frequency=sampling_frequency
    )
    s_true, e_target, e_interf, e_artif = decomposed_arrays

    # Step 2: Energy ratios
    ISR, SIR, SAR, SDR = calculate_energy_ratios(s_true, e_target, e_interf, e_artif)
    assert ISR > 0
    assert SIR > 0

    # Step 3: Auditory Quality Features
    q_target, q_interf, q_artif, q_global = audio_quality_features(decomposed_arrays, sampling_frequency)
    assert -1.0 <= q_target <= 1.0
    assert -1.0 <= q_interf <= 1.0
    assert -1.0 <= q_artif <= 1.0
    assert -1.0 <= q_global <= 1.0
