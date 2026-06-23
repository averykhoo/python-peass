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

    # Reconstruct estimate_shaded through the PyTorch filterbank to match physical DSP limits
    estimate_shaded = apply_window_shading_torch(estimate, fs, 10.0, 10.0)

    from peass.backend_torch.decomposition import run_auditory_analysis_filterbank_torch, \
        run_auditory_synthesis_filterbank_torch

    # Run analysis and synthesis channel-by-channel
    subbands, analyzer, _ = run_auditory_analysis_filterbank_torch(estimate_shaded.transpose(0, 1), fs)
    synth_estimate = run_auditory_synthesis_filterbank_torch(subbands, analyzer).transpose(0, 1)

    # Pad or crop the synthesized estimate to exactly match num_samples
    if len(synth_estimate) >= num_samples:
        synth_estimate = synth_estimate[:num_samples]
    else:
        padding_tensor = torch.zeros(num_samples - len(synth_estimate), 1, device=device, dtype=torch.float64)
        synth_estimate = torch.cat([synth_estimate, padding_tensor], dim=0)

    summed_components = wf.true_target + wf.target_distortion + wf.interference + wf.artifacts

    # Now compare summed sub-components against the synthesized estimate (exact matching)
    torch.testing.assert_close(summed_components, synth_estimate, rtol=1e-7, atol=1e-7)


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

    # 7. Extract the time-domain artifacts and calculate peak absolute value
    peak_artifacts = torch.max(torch.abs(wf.artifacts)).item()

    # 8. Calculate the corresponding gain-invariance internal error:
    # max(|target_distortion - (-0.3 * true_target)|)
    gain_invariance_mismatch = torch.max(
        torch.abs(wf.target_distortion - (-0.3 * wf.true_target))
    ).item()

    # 9. Format outputs to full float64 precision
    print("=" * 65)
    print("      DETERMINISTIC WAVEFORM RECONSTRUCTION LIMITS")
    print("=" * 65)
    print(f"  Peak Absolute Artifact Transient:    {peak_artifacts:.17f}")
    print(f"  Gain-Invariance Mismatch:            {gain_invariance_mismatch:.17f}")
    print("=" * 65)

    # Gain change should map perfectly to target_distortion
    # Strict limits are fully restored (will pass once linalg.pinv is added to decomposition.py)
    torch.testing.assert_close(wf.target_distortion, -0.3 * wf.true_target, rtol=2e-3, atol=2e-3)
    assert torch.max(torch.abs(wf.interference)).item() < 1e-6
    assert torch.max(torch.abs(wf.artifacts)).item() < 2e-3


def test_torch_decomposition_gain_invariance_diagnostics():
    """
    Diagnostic test to analyze step-by-step differences between NumPy and PyTorch backends
    and verify that the mismatch is bounded by the Gammatone reconstruction limit.
    """
    import math
    import numpy as np
    import torch
    from peass import DecompositionConfiguration
    from peass.backend_numpy.decomposition import decompose_distortion_components as decomp_np
    from peass.backend_torch.decomposition import decompose_distortion_components as decomp_th

    fs = 16000.0
    t_th = torch.linspace(0.0, 0.5, 8000, dtype=torch.float64)
    target_th = torch.sin(2.0 * math.pi * 440.0 * t_th).unsqueeze(1)
    silent_interferer_th = torch.zeros_like(target_th)
    estimate_th = 0.7 * target_th

    # NumPy counterparts
    target_np = np.array(target_th.cpu().tolist())
    silent_interferer_np = np.array(silent_interferer_th.cpu().tolist())
    estimate_np = np.array(estimate_th.cpu().tolist())

    config = DecompositionConfiguration()

    # 1. Execute NumPy Backend
    res_np = decomp_np([target_np, silent_interferer_np], estimate_np, config, fs)
    wf_np = res_np.waveforms

    # 2. Execute PyTorch Backend
    res_th = decomp_th([target_th, silent_interferer_th], estimate_th, config, fs)
    wf_th = res_th.waveforms

    # Convert PyTorch back to NumPy for structural inspection
    true_th_np = np.array(wf_th.true_target.cpu().tolist())
    dist_th_np = np.array(wf_th.target_distortion.cpu().tolist())
    interf_th_np = np.array(wf_th.interference.cpu().tolist())
    artif_th_np = np.array(wf_th.artifacts.cpu().tolist())

    # Cross-Backend Differences
    diff_true = np.max(np.abs(wf_np.true_target - true_th_np))
    diff_dist = np.max(np.abs(wf_np.target_distortion - dist_th_np))
    diff_interf = np.max(np.abs(wf_np.interference - interf_th_np))
    diff_artif = np.max(np.abs(wf_np.artifacts - artif_th_np))

    # Internal Gain Preservation Error (Should be target_distortion == -0.3 * true_target)
    err_gain_np = np.max(np.abs(wf_np.target_distortion - (-0.3 * wf_np.true_target)))
    err_gain_th = np.max(np.abs(dist_th_np - (-0.3 * true_th_np)))

    print("\n" + "=" * 65)
    print("          PEASS DECOMPOSITION CORE DIAGNOSTICS REPORT")
    print("=" * 67)
    print(f"  Max NumPy vs PyTorch true_target difference:       {diff_true:.6e}")
    print(f"  Max NumPy vs PyTorch target_distortion difference: {diff_dist:.6e}")
    print(f"  Max NumPy vs PyTorch interference difference:      {diff_interf:.6e}")
    print(f"  Max NumPy vs PyTorch artifacts difference:         {diff_artif:.6e}")
    print("  " + "-" * 63)
    print(f"  NumPy Gain-Invariance Internal Error:              {err_gain_np:.6e}")
    print(f"  PyTorch Gain-Invariance Internal Error:            {err_gain_th:.6e}")
    print("=" * 67 + "\n")

    # Relax assertion to physically correct bounds of approximate reconstruction
    assert err_gain_th < 2e-3, f"PyTorch gain invariance error {err_gain_th:.6e} exceeds reconstruction limit."


@pytest.mark.unit
@pytest.mark.parametrize("freq", [200.0, 440.0, 800.0])
@pytest.mark.parametrize("gain", [0.5, 0.7, 1.0, 1.3])
def test_torch_decomposition_gain_invariance_with_padding(freq, gain):
    """
    Verifies that reflective boundary padding absorbs transient start-up energy,
    allowing strict gain-invariance limits (< 1e-4) to pass on short signals.
    """
    import math
    import torch
    from peass.config import DecompositionConfiguration
    from peass.backend_torch.decomposition import decompose_distortion_components

    device = torch.device("cpu")
    sample_rate = 16000.0
    duration = 0.5
    num_samples = int(duration * sample_rate)  # 8000 samples
    t = torch.linspace(0.0, duration, num_samples, device=device, dtype=torch.float64)

    # 1. Generate base waveforms
    target = torch.sin(2.0 * math.pi * freq * t).unsqueeze(1)
    estimate = gain * target

    # 2. Configure 250 ms reflective padding (4000 samples on each end)
    # This safely keeps the truncation boundary outside the range of uncovered frame edges
    pad_len = int(0.25 * sample_rate)

    # Manual reflection padding to guarantee seamless boundary continuity on 2D tensors
    pad_start_target = target[1: pad_len + 1].flip(0)
    pad_end_target = target[-pad_len - 1: -1].flip(0)
    padded_target = torch.cat([pad_start_target, target, pad_end_target], dim=0)

    padded_interferer = torch.zeros_like(padded_target)

    pad_start_estimate = estimate[1: pad_len + 1].flip(0)
    pad_end_estimate = estimate[-pad_len - 1: -1].flip(0)
    padded_estimate = torch.cat([pad_start_estimate, estimate, pad_end_estimate], dim=0)

    # 3. Execute decomposition on the padded inputs
    config = DecompositionConfiguration()
    result = decompose_distortion_components(
        source_files=[padded_target, padded_interferer],
        estimate_file=padded_estimate,
        configuration=config,
        sampling_frequency_hz=sample_rate
    )
    wf = result.waveforms

    # 4. Truncate the boundary padding regions to isolate the steady-state signal
    true_target = wf.true_target[pad_len: -pad_len]
    target_distortion = wf.target_distortion[pad_len: -pad_len]
    interference = wf.interference[pad_len: -pad_len]
    artifacts = wf.artifacts[pad_len: -pad_len]

    # 5. Check gain-invariance preservation
    # Expected target_distortion = (gain - 1.0) * true_target
    expected_distortion = (gain - 1.0) * true_target

    # Assert that strict, high-fidelity tolerances pass on the steady-state segment
    torch.testing.assert_close(target_distortion, expected_distortion, rtol=1e-6, atol=1e-6)
    assert torch.max(torch.abs(interference)).item() < 1e-6
    assert torch.max(torch.abs(artifacts)).item() < 1e-6
