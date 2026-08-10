import math

import pytest
import torch

from peass import DecompositionConfiguration
from peass.backend_torch.decomposition import apply_window_shading_torch
from peass.backend_torch.decomposition import decompose_distortion_components
from peass.backend_torch.decomposition import matlab_shade_length
from peass.backend_torch.decomposition import matlab_shade_window_torch

# (fs, shade_ms) pairs exercised by the MATLAB-fidelity shade window tests. The
# 44100/5.0, 44100/25.0 and 22050/10.0 entries land exactly on a .5 sample count,
# which pins MATLAB's round-half-away-from-zero tie rule.
SHADE_WINDOW_CASES = [
    (8000.0, 10.0), (8000.0, 5.0), (8000.0, 25.0),
    (16000.0, 10.0), (16000.0, 5.0), (16000.0, 25.0),
    (22050.0, 10.0),
    (44100.0, 10.0), (44100.0, 5.0), (44100.0, 25.0),
    (48000.0, 10.0), (48000.0, 5.0), (48000.0, 25.0),
    (8000.0, 0.125), (16000.0, 0.2),
]


def matlab_shade_window_reference(fs: float, shade_ms: float) -> torch.Tensor:
    """
    Literal transcription of MATLAB `extractDistortionComponents.m` (v2.0.1)::

        wShadeIn = hann(2*round(shadeInMs/1000*fs+1),'periodic');
        wShadeIn = wShadeIn(2:end/2);

    with ``hann(N,'periodic')[k] = 0.5*(1 - cos(2*pi*(k-1)/N))`` for 1-based k=1..N and
    MATLAB's round-half-away-from-zero. Independent oracle for the production closed form.
    """
    n_total = 2 * int(math.floor(shade_ms / 1000.0 * fs + 1.0 + 0.5))
    k = torch.arange(1, n_total + 1, dtype=torch.float64)  # 1-based MATLAB index
    full_hann = 0.5 * (1.0 - torch.cos(2.0 * math.pi * (k - 1) / n_total))
    # MATLAB w(2:end/2) keeps 1-based k = 2 .. N/2 -> 0-based 1 .. N//2 - 1
    return full_hann[1:n_total // 2]


@pytest.mark.unit
@pytest.mark.parametrize("fs, shade_ms", SHADE_WINDOW_CASES)
def test_torch_shade_window_matches_matlab_hann_slice(fs, shade_ms):
    """
    The torch shade window must reproduce MATLAB's
    `hann(2*round(ms/1000*fs+1),'periodic')` sliced with `(2:end/2)` exactly.
    """
    reference_window = matlab_shade_window_reference(fs, shade_ms)

    fade_samples = matlab_shade_length(shade_ms, fs)
    assert fade_samples == reference_window.numel()

    produced_window = matlab_shade_window_torch(fade_samples, torch.device("cpu"), torch.float64)
    torch.testing.assert_close(produced_window, reference_window, rtol=0.0, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("fs, shade_ms", SHADE_WINDOW_CASES)
def test_torch_shade_window_is_strict_hann_interior(fs, shade_ms):
    """
    MATLAB drops both the leading zero and the unity midpoint of the Hann rise, so the
    window is strictly inside (0, 1). A plain 0->1 ramp would violate this.
    """
    window = matlab_shade_window_torch(
        matlab_shade_length(shade_ms, fs), torch.device("cpu"), torch.float64
    )

    assert window.numel() > 0
    assert bool(torch.all(window > 0.0))
    assert bool(torch.all(window < 1.0))
    assert bool(torch.all(torch.diff(window) > 0.0))


@pytest.mark.unit
def test_torch_shade_out_window_is_exact_reverse_of_shade_in():
    """MATLAB derives wShadeOut from wShadeIn via flipud(), so the two must mirror exactly."""
    fs = 44100.0
    num_samples = 4410
    signal = torch.ones(num_samples, 2, dtype=torch.float64)
    shaded = apply_window_shading_torch(signal, fs, 10.0, 10.0)

    fade_samples = matlab_shade_length(10.0, fs)
    shade_in_applied = shaded[:fade_samples, 0]
    shade_out_applied = shaded[-fade_samples:, 0]

    torch.testing.assert_close(shade_out_applied, shade_in_applied.flip(0), rtol=0.0, atol=0.0)
    assert bool(torch.all(shade_in_applied > 0.0))
    assert bool(torch.all(shade_in_applied < 1.0))


@pytest.mark.unit
def test_torch_shade_window_matches_numpy_backend():
    """Both backends must produce bit-comparable shade windows for the same (fs, ms)."""
    from peass.backend_numpy.decomposition import matlab_shade_length as np_shade_length
    from peass.backend_numpy.decomposition import matlab_shade_window as np_shade_window

    for fs, shade_ms in SHADE_WINDOW_CASES:
        fade_samples = matlab_shade_length(shade_ms, fs)
        assert fade_samples == np_shade_length(shade_ms, fs)

        torch_window = matlab_shade_window_torch(fade_samples, torch.device("cpu"), torch.float64)
        numpy_window = torch.from_numpy(np_shade_window(fade_samples))
        torch.testing.assert_close(torch_window, numpy_window, rtol=0.0, atol=1e-15)


@pytest.mark.unit
def test_torch_hermitian_solve_falls_back_on_rank_deficient_frames():
    """The Cholesky solver must reproduce `pinv` on frames it cannot factorize.

    `_solve_hermitian_batch` replaced an unconditional `torch.linalg.pinv` with
    `cholesky_ex` plus a per-matrix fallback. Silent frames make the Gram identically
    zero and genuinely rank-deficient ones make it merely singular, so both have to
    keep resolving to the same minimum-norm solution `pinv` gave -- and, just as
    importantly, the batched factorization must not leak inf/NaN out of the frames it
    failed on. Nothing else in the suite exercises the failing branch directly.
    """
    from peass.backend_torch.decomposition import _solve_hermitian_batch

    generator = torch.Generator().manual_seed(0)
    size, batch = 9, 6
    factor = torch.randn(batch, size, size, dtype=torch.float64, generator=generator)
    gram = factor @ factor.transpose(1, 2) + size * torch.eye(size, dtype=torch.float64)
    gram[1] = 0.0  # silent frame
    gram[3] = gram[3][:, :1] @ gram[3][:1, :]  # rank 1, not positive definite
    rhs = torch.randn(batch, size, 2, dtype=torch.float64, generator=generator)

    _, info = torch.linalg.cholesky_ex(gram)
    assert bool(info.any()), "test no longer exercises the fallback branch"

    solved = _solve_hermitian_batch(gram, rhs)
    assert bool(torch.isfinite(solved).all())
    torch.testing.assert_close(solved, torch.linalg.pinv(gram) @ rhs, rtol=0.0, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("device_str", ["cpu", "cuda", "mps"])
def test_torch_decomposition_algebraic_reconstruction(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if device_str == "mps":
        pytest.skip("MPS does not support float64; the PEASS torch backend is float64-only.")
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
    result = decompose_distortion_components(source_files=[target, interferer], estimate_file=estimate,
                                             configuration=config, sampling_frequency_hz=fs)
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
        decompose_distortion_components(source_files=[target, interferer], estimate_file=estimate_mismatched,
                                        sampling_frequency_hz=fs)

    with pytest.raises(ValueError, match="requires explicit sampling rate"):
        decompose_distortion_components(source_files=[target, interferer], estimate_file=target,
                                        sampling_frequency_hz=None)


@pytest.mark.unit
@pytest.mark.parametrize("device_str", ["cpu", "cuda", "mps"])
def test_torch_decomposition_gain_invariance(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if device_str == "mps":
        pytest.skip("MPS does not support float64; the PEASS torch backend is float64-only.")
    device = torch.device(device_str)

    fs = 16000.0
    t = torch.linspace(0.0, 0.5, 8000, device=device, dtype=torch.float64)
    target = torch.sin(2.0 * math.pi * 440.0 * t).unsqueeze(1)
    silent_interferer = torch.zeros_like(target)

    # 30% amplitude reduction
    estimate = 0.7 * target

    result = decompose_distortion_components(source_files=[target, silent_interferer], estimate_file=estimate,
                                             sampling_frequency_hz=fs)
    wf = result.waveforms

    # 7. Extract the time-domain artifacts and calculate peak absolute value
    peak_artifacts = torch.max(torch.abs(wf.artifacts)).item()

    # 8. Calculate the corresponding gain-invariance internal error:
    # max(|target_distortion - (-0.3 * true_target)|)
    gain_invariance_mismatch = torch.max(torch.abs(wf.target_distortion - (-0.3 * wf.true_target))).item()

    # 9. Format outputs to full float64 precision
    print("=" * 65)
    print("      DETERMINISTIC WAVEFORM RECONSTRUCTION LIMITS")
    print("=" * 65)
    print(f"  Peak Absolute Artifact Transient:    {peak_artifacts:.17f}")
    print(f"  Gain-Invariance Mismatch:            {gain_invariance_mismatch:.17f}")
    print("=" * 65)

    # A pure gain change should map entirely into target_distortion. The 2e-3
    # bound is the Gammatone analysis-synthesis reconstruction floor on a short
    # (0.5 s) signal, NOT a backend approximation: the NumPy reference hits the
    # same ~1.9e-3 error here (see the diagnostics test). The padding test below
    # drives this under 1e-4 with reflective boundary padding.
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

    # The gain-invariance error is the shared Gammatone reconstruction floor
    # (~1.9e-3), so both backends hit it; the meaningful check is that the two
    # backends AGREE closely and that torch is no worse than numpy at this limit.
    assert err_gain_th < 2e-3, f"PyTorch gain invariance error {err_gain_th:.6e} exceeds reconstruction limit."
    assert abs(err_gain_th - err_gain_np) < 1e-5, "torch and numpy reconstruction limits diverge"
    assert diff_true < 1e-4, f"true_target backend mismatch too large: {diff_true:.3e}"
    assert diff_dist < 1e-4, f"target_distortion backend mismatch too large: {diff_dist:.3e}"
    assert diff_interf < 1e-4, f"interference backend mismatch too large: {diff_interf:.3e}"
    assert diff_artif < 1e-4, f"artifacts backend mismatch too large: {diff_artif:.3e}"


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
    result = decompose_distortion_components(source_files=[padded_target, padded_interferer],
                                             estimate_file=padded_estimate, configuration=config,
                                             sampling_frequency_hz=sample_rate)
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
    torch.testing.assert_close(target_distortion, expected_distortion, rtol=2e-6, atol=2e-6)
    assert torch.max(torch.abs(interference)).item() < 2e-6
    assert torch.max(torch.abs(artifacts)).item() < 2e-6
