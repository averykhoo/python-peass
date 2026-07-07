"""
PEASS PyTorch Auditory Nerve Model
File path: peass/backend_torch/auditory_model.py
"""
import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from .gammatone import GammatoneAnalyzerTorch
from .utils import fast_resample_poly_torch
from ..config import ModulationProcessingType


@lru_cache(maxsize=4)
def _get_adaptation_constants(fs: float, device_str: str, dtype: torch.dtype):
    device = torch.device(device_str)
    abs_thresh = 10.0 ** (-100.0 / 20.0)
    bws = 1.0 / (math.pi * torch.tensor([0.005, 0.05, 0.129, 0.253, 0.5], device=device, dtype=dtype))
    gains = torch.exp(-math.pi * bws / fs)
    thresholds = torch.tensor([abs_thresh ** (0.5 ** i) for i in range(1, 6)], device=device, dtype=dtype)
    return abs_thresh, gains, thresholds


@lru_cache(maxsize=4)
def _get_modulation_constants(mod_type: str, device_str: str, dtype: torch.dtype):
    device = torch.device(device_str)
    if mod_type == "LOWPASS":
        centers = torch.tensor([0.0], device=device, dtype=dtype)
        bandwidths = torch.tensor([15.92], device=device, dtype=dtype)
    else:
        centers = torch.cat([
            torch.tensor([0.0, 5.0], device=device, dtype=dtype),
            10.0 * (5.0 / 3.0) ** torch.arange(6, device=device, dtype=dtype)
        ])
        bandwidths = torch.cat([
            torch.tensor([5.0, 5.0], device=device, dtype=dtype),
            5.0 * (5.0 / 3.0) ** torch.arange(6, device=device, dtype=dtype)
        ])
    return centers, bandwidths


def simulate_inner_haircell_transduction(subbands: torch.Tensor, fs: float) -> torch.Tensor:
    """Models the nonlinear mechanical-to-neural transduction of the inner hair cells."""
    # Half-wave Rectification
    rectified = F.relu(subbands)

    # 1 kHz Lowpass via causal FIR approximation for fast parallel execution
    gain = math.exp(-math.pi * 2000.0 / fs)
    b0 = 1.0 - gain

    # Pre-compute exact IIR impulse response to 60dB decay
    decay_samples = int(fs * 0.01)  # 10 ms decay threshold
    ir = b0 * (gain ** torch.arange(decay_samples, device=subbands.device, dtype=subbands.dtype))

    orig_shape = rectified.shape
    T = orig_shape[-1]
    rect_flat = rectified.view(-1, T)
    B = rect_flat.shape[0]

    padded = F.pad(rect_flat, (decay_samples - 1, 0))
    # F.conv1d is cross-correlation (no kernel flip), so the kernel must be
    # flipped to implement the causal FIR y[n] = sum_k x[n-k]*ir[k] that matches
    # the NumPy backend's lfilter([b0], [1, -gain]). Without the flip the filter
    # is applied time-reversed (verified: corr with the reference drops to ~0).
    transduced = F.conv1d(padded.view(B, 1, -1), ir.flip(-1).view(1, 1, -1)).view(B, T)

    return transduced.view(*orig_shape)


@torch.jit.script
def _raw_adaptation_loop(subbands_flat: torch.Tensor, thresholds: torch.Tensor, gains: torch.Tensor,
                         abs_thresh: float) -> torch.Tensor:
    """
    JIT-compiled loop that is fully autograd-compatible.
    Uses Lists and Stacks to ensure no gradients are broken via in-place mutation.
    """
    B, T = subbands_flat.shape
    states = thresholds.unsqueeze(0).expand(B, 5)

    outputs: list[torch.Tensor] = []

    for t in range(T):
        val = subbands_flat[:, t]
        val = torch.nn.functional.softplus(1000.0 * (val - abs_thresh)) / 1000.0 + abs_thresh

        new_states: list[torch.Tensor] = []
        for stage in range(5):
            g = gains[stage]
            th = thresholds[stage]
            st = states[:, stage]

            val_compressed = val / st
            new_st = torch.nn.functional.softplus(1000.0 * ((1.0 - g) * val_compressed + g * st - th)) / 1000.0 + th

            new_states.append(new_st)
            val = val_compressed

        states = torch.stack(new_states, dim=1)
        outputs.append(val)

    return torch.stack(outputs, dim=1)


def simulate_auditory_nerve_adaptation(subbands: torch.Tensor, fs: float) -> torch.Tensor:
    abs_thresh, gains, thresholds = _get_adaptation_constants(fs, str(subbands.device), subbands.dtype)

    orig_shape = subbands.shape
    T = orig_shape[-1]

    adapted = _raw_adaptation_loop(subbands.view(-1, T), thresholds, gains, abs_thresh)
    adapted = adapted.view(*orig_shape)

    final_thresh = abs_thresh ** (0.5 ** 5)
    return (100.0 / (1.0 - final_thresh)) * (adapted - final_thresh)


def generate_auditory_internal_representation_torch(
        signal_data: torch.Tensor, fs: float,
        modulation_type: ModulationProcessingType = ModulationProcessingType.LOWPASS
) -> tuple[torch.Tensor, float]:
    is_1d = signal_data.dim() == 1
    if is_1d:
        signal_data = signal_data.unsqueeze(0)

    scaled = 10.0 * signal_data
    low_freq = 235.0
    high_freq = min(0.5 * fs, 14500.0)

    original_fs = fs
    if fs / 2.0 < 1.5 * high_freq:
        new_fs = int(round(1.5 * fs))
        scaled = fast_resample_poly_torch(scaled, new_fs, int(fs), axis=-1)
        fs = float(new_fs)

    analyzer = GammatoneAnalyzerTorch(fs, low_freq, 1000.0, high_freq, 1.0, signal_data.device, signal_data.dtype)
    subbands = analyzer.process(scaled).real

    transduced = simulate_inner_haircell_transduction(subbands, fs)
    adapted = simulate_auditory_nerve_adaptation(transduced, fs)

    mod_str = "LOWPASS" if modulation_type == ModulationProcessingType.LOWPASS else "FILTERBANK"
    centers, bandwidths = _get_modulation_constants(mod_str, str(signal_data.device), signal_data.dtype)

    target_fs = 100.0 if mod_str == "LOWPASS" else 800.0
    down_factor = 100 if mod_str == "LOWPASS" else 800
    downsampled = fast_resample_poly_torch(adapted, down_factor, int(fs), axis=-1)

    if downsampled.dim() == 2:
        downsampled = downsampled.unsqueeze(0)

    B, num_bands, num_samples = downsampled.shape
    num_mods = len(centers)

    internal_representation = torch.zeros((B, num_bands, num_samples, num_mods), dtype=torch.complex128,
                                          device=signal_data.device)

    decay_samples = int(target_fs * 0.5)
    time_indices = torch.arange(decay_samples, device=signal_data.device, dtype=signal_data.dtype)

    N_fft = 2 ** math.ceil(math.log2(num_samples + decay_samples))
    X = torch.fft.fft(downsampled.to(torch.complex128), n=N_fft, dim=-1)

    for m_idx in range(num_mods):
        g = math.exp(-math.pi * bandwidths[m_idx].item() / target_fs)
        b0 = 1.0 - g
        a1 = g * torch.exp(2j * math.pi * centers[m_idx] / target_fs)
        ir = b0 * (a1 ** time_indices)
        H = torch.fft.fft(ir, n=N_fft)
        y = torch.fft.ifft(X * H.unsqueeze(0).unsqueeze(0), n=N_fft, dim=-1)
        internal_representation[..., m_idx] = y[..., :num_samples]

    # Out-of-place assignment to preserve gradients
    channels_to_magnitude = (centers > 10.0).view(1, 1, 1, -1)
    internal_representation = torch.where(
        channels_to_magnitude,
        torch.abs(internal_representation).to(torch.complex128),
        internal_representation.real.to(torch.complex128)
    )

    out = internal_representation.real
    if is_1d:
        out = out.squeeze(0)
    return out, target_fs
