"""
PEASS PyTorch Auditory Nerve Model
File path: peass/backend_torch/auditory_model.py
"""
import math
import os
from functools import lru_cache

import torch
import torch.nn.functional as F

from .gammatone import GammatoneAnalyzerTorch
from .utils import fast_resample_poly_torch
from .utils import smoothmax
from ..config import ModulationProcessingType


@lru_cache(maxsize=8)
def _get_modulation_filters_fft(target_fs: float, N_fft: int, centers_tuple: tuple, bandwidths_tuple: tuple,
                                device_str: str):
    device = torch.device(device_str)
    num_mods = len(centers_tuple)
    H_all = torch.zeros((num_mods, N_fft), dtype=torch.complex128, device=device)

    decay_samples = int(target_fs * 0.5)
    time_indices = torch.arange(decay_samples, device=device, dtype=torch.float64)

    for m_idx in range(num_mods):
        g = math.exp(-math.pi * bandwidths_tuple[m_idx] / target_fs)
        b0 = 1.0 - g

        # Math Shortcut: Use vector exp instead of expensive array exponentiation `a1 ** t`
        log_g = math.log(g)
        phase = 2j * math.pi * centers_tuple[m_idx] / target_fs
        ir = b0 * torch.exp(time_indices * (log_g + phase))

        H_all[m_idx] = torch.fft.fft(ir, n=N_fft)

    return H_all


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
    transduced = F.conv1d(padded.view(B, 1, -1), ir.view(1, 1, -1)).view(B, T)

    return transduced.view(*orig_shape)


def _raw_adaptation_loop(subbands: torch.Tensor, fs: float, thresholds: torch.Tensor, gains: torch.Tensor,
                         abs_thresh: float) -> torch.Tensor:
    orig_shape = subbands.shape
    T = orig_shape[-1]
    subbands_flat = subbands.view(-1, T)
    B = subbands_flat.shape[0]

    adapted = torch.empty_like(subbands_flat)
    states = thresholds.unsqueeze(0).expand(B, 5).clone()

    for t in range(T):
        val = smoothmax(subbands_flat[:, t], abs_thresh)
        for stage in range(5):
            g = gains[stage]
            th = thresholds[stage]
            st = states[:, stage]

            val_compressed = val / st
            # Smoothmax preserves gradients
            new_st = smoothmax((1.0 - g) * val_compressed + g * st, th)

            states[:, stage] = new_st
            val = val_compressed

        adapted[:, t] = val

    return adapted.view(*orig_shape)


# Optimize compilation path: Bypass cold-start JIT compilations on CPU / Windows environments
_SHOULD_COMPILE = torch.cuda.is_available() and os.environ.get("PEASS_NO_COMPILE") != "1"

if _SHOULD_COMPILE:
    try:
        _compiled_adaptation_loop = torch.compile(_raw_adaptation_loop, mode="reduce-overhead", fullgraph=True)
    except Exception:
        _compiled_adaptation_loop = _raw_adaptation_loop
else:
    _compiled_adaptation_loop = _raw_adaptation_loop


def simulate_auditory_nerve_adaptation(subbands: torch.Tensor, fs: float) -> torch.Tensor:
    """Simulates the physiological adaptive properties of the auditory nerve."""
    abs_thresh = 10.0 ** (-100.0 / 20.0)
    bws = 1.0 / (math.pi * torch.tensor([0.005, 0.05, 0.129, 0.253, 0.5], device=subbands.device, dtype=subbands.dtype))

    gains = torch.exp(-math.pi * bws / fs)
    thresholds = torch.tensor([abs_thresh ** (0.5 ** i) for i in range(1, 6)], device=subbands.device,
                              dtype=subbands.dtype)

    adapted = _compiled_adaptation_loop(subbands, fs, thresholds, gains, abs_thresh)

    final_thresh = abs_thresh ** (0.5 ** 5)
    return (100.0 / (1.0 - final_thresh)) * (adapted - final_thresh)


def generate_auditory_internal_representation_torch(
        signal_data: torch.Tensor,
        fs: float,
        modulation_type: ModulationProcessingType = ModulationProcessingType.LOWPASS
) -> tuple[torch.Tensor, float]:
    """Generates the 3D internal auditory representation natively on targeted hardware."""
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

    # 1. Gammatone Analyzer stage
    analyzer = GammatoneAnalyzerTorch(fs, low_freq, 1000.0, high_freq, 1.0, signal_data.device, signal_data.dtype)
    subbands = analyzer.process(scaled).real

    # 2 & 3. IHC Transduction & Adaptation stage
    transduced = simulate_inner_haircell_transduction(subbands, fs)
    adapted = simulate_auditory_nerve_adaptation(transduced, fs)

    # 4. Decimation & Modulation Filtering step
    if modulation_type == ModulationProcessingType.LOWPASS:
        downsampled = fast_resample_poly_torch(adapted, 100, int(fs), axis=-1)
        target_fs = 100.0

        # Center frequencies and Bandwidths (Single 0 Hz lowpass filter)
        centers = torch.tensor([0.0], device=signal_data.device, dtype=signal_data.dtype)
        bandwidths = torch.tensor([15.92], device=signal_data.device, dtype=signal_data.dtype)
    else:
        downsampled = fast_resample_poly_torch(adapted, 800, int(fs), axis=-1)
        target_fs = 800.0

        # Original 8-band modulation filterbank centers and bandwidths [2.2]
        centers = torch.cat([
            torch.tensor([0.0, 5.0], device=signal_data.device, dtype=signal_data.dtype),
            10.0 * (5.0 / 3.0) ** torch.arange(6, device=signal_data.device, dtype=signal_data.dtype)
        ])
        bandwidths = torch.cat([
            torch.tensor([5.0, 5.0], device=signal_data.device, dtype=signal_data.dtype),
            5.0 * (5.0 / 3.0) ** torch.arange(6, device=signal_data.device, dtype=signal_data.dtype)
        ])

    # BULLETPROOF DIMENSION UNPACKING:
    if downsampled.dim() == 2:
        num_bands, num_samples = downsampled.shape
        B = 1
        downsampled = downsampled.unsqueeze(0)
    else:
        B, num_bands, num_samples = downsampled.shape

    num_mods = len(centers)

    # Evaluate first-order complex modulation filters via analytical impulse responses
    decay_samples = int(target_fs * 0.5)  # 500 ms decay window
    N_fft = 2 ** math.ceil(math.log2(num_samples + decay_samples))
    X = torch.fft.fft(downsampled.to(torch.complex128), n=N_fft, dim=-1)  # (B, Bands, N_fft)

    # Fetch cached frequency response of modulation filters
    H_all = _get_modulation_filters_fft(
        target_fs, N_fft, tuple(centers.tolist()), tuple(bandwidths.tolist()), str(signal_data.device)
    )

    # Perform a single massive parallel multiply-and-IFFT across all modulation frequencies
    # X shape: (B, Bands, N_fft, 1), H_all shape: (1, 1, N_fft, Mods)
    Y = X.unsqueeze(-1) * H_all.T.view(1, 1, N_fft, num_mods)

    # Run IFFT simultaneously over all bands and modulations
    y = torch.fft.ifft(Y, n=N_fft, dim=2)
    internal_representation = y[:, :, :num_samples, :]

    # Map complex envelopes to real magnitudes for frequencies above 10 Hz
    channels_to_magnitude = (centers > 10.0)
    if channels_to_magnitude.any():
        internal_representation[..., channels_to_magnitude] = torch.abs(
            internal_representation[..., channels_to_magnitude]).to(torch.complex128)
    if (~channels_to_magnitude).any():
        internal_representation[..., ~channels_to_magnitude] = internal_representation[
            ..., ~channels_to_magnitude].real.to(torch.complex128)

    out = internal_representation.real
    if is_1d:
        out = out.squeeze(0)
    return out, target_fs
