"""
PEASS PyTorch Gammatone Filterbank
File path: peass/backend_torch/gammatone.py

Uses mass-parallel FFTs in place of sequential recursive time loops.
"""
import math

import torch


def calculate_erb(fc: torch.Tensor) -> torch.Tensor:
    return 24.7 * (0.00437 * fc + 1.0)


def get_erb_center_frequencies(filters_per_erb, low, center, high, device=None, dtype=None):
    # Only processes Python scalar floats at initialization
    def freq2erb(f): return 9.265 * math.log(1.0 + f / (24.7 * 9.265))

    # Processes multi-element PyTorch tensors; must use torch.exp
    def erb2freq(e):
        return (torch.exp(e / 9.265) - 1.0) * (24.7 * 9.265)

    e_low = freq2erb(low)
    e_center = freq2erb(center)
    e_high = freq2erb(high)

    num_below = math.floor((e_center - e_low) * filters_per_erb)
    start_erb = e_center - (num_below / filters_per_erb)

    erbs = torch.arange(start_erb, e_high + 1e-9, 1.0 / filters_per_erb, device=device, dtype=dtype)
    return erb2freq(erbs)


class GammatoneAnalyzerTorch:
    def __init__(self, fs, low, center, high, filters_per_erb, device, dtype):
        self.fs = fs
        self.center_frequencies = get_erb_center_frequencies(filters_per_erb, low, center, high, device, dtype)
        self.bandwidths = calculate_erb(self.center_frequencies)

        # Gammatone Constants
        gamma_const = (math.pi * math.factorial(6) * (2.0 ** -6) / (math.factorial(3) ** 2))
        decay_const = self.bandwidths / gamma_const

        lambda_decay = torch.exp(-2.0 * math.pi * decay_const / fs)
        phase_step = 2.0 * math.pi * self.center_frequencies / fs

        self.coefs = lambda_decay * torch.exp(1j * phase_step)
        self.norms = 2.0 * (1.0 - torch.abs(self.coefs)) ** 4

        # Calculate Decimation
        decimation_alpha = 2.0
        self.decimations = torch.clamp(torch.floor(fs / (self.bandwidths * decimation_alpha)), min=1).to(torch.int32)

    def process(self, x: torch.Tensor) -> torch.Tensor:
        """
        Massively parallel FFT evaluation of 4th-order complex IIR filters.
        x shape: (T)
        Returns: (num_bands, T)
        """
        T = x.shape[0]
        # Pad by 200ms to avoid circular convolution tail wrap-around
        pad_len = int(0.2 * self.fs)
        N_fft = 2 ** math.ceil(math.log2(T + pad_len))

        # Convert audio to freq domain
        X = torch.fft.fft(x.to(torch.complex128), n=N_fft)

        # Corrected: Use normalized cycles/sample directly to prevent frequency scale stretching
        freqs_norm = torch.fft.fftfreq(N_fft, device=x.device)
        z_inv = torch.exp(-2j * math.pi * freqs_norm)

        # Shape: (num_bands, N_fft)
        H = self.norms.view(-1, 1) / (1.0 - self.coefs.view(-1, 1) * z_inv.unsqueeze(0)) ** 4

        # Pointwise multiplication and back to time domain
        Y = X.unsqueeze(0) * H
        y = torch.fft.ifft(Y, n=N_fft, dim=1)

        # Slice valid region (linear convolution match)
        return y[:, :T]


class GammatoneSynthesizerTorch:
    def __init__(self, analyzer: GammatoneAnalyzerTorch, delay_sec: float):
        self.fs = analyzer.fs
        self.cfs = analyzer.center_frequencies

        target_delay = int(round(delay_sec * self.fs))
        impulse = torch.zeros(target_delay + 2, dtype=torch.complex128, device=self.cfs.device)
        impulse[0] = 1.0

        ir = analyzer.process(impulse)
        slice_dur = torch.abs(ir[:, :target_delay + 1])
        max_idx = torch.argmax(slice_dur, dim=1)

        self.delays = target_delay - max_idx

        # Frequency slope mapping
        prev_vals = torch.where(max_idx > 0, ir[torch.arange(ir.shape[0]), max_idx - 1],
                                torch.tensor(0j, device=ir.device))
        next_vals = ir[torch.arange(ir.shape[0]), max_idx + 1]
        slopes = next_vals - prev_vals
        slopes = slopes / (torch.abs(slopes) + 1e-15)
        self.phase_factors = 1j / slopes

        # Calculate mixing gains
        z = torch.exp(2j * math.pi * self.cfs / self.fs)
        self.gains = torch.ones_like(self.cfs)

        def z_response(z_pts):
            z_col = z_pts.unsqueeze(1)
            # Corrected attribute reference (analyzer.norms and analyzer.coefs)
            H = analyzer.norms.unsqueeze(0) / (1.0 - analyzer.coefs.unsqueeze(0) / z_col) ** 4
            return H

        # Fix column-wise scaling using unsqueeze(0) for phase factors
        pos = z_response(z) * self.phase_factors.unsqueeze(0) * (z.unsqueeze(1) ** -self.delays.unsqueeze(0))
        neg = z_response(torch.conj(z)) * self.phase_factors.unsqueeze(0) * (torch.conj(z).unsqueeze(1) ** -self.delays.unsqueeze(0))
        combo = (pos + torch.conj(neg)) / 2.0

        for _ in range(100):
            spec = combo @ self.gains.to(torch.complex128)
            self.gains = self.gains / (torch.abs(spec) + 1e-15).real

    def process(self, subbands: torch.Tensor) -> torch.Tensor:
        # Phase alignment and time shifting
        num_bands, num_samples = subbands.shape
        aligned = (subbands * self.phase_factors.view(-1, 1)).real

        out = torch.zeros_like(aligned)
        for b in range(num_bands):
            d = self.delays[b].item()
            if d == 0:
                out[b] = aligned[b]
            else:
                out[b, d:] = aligned[b, :-d]

        # Matrix mixing
        return self.gains @ out
