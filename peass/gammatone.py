"""
PEASS Auditory Package - Hohmann 2002 Gammatone Filterbank [1, 3]

This module implements the complex-valued Gammatone Filterbank as described in [3].
It provides complete physical modeling of frequency analysis, delay/phase alignment,
and synthesize capabilities to reconstruct fullband audio from subbands.
"""

from typing import List

import numpy as np
import scipy.signal as signal


def calculate_erb_bandwidth(center_frequency: float) -> float:
    """
    Computes the Equivalent Rectangular Bandwidth of auditory filters.

    Formula defined in Eq. (13) of [3]:
    % bw = 24.7*(.00437*fc+1);
    """
    return 24.7 * (0.00437 * center_frequency + 1.0)


def frequency_to_erb_scale(frequency_hz: float) -> float:
    """
    Converts frequency in Hz to Equivalent Rectangular Bandwidth (ERB) scale.

    Formula defined in Eq. (16) of [3]:
    % ERBscale = GFB_Q * log(1 + Hz / (GFB_L * GFB_Q));
    """
    return 9.265 * np.log(1.0 + frequency_hz / (24.7 * 9.265))


def erb_scale_to_frequency(erb_scale: float) -> float:
    """
    Converts Equivalent Rectangular Bandwidth (ERB) scale value to frequency in Hz.

    Formula defined in Eq. (17) of [3]:
    % Hz = (exp(ERBscale / GFB_Q) - 1) * (GFB_L * GFB_Q);
    """
    return (np.exp(erb_scale / 9.265) - 1.0) * (24.7 * 9.265)


def get_center_frequencies(
        filters_per_erb: float,
        lower_cutoff_hz: float,
        specified_center_hz: float,
        upper_cutoff_hz: float
) -> np.ndarray:
    """
    Constructs a vector of center frequencies equidistant on the ERB scale.
    Equivalent to Gfb_center_frequencies.m [3].
    """
    lower_erb = frequency_to_erb_scale(lower_cutoff_hz)
    specified_erb = frequency_to_erb_scale(specified_center_hz)
    upper_erb = frequency_to_erb_scale(upper_cutoff_hz)

    erbs_below_base = specified_erb - lower_erb
    num_filters_below = int(np.floor(erbs_below_base * filters_per_erb))

    start_erb = specified_erb - (num_filters_below / filters_per_erb)
    center_erbs = np.arange(start_erb, upper_erb + 1e-9, 1.0 / filters_per_erb)
    return erb_scale_to_frequency(center_erbs)


class GammatoneFilter:
    """
    Represents a single 4th-order complex-valued all-pole Gammatone filter.
    Equivalent to Gfb_Filter class in MATLAB [3].
    """

    def __init__(
            self,
            sampling_frequency: float,
            center_frequency: float,
            gamma_order: int = 4,
            bandwidth_factor: float = 1.0
    ):
        self.gamma_order: int = gamma_order
        self.sampling_frequency: float = sampling_frequency
        self.center_frequency: float = center_frequency

        # Auditory bandwidth scaling (Eq. 14 of [3])
        audiological_erb = calculate_erb_bandwidth(center_frequency) * bandwidth_factor
        a_gamma = (np.pi * math_factorial(2 * gamma_order - 2) * (2.0 ** -(2 * gamma_order - 2)) /
                   (math_factorial(gamma_order - 1) ** 2))
        b = audiological_erb / a_gamma

        self.lambda_val: float = np.exp(-2.0 * np.pi * b / sampling_frequency)
        self.beta: float = 2.0 * np.pi * center_frequency / sampling_frequency

        # Complex pole coefficient
        self.coefficient: complex = self.lambda_val * np.exp(1j * self.beta)
        self.normalization_factor: float = 2.0 * (1.0 - np.abs(self.coefficient)) ** gamma_order
        self.state: np.ndarray = np.zeros(gamma_order, dtype=complex)

    def process(self, input_signal: np.ndarray) -> np.ndarray:
        factor = self.normalization_factor
        coeff = self.coefficient
        filter_state = self.state * coeff

        y = input_signal.copy()
        b_stage = np.array([factor], dtype=complex)
        a_stage = np.array([1.0, -coeff], dtype=complex)

        new_state = np.zeros(self.gamma_order, dtype=complex)
        for i in range(self.gamma_order):
            b_coef = b_stage if i == 0 else np.array([1.0], dtype=complex)
            zi = np.array([filter_state[i]], dtype=complex)
            y, zf = signal.lfilter(b_coef, a_stage, y, zi=zi)
            new_state[i] = zf[0]

        self.state = new_state / coeff
        return y

    def clear_state(self) -> None:
        """Resets the internal filter state to zeros."""
        self.state = np.zeros(self.gamma_order, dtype=complex)


class GammatoneAnalyzer:
    """
    A collection of GammatoneFilters acting as an analysis filterbank.
    Equivalent to Gfb_Analyzer class in MATLAB [3].
    """

    def __init__(
            self,
            sampling_frequency: float,
            lower_cutoff_hz: float,
            specified_center_hz: float,
            upper_cutoff_hz: float,
            filters_per_erb: float,
            gamma_order: int = 4,
            bandwidth_factor: float = 1.0
    ):
        self.sampling_frequency: float = sampling_frequency
        self.center_frequencies: np.ndarray = get_center_frequencies(
            filters_per_erb, lower_cutoff_hz, specified_center_hz, upper_cutoff_hz
        )
        self.filters: List[GammatoneFilter] = [
            GammatoneFilter(sampling_frequency, cf, gamma_order, bandwidth_factor)
            for cf in self.center_frequencies
        ]
        self.bandwidths: np.ndarray = calculate_erb_bandwidth(self.center_frequencies)

    def process(self, input_signal: np.ndarray) -> np.ndarray:
        num_bands = len(self.filters)
        output = np.zeros((num_bands, input_signal.shape[0]), dtype=complex)
        for band in range(num_bands):
            output[band, :] = self.filters[band].process(input_signal)
        return output

    def get_z_response(self, z: np.ndarray) -> np.ndarray:
        z_col = z[:, np.newaxis]
        num_bands = len(self.filters)
        response = np.ones((z_col.shape[0], num_bands), dtype=complex)
        for band in range(num_bands):
            coeff = self.filters[band].coefficient
            norm = self.filters[band].normalization_factor
            gamma = self.filters[band].gamma_order
            response[:, band] = ((1.0 - coeff / z_col[:, 0]) ** -gamma) * norm
        return response

    def clear_state(self) -> None:
        """Resets all filters' states to zeros."""
        for filter_obj in self.filters:
            filter_obj.clear_state()


class GammatoneDelay:
    """
    Handles phase alignment and group delay estimation across subbands.
    Equivalent to Gfb_Delay class in MATLAB [3].
    """

    def __init__(self, analyzer: GammatoneAnalyzer, delay_samples: int):
        # Reset the analyzer states before analyzing the impulse response
        analyzer.clear_state()

        impulse = np.zeros(delay_samples + 2)
        impulse[0] = 1.0

        # Analyze impulse
        impulse_response = analyzer.process(impulse)
        num_bands = impulse_response.shape[0]

        ir_slice = np.abs(impulse_response[:, :delay_samples + 1])
        max_indices = np.argmax(ir_slice, axis=1)

        self.delays_samples: np.ndarray = delay_samples - max_indices
        slopes = np.zeros(num_bands, dtype=complex)
        for band in range(num_bands):
            idx = max_indices[band]
            slopes[band] = impulse_response[band, idx + 1] - impulse_response[band, idx - 1]

        slopes = slopes / np.abs(slopes)
        self.phase_factors: np.ndarray = 1j / slopes
        self.memory: np.ndarray = np.zeros((num_bands, int(np.max(self.delays_samples))), dtype=float)

    def process(self, input_data: np.ndarray) -> np.ndarray:
        num_bands, num_samples = input_data.shape
        output = np.zeros((num_bands, num_samples))
        for band in range(num_bands):
            delay_val = int(self.delays_samples[band])
            phase_corrected = np.real(input_data[band, :] * self.phase_factors[band])
            if delay_val == 0:
                output[band, :] = phase_corrected
            else:
                tmp_out = np.concatenate((self.memory[band, :delay_val], phase_corrected))
                self.memory[band, :delay_val] = tmp_out[num_samples:]
                output[band, :] = tmp_out[:num_samples]
        return output


class GammatoneMixer:
    """
    Optimizes subband synthesis gains to flat-response outputs.
    Equivalent to Gfb_Mixer class in MATLAB [3].
    """

    def __init__(self, analyzer: GammatoneAnalyzer, delay: GammatoneDelay, iterations: int = 100):
        center_frequencies = analyzer.center_frequencies
        num_bands = len(center_frequencies)
        fs = analyzer.sampling_frequency

        z_c = np.exp(2j * np.pi * center_frequencies / fs)
        self.gains: np.ndarray = np.ones(num_bands)

        pos_response = analyzer.get_z_response(z_c)
        neg_response = analyzer.get_z_response(np.conj(z_c))

        for band in range(num_bands):
            pos_response[:, band] = pos_response[:, band] * delay.phase_factors[band] * (
                    z_c ** -delay.delays_samples[band])
            neg_response[:, band] = neg_response[:, band] * delay.phase_factors[band] * (
                    np.conj(z_c) ** -delay.delays_samples[band])

        f_response = (pos_response + np.conj(neg_response)) / 2.0

        for _ in range(iterations):
            selected_spectrum = f_response @ self.gains
            self.gains = self.gains / np.abs(selected_spectrum)

    def process(self, input_data: np.ndarray) -> np.ndarray:
        return self.gains @ input_data


class GammatoneSynthesizer:
    """
    Synthesizes multiple subband signals back into a single fullband waveform.
    Equivalent to Gfb_Synthesizer class in MATLAB [3].
    """

    def __init__(self, analyzer: GammatoneAnalyzer, desired_delay_seconds: float):
        self.delay = GammatoneDelay(analyzer, int(round(desired_delay_seconds * analyzer.sampling_frequency)))
        self.mixer = GammatoneMixer(analyzer, self.delay)

    def process(self, input_data: np.ndarray) -> np.ndarray:
        delayed = self.delay.process(input_data)
        return self.mixer.process(delayed)


def math_factorial(n: int) -> int:
    """Standard integer factorial calculation."""
    return 1 if n <= 1 else n * math_factorial(n - 1)
