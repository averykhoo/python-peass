"""
PEASS Auditory Package - Hohmann 2002 Gammatone Filterbank

Implements the complex-valued Gammatone Filterbank providing physical modeling
of frequency analysis, delay/phase alignment, and synthesis reconstruction.
"""

import math
from typing import List, Optional

import numpy as np
import scipy.signal as signal


def calculate_equivalent_rectangular_bandwidth(center_frequency_hz: float) -> float:
    """
    Computes the Equivalent Rectangular Bandwidth of auditory filters.
    """
    return 24.7 * (0.00437 * center_frequency_hz + 1.0)


def convert_frequency_to_equivalent_rectangular_bandwidth_scale(frequency_hz: float) -> float:
    """
    Converts frequency in Hz to Equivalent Rectangular Bandwidth (ERB) scale.
    """
    return 9.265 * np.log(1.0 + frequency_hz / (24.7 * 9.265))


def convert_equivalent_rectangular_bandwidth_scale_to_frequency(erb_scale_value: float) -> float:
    """
    Converts Equivalent Rectangular Bandwidth (ERB) scale value to frequency in Hz.
    """
    return (np.exp(erb_scale_value / 9.265) - 1.0) * (24.7 * 9.265)


def get_equivalent_rectangular_bandwidth_center_frequencies(
        filters_per_erb: float,
        lower_cutoff_frequency_hz: float,
        specified_center_frequency_hz: float,
        upper_cutoff_frequency_hz: float
) -> np.ndarray:
    """
    Constructs a vector of center frequencies equidistant on the ERB scale.
    """
    lower_erb = convert_frequency_to_equivalent_rectangular_bandwidth_scale(lower_cutoff_frequency_hz)
    specified_erb = convert_frequency_to_equivalent_rectangular_bandwidth_scale(specified_center_frequency_hz)
    upper_erb = convert_frequency_to_equivalent_rectangular_bandwidth_scale(upper_cutoff_frequency_hz)

    erbs_below_base = specified_erb - lower_erb
    num_filters_below = int(np.floor(erbs_below_base * filters_per_erb))

    start_erb = specified_erb - (num_filters_below / filters_per_erb)
    center_erbs = np.arange(start_erb, upper_erb + 1e-9, 1.0 / filters_per_erb)
    return convert_equivalent_rectangular_bandwidth_scale_to_frequency(center_erbs)


class GammatoneFilter:
    """
    Represents a complex-valued all-pole Gammatone filter.
    Optimized by collapsing the cascade into a single polynomial transfer function.
    """

    def __init__(
            self,
            sampling_frequency_hz: Optional[float] = None,
            center_frequency_hz: Optional[float] = None,
            filter_order: int = 4,
            bandwidth_factor: float = 1.0,
            sampling_frequency: Optional[float] = None, # Legacy alias
            center_frequency: Optional[float] = None # Legacy alias
    ):
        fs = sampling_frequency_hz if sampling_frequency_hz is not None else sampling_frequency
        fc = center_frequency_hz if center_frequency_hz is not None else center_frequency

        if fs is None:
            raise TypeError("GammatoneFilter.__init__() missing 1 required positional argument: 'sampling_frequency_hz'")
        if fc is None:
            raise TypeError("GammatoneFilter.__init__() missing 1 required positional argument: 'center_frequency_hz'")

        self.filter_order: int = filter_order
        self.sampling_frequency_hz: float = fs
        self.center_frequency_hz: float = fc

        audiological_bandwidth = calculate_equivalent_rectangular_bandwidth(fc) * bandwidth_factor
        gamma_constant = (np.pi * math.factorial(2 * filter_order - 2) * (2.0 ** -(2 * filter_order - 2)) /
                          (math.factorial(filter_order - 1) ** 2))
        decay_constant = audiological_bandwidth / gamma_constant

        self.lambda_decay_factor: float = np.exp(-2.0 * np.pi * decay_constant / fs)
        self.frequency_phase_step: float = 2.0 * np.pi * fc / fs

        self.complex_filter_coefficient: complex = self.lambda_decay_factor * np.exp(1j * self.frequency_phase_step)
        self.normalization_factor: float = 2.0 * (1.0 - np.abs(self.complex_filter_coefficient)) ** filter_order

        # Expand the cascade of N 1st-order poles into a single Nth-order denominator polynomial
        self.numerator_coefficients = np.array([self.normalization_factor], dtype=complex)
        self.denominator_coefficients = np.poly([self.complex_filter_coefficient] * self.filter_order)

        self.filter_state: np.ndarray = np.zeros(self.filter_order, dtype=complex)

    @property
    def state(self) -> np.ndarray:
        return self.filter_state

    @state.setter
    def state(self, value: np.ndarray) -> None:
        self.filter_state = value

    @property
    def coefficient(self) -> complex:
        return self.complex_filter_coefficient

    @property
    def gamma_order(self) -> int:
        return self.filter_order

    def process(self, input_signal: np.ndarray) -> np.ndarray:
        # Offload the entire N-th order filter mathematically to SciPy's C backend
        output_signal, self.filter_state = signal.lfilter(
            self.numerator_coefficients,
            self.denominator_coefficients,
            input_signal,
            zi=self.filter_state
        )
        return output_signal

    def clear_filter_state(self) -> None:
        self.filter_state = np.zeros(self.filter_order, dtype=complex)

    def clear_state(self) -> None:
        self.clear_filter_state()


class GammatoneAnalyzer:
    """
    A collection of GammatoneFilters acting as an analysis filterbank.
    """

    def __init__(
            self,
            sampling_frequency_hz: Optional[float] = None,
            lower_cutoff_frequency_hz: Optional[float] = None,
            specified_center_frequency_hz: Optional[float] = None,
            upper_cutoff_frequency_hz: Optional[float] = None,
            filters_per_equivalent_rectangular_bandwidth: Optional[float] = None,
            filter_order: int = 4,
            bandwidth_factor: float = 1.0,
            sampling_frequency: Optional[float] = None, # Legacy alias
            lower_cutoff_hz: Optional[float] = None, # Legacy alias
            specified_center_hz: Optional[float] = None, # Legacy alias
            upper_cutoff_hz: Optional[float] = None, # Legacy alias
            filters_per_erb: Optional[float] = None # Legacy alias
    ):
        fs = sampling_frequency_hz if sampling_frequency_hz is not None else sampling_frequency
        lower_cf = lower_cutoff_frequency_hz if lower_cutoff_frequency_hz is not None else lower_cutoff_hz
        base_cf = specified_center_frequency_hz if specified_center_frequency_hz is not None else specified_center_hz
        upper_cf = upper_cutoff_frequency_hz if upper_cutoff_frequency_hz is not None else upper_cutoff_hz
        density = filters_per_equivalent_rectangular_bandwidth if filters_per_equivalent_rectangular_bandwidth is not None else filters_per_erb

        if fs is None:
            raise TypeError("GammatoneAnalyzer.__init__() missing 1 required positional argument: 'sampling_frequency_hz'")
        if lower_cf is None:
            raise TypeError("GammatoneAnalyzer.__init__() missing 1 required positional argument: 'lower_cutoff_frequency_hz'")
        if base_cf is None:
            raise TypeError("GammatoneAnalyzer.__init__() missing 1 required positional argument: 'specified_center_frequency_hz'")
        if upper_cf is None:
            raise TypeError("GammatoneAnalyzer.__init__() missing 1 required positional argument: 'upper_cutoff_frequency_hz'")
        if density is None:
            raise TypeError("GammatoneAnalyzer.__init__() missing 1 required positional argument: 'filters_per_equivalent_rectangular_bandwidth'")

        self._sampling_frequency_hz: float = fs
        self.center_frequencies: np.ndarray = get_equivalent_rectangular_bandwidth_center_frequencies(
            density,
            lower_cf,
            base_cf,
            upper_cf
        )
        self.filters: List[GammatoneFilter] = [
            GammatoneFilter(fs, freq, filter_order, bandwidth_factor)
            for freq in self.center_frequencies
        ]
        self.bandwidths: np.ndarray = calculate_equivalent_rectangular_bandwidth(self.center_frequencies)
        self.original_sampling_frequency_hz: float = fs
        self.decimation_factors: np.ndarray = np.ones(len(self.filters), dtype=int)

    @property
    def sampling_frequency_hz(self) -> float:
        return self._sampling_frequency_hz

    @sampling_frequency_hz.setter
    def sampling_frequency_hz(self, value: float) -> None:
        self._sampling_frequency_hz = value

    @property
    def sampling_frequency(self) -> float:
        return self._sampling_frequency_hz

    @property
    def center_frequencies_hz(self) -> np.ndarray:
        return self.center_frequencies

    def process(self, input_signal: np.ndarray) -> np.ndarray:
        num_bands = len(self.filters)
        output_matrix = np.zeros((num_bands, input_signal.shape[0]), dtype=complex)
        for band_idx in range(num_bands):
            output_matrix[band_idx, :] = self.filters[band_idx].process(input_signal)
        return output_matrix

    def get_z_plane_frequency_response(self, z_points: np.ndarray) -> np.ndarray:
        z_col = z_points[:, np.newaxis]
        num_bands = len(self.filters)
        response_matrix = np.ones((z_col.shape[0], num_bands), dtype=complex)
        for band_idx in range(num_bands):
            coefficient = self.filters[band_idx].complex_filter_coefficient
            normalization = self.filters[band_idx].normalization_factor
            order = self.filters[band_idx].filter_order
            response_matrix[:, band_idx] = ((1.0 - coefficient / z_col[:, 0]) ** -order) * normalization
        return response_matrix

    def clear_filterbank_states(self) -> None:
        for filter_instance in self.filters:
            filter_instance.clear_filter_state()

    def clear_state(self) -> None:
        self.clear_filterbank_states()


class GammatoneDelay:
    """
    Handles phase alignment and group delay estimation across subbands.
    """

    def __init__(self, analyzer: GammatoneAnalyzer, target_delay_samples: int):
        analyzer.clear_filterbank_states()
        impulse_signal = np.zeros(target_delay_samples + 2)
        impulse_signal[0] = 1.0

        impulse_response = analyzer.process(impulse_signal)
        num_bands = impulse_response.shape[0]

        slice_duration = np.abs(impulse_response[:, :target_delay_samples + 1])
        max_amplitude_indices = np.argmax(slice_duration, axis=1)

        self.sample_delays: np.ndarray = target_delay_samples - max_amplitude_indices
        frequency_slopes = np.zeros(num_bands, dtype=complex)
        for band_idx in range(num_bands):
            max_idx = max_amplitude_indices[band_idx]
            frequency_slopes[band_idx] = (
                    impulse_response[band_idx, max_idx + 1] - impulse_response[band_idx, max_idx - 1]
            )

        frequency_slopes = frequency_slopes / (np.abs(frequency_slopes) + np.finfo(float).eps)
        self.phase_alignment_factors: np.ndarray = 1j / frequency_slopes
        self.state_memory: np.ndarray = np.zeros((num_bands, int(np.max(self.sample_delays))), dtype=float)

    @property
    def delays_samples(self) -> np.ndarray:
        return self.sample_delays

    @property
    def phase_factors(self) -> np.ndarray:
        return self.phase_alignment_factors

    @property
    def memory(self) -> np.ndarray:
        return self.state_memory

    def process(self, input_data: np.ndarray) -> np.ndarray:
        num_bands, num_samples = input_data.shape
        output_matrix = np.zeros((num_bands, num_samples))
        for band_idx in range(num_bands):
            delay_value = int(self.sample_delays[band_idx])
            phase_corrected_signal = np.real(input_data[band_idx, :] * self.phase_alignment_factors[band_idx])
            if delay_value == 0:
                output_matrix[band_idx, :] = phase_corrected_signal
            else:
                combined_signal = np.concatenate((self.state_memory[band_idx, :delay_value], phase_corrected_signal))
                self.state_memory[band_idx, :delay_value] = combined_signal[num_samples:]
                output_matrix[band_idx, :] = combined_signal[:num_samples]
        return output_matrix


class GammatoneMixer:
    """
    Optimizes subband synthesis gains to flat-response outputs.
    """

    def __init__(self, analyzer: GammatoneAnalyzer, delay_unit: GammatoneDelay, optimization_iterations: int = 100):
        center_frequencies = analyzer.center_frequencies
        num_bands = len(center_frequencies)
        sampling_rate = analyzer.sampling_frequency_hz

        z_center = np.exp(2j * np.pi * center_frequencies / sampling_rate)
        self.synthesis_gains: np.ndarray = np.ones(num_bands)

        positive_response = analyzer.get_z_plane_frequency_response(z_center)
        negative_response = analyzer.get_z_plane_frequency_response(np.conj(z_center))

        for band_idx in range(num_bands):
            positive_response[:, band_idx] = (
                    positive_response[:, band_idx] *
                    delay_unit.phase_alignment_factors[band_idx] *
                    (z_center ** -delay_unit.sample_delays[band_idx])
            )
            negative_response[:, band_idx] = (
                    negative_response[:, band_idx] *
                    delay_unit.phase_alignment_factors[band_idx] *
                    (np.conj(z_center) ** -delay_unit.sample_delays[band_idx])
            )

        combined_frequency_response = (positive_response + np.conj(negative_response)) / 2.0

        for _ in range(optimization_iterations):
            composite_spectrum = combined_frequency_response @ self.synthesis_gains
            self.synthesis_gains = self.synthesis_gains / (np.abs(composite_spectrum) + np.finfo(float).eps)

    @property
    def gains(self) -> np.ndarray:
        return self.synthesis_gains

    def process(self, input_data: np.ndarray) -> np.ndarray:
        return self.synthesis_gains @ input_data


class GammatoneSynthesizer:
    """
    Synthesizes multiple subband signals back into a single fullband waveform.
    """

    def __init__(self, analyzer: GammatoneAnalyzer, desired_delay_seconds: float):
        self.delay_unit = GammatoneDelay(analyzer, int(round(desired_delay_seconds * analyzer.sampling_frequency_hz)))
        self.mixer_unit = GammatoneMixer(analyzer, self.delay_unit)

    @property
    def delay(self) -> GammatoneDelay:
        return self.delay_unit

    @property
    def mixer(self) -> GammatoneMixer:
        return self.mixer_unit

    def process(self, input_data: np.ndarray) -> np.ndarray:
        delayed_signal = self.delay_unit.process(input_data)
        return self.mixer_unit.process(delayed_signal)


# -----------------------------------------------------------------------------
# LEGACY BACKWARD-COMPATIBILITY ALIASES
# -----------------------------------------------------------------------------
calculate_erb_bandwidth = calculate_equivalent_rectangular_bandwidth
frequency_to_erb_scale = convert_frequency_to_equivalent_rectangular_bandwidth_scale
erb_scale_to_frequency = convert_equivalent_rectangular_bandwidth_scale_to_frequency
get_center_frequencies = get_equivalent_rectangular_bandwidth_center_frequencies