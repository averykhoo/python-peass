"""
PEASS Auditory Package - Hohmann 2002 Gammatone Filterbank

Implements the complex-valued Gammatone Filterbank providing physical modeling
of frequency analysis, delay/phase alignment, and synthesis reconstruction.
"""

import math
from functools import lru_cache
from typing import List

import numpy as np
import scipy.signal as signal

try:
    import numba

    _HAS_NUMBA = True


    @numba.njit(cache=True)
    def _numba_gfb_analyze(
            input_signal: np.ndarray,
            coefficients: np.ndarray,
            normalization_factors: np.ndarray,
            states: np.ndarray,
            order: int
    ) -> np.ndarray:
        num_bands = len(coefficients)
        num_samples = len(input_signal)
        output = np.empty((num_bands, num_samples), dtype=numba.complex128)

        for sample_idx in range(num_samples):
            val = input_signal[sample_idx]
            for band_idx in range(num_bands):
                coef = coefficients[band_idx]
                norm = normalization_factors[band_idx]

                # Retrieve states for this band
                s0 = states[band_idx, 0]
                s1 = states[band_idx, 1]
                s2 = states[band_idx, 2]
                s3 = states[band_idx, 3]

                # 4th-order cascaded state update (fully unrolled matching SciPy lfilter)
                y0 = val * norm + s0
                s0 = y0 * coef
                y1 = y0 + s1
                s1 = y1 * coef
                y2 = y1 + s2
                s2 = y2 * coef
                y3 = y2 + s3
                s3 = y3 * coef

                states[band_idx, 0] = s0
                states[band_idx, 1] = s1
                states[band_idx, 2] = s2
                states[band_idx, 3] = s3

                output[band_idx, sample_idx] = y3

        return output


    @numba.njit(cache=True)
    def _numba_delay_process(
            input_data: np.ndarray,
            sample_delays: np.ndarray,
            phase_alignment_factors: np.ndarray,
            state_memory: np.ndarray
    ) -> np.ndarray:
        num_bands, num_samples = input_data.shape
        output_matrix = np.empty((num_bands, num_samples), dtype=numba.float64)

        for band_idx in range(num_bands):
            delay_value = int(sample_delays[band_idx])

            # Complex multiplication real part variables
            phase_factor = phase_alignment_factors[band_idx]
            phase_real = phase_factor.real
            phase_imag = phase_factor.imag

            if delay_value == 0:
                # Write directly to the output matrix, avoiding temporary allocations
                for i in range(num_samples):
                    val = input_data[band_idx, i]
                    output_matrix[band_idx, i] = val.real * phase_real - val.imag * phase_imag
            else:
                # Direct allocation for the combined queue shift
                combined_signal = np.empty(delay_value + num_samples, dtype=numba.float64)
                combined_signal[:delay_value] = state_memory[band_idx, :delay_value]

                # Write phase-rotated values directly into the combined buffer
                for i in range(num_samples):
                    val = input_data[band_idx, i]
                    combined_signal[delay_value + i] = val.real * phase_real - val.imag * phase_imag

                state_memory[band_idx, :delay_value] = combined_signal[num_samples:]
                output_matrix[band_idx, :] = combined_signal[:num_samples]

        return output_matrix


except ImportError:
    _HAS_NUMBA = False


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
            sampling_frequency_hz: float,
            center_frequency_hz: float,
            filter_order: int = 4,
            bandwidth_factor: float = 1.0
    ):
        self.filter_order: int = filter_order
        self.sampling_frequency_hz: float = sampling_frequency_hz
        self.center_frequency_hz: float = center_frequency_hz

        audiological_bandwidth = calculate_equivalent_rectangular_bandwidth(center_frequency_hz) * bandwidth_factor
        gamma_constant = (np.pi * math.factorial(2 * filter_order - 2) * (2.0 ** -(2 * filter_order - 2)) /
                          (math.factorial(filter_order - 1) ** 2))
        decay_constant = audiological_bandwidth / gamma_constant

        self.lambda_decay_factor: float = np.exp(-2.0 * np.pi * decay_constant / sampling_frequency_hz)
        self.frequency_phase_step: float = 2.0 * np.pi * center_frequency_hz / sampling_frequency_hz

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
        factor = self.normalization_factor
        coeff = self.complex_filter_coefficient
        filter_state = self.filter_state * coeff

        y = input_signal.copy()
        b_stage = np.array([factor], dtype=complex)
        a_stage = np.array([1.0, -coeff], dtype=complex)

        new_state = np.zeros(self.filter_order, dtype=complex)
        for i in range(self.filter_order):
            b_coef = b_stage if i == 0 else np.array([1.0], dtype=complex)
            zi = np.array([filter_state[i]], dtype=complex)
            y, zf = signal.lfilter(b_coef, a_stage, y, zi=zi)
            new_state[i] = zf[0]

        self.filter_state = new_state / coeff
        return y

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
            sampling_frequency_hz: float,
            lower_cutoff_frequency_hz: float,
            specified_center_frequency_hz: float,
            upper_cutoff_frequency_hz: float,
            filters_per_equivalent_rectangular_bandwidth: float,
            filter_order: int = 4,
            bandwidth_factor: float = 1.0
    ):
        self._sampling_frequency_hz: float = sampling_frequency_hz
        self.center_frequencies: np.ndarray = get_equivalent_rectangular_bandwidth_center_frequencies(
            filters_per_equivalent_rectangular_bandwidth,
            lower_cutoff_frequency_hz,
            specified_center_frequency_hz,
            upper_cutoff_frequency_hz
        )
        self.filters: List[GammatoneFilter] = [
            GammatoneFilter(sampling_frequency_hz, freq, filter_order, bandwidth_factor)
            for freq in self.center_frequencies
        ]
        self.bandwidths: np.ndarray = calculate_equivalent_rectangular_bandwidth(self.center_frequencies)
        self.original_sampling_frequency_hz: float = sampling_frequency_hz
        self.decimation_factors: np.ndarray = np.ones(len(self.filters), dtype=int)

    @property
    def sampling_frequency_hz(self) -> float:
        return self._sampling_frequency_hz

    @sampling_frequency_hz.setter
    def sampling_frequency_hz(self, value: float) -> None:
        self._sampling_frequency_hz = value

    @property
    def center_frequencies_hz(self) -> np.ndarray:
        return self.center_frequencies

    def process(self, input_signal: np.ndarray) -> np.ndarray:
        num_bands = len(self.filters)

        if _HAS_NUMBA and self.filters and self.filters[0].filter_order == 4:
            # Vectorized JIT path: Process all bands simultaneously
            coeffs = np.array([f.complex_filter_coefficient for f in self.filters], dtype=complex)
            norms = np.array([f.normalization_factor for f in self.filters], dtype=float)

            # Pack states into a contiguous contiguous 2D array [bands, order]
            states = np.empty((num_bands, 4), dtype=complex)
            for b in range(num_bands):
                states[b, :] = self.filters[b].state * self.filters[b].complex_filter_coefficient

            output_matrix = _numba_gfb_analyze(input_signal, coeffs, norms, states, 4)

            # Unpack states back
            for b in range(num_bands):
                self.filters[b].state = states[b, :] / self.filters[b].complex_filter_coefficient

            return output_matrix
        else:
            # Fallback path
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


# -----------------------------------------------------------------------------
# CACHED HELPER FUNCTIONS (Using functools.lru_cache with hashable primitives)
# -----------------------------------------------------------------------------

@lru_cache
def _get_delay_unit_parameters_cached(
        sampling_frequency_hz: float,
        target_delay_samples: int,
        center_frequencies_tuple: tuple
) -> tuple:
    # Recreate a lightweight local analyzer to perform the impulse response processing.
    # This prevents mutating the state of the active filterbank and ensures cache stability
    # across different physical instances with identical parameters.
    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=sampling_frequency_hz,
        lower_cutoff_frequency_hz=center_frequencies_tuple[0],
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=center_frequencies_tuple[-1],
        filters_per_equivalent_rectangular_bandwidth=1.0
    )
    # Force exact match of center frequencies to avoid floating-point tolerances
    analyzer.center_frequencies = np.array(center_frequencies_tuple)
    for i, freq in enumerate(center_frequencies_tuple):
        analyzer.filters[i].center_frequency_hz = freq

    impulse_signal = np.zeros(target_delay_samples + 2)
    impulse_signal[0] = 1.0

    impulse_response = analyzer.process(impulse_signal)
    num_bands = impulse_response.shape[0]

    slice_duration = np.abs(impulse_response[:, :target_delay_samples + 1])
    max_amplitude_indices = np.argmax(slice_duration, axis=1)

    sample_delays = target_delay_samples - max_amplitude_indices
    frequency_slopes = np.zeros(num_bands, dtype=complex)
    for band_idx in range(num_bands):
        max_idx = max_amplitude_indices[band_idx]
        prev_val = impulse_response[band_idx, max_idx - 1] if max_idx > 0 else 0.0j
        next_val = impulse_response[band_idx, max_idx + 1] if max_idx + 1 < impulse_response.shape[1] else 0.0j
        frequency_slopes[band_idx] = next_val - prev_val

    frequency_slopes = frequency_slopes / (np.abs(frequency_slopes) + np.finfo(float).eps)
    phase_alignment_factors = 1j / frequency_slopes

    return sample_delays, phase_alignment_factors


def get_delay_unit_parameters(analyzer: GammatoneAnalyzer, target_delay_samples: int) -> tuple:
    """Helper to lazily compute and cache impulse response delays and phase factors."""
    return _get_delay_unit_parameters_cached(
        analyzer.sampling_frequency_hz,
        target_delay_samples,
        tuple(analyzer.center_frequencies)
    )


@lru_cache
def _get_mixer_gains_cached(
        sampling_rate: float,
        center_frequencies_tuple: tuple,
        sample_delays_tuple: tuple,
        phase_alignment_factors_tuple: tuple,
        optimization_iterations: int
) -> np.ndarray:
    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=sampling_rate,
        lower_cutoff_frequency_hz=center_frequencies_tuple[0],
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=center_frequencies_tuple[-1],
        filters_per_equivalent_rectangular_bandwidth=1.0
    )
    analyzer.center_frequencies = np.array(center_frequencies_tuple)

    center_frequencies = np.array(center_frequencies_tuple)
    num_bands = len(center_frequencies)

    z_center = np.exp(2j * np.pi * center_frequencies / sampling_rate)
    synthesis_gains = np.ones(num_bands)

    positive_response = analyzer.get_z_plane_frequency_response(z_center)
    negative_response = analyzer.get_z_plane_frequency_response(np.conj(z_center))

    for band_idx in range(num_bands):
        positive_response[:, band_idx] = (
                positive_response[:, band_idx] *
                phase_alignment_factors_tuple[band_idx] *
                (z_center ** -sample_delays_tuple[band_idx])
        )
        negative_response[:, band_idx] = (
                negative_response[:, band_idx] *
                phase_alignment_factors_tuple[band_idx] *
                (np.conj(z_center) ** -sample_delays_tuple[band_idx])
        )

    combined_frequency_response = (positive_response + np.conj(negative_response)) / 2.0

    for _ in range(optimization_iterations):
        composite_spectrum = combined_frequency_response @ synthesis_gains
        synthesis_gains = synthesis_gains / (np.abs(composite_spectrum) + np.finfo(float).eps)

    return synthesis_gains


def get_mixer_gains(analyzer: GammatoneAnalyzer, delay_unit: "GammatoneDelay",
                    optimization_iterations: int) -> np.ndarray:
    """Helper to lazily optimize and cache synthesis mixer gains."""
    return _get_mixer_gains_cached(
        analyzer.sampling_frequency_hz,
        tuple(analyzer.center_frequencies),
        tuple(delay_unit.sample_delays),
        tuple(delay_unit.phase_alignment_factors),
        optimization_iterations
    )


class GammatoneDelay:
    """
    Handles phase alignment and group delay estimation across subbands.
    """

    def __init__(self, analyzer: GammatoneAnalyzer, target_delay_samples: int):
        # Fetch pre-calculated delay parameters instantly from cache
        sample_delays, phase_alignment_factors = get_delay_unit_parameters(
            analyzer, target_delay_samples
        )

        self.sample_delays: np.ndarray = sample_delays
        self.phase_alignment_factors: np.ndarray = phase_alignment_factors
        num_bands = len(sample_delays)
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
        if _HAS_NUMBA:
            return _numba_delay_process(
                input_data,
                self.sample_delays,
                self.phase_alignment_factors,
                self.state_memory
            )
        else:
            num_bands, num_samples = input_data.shape
            output_matrix = np.zeros((num_bands, num_samples))
            for band_idx in range(num_bands):
                delay_value = int(self.sample_delays[band_idx])
                phase_corrected_signal = np.real(input_data[band_idx, :] * self.phase_alignment_factors[band_idx])
                if delay_value == 0:
                    output_matrix[band_idx, :] = phase_corrected_signal
                else:
                    combined_signal = np.concatenate(
                        (self.state_memory[band_idx, :delay_value], phase_corrected_signal))
                    self.state_memory[band_idx, :delay_value] = combined_signal[num_samples:]
                    output_matrix[band_idx, :] = combined_signal[:num_samples]
            return output_matrix


class GammatoneMixer:
    """
    Optimizes subband synthesis gains to flat-response outputs.
    """

    def __init__(self, analyzer: GammatoneAnalyzer, delay_unit: GammatoneDelay, optimization_iterations: int = 100):
        # Fetch pre-calculated synthesis gains instantly from cache
        self.synthesis_gains: np.ndarray = get_mixer_gains(analyzer, delay_unit, optimization_iterations)

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


@lru_cache(maxsize=256)
def get_resample_filter(up: int, down: int) -> tuple:
    g = math.gcd(up, down)
    up_reduced = up // g
    down_reduced = down // g

    max_len = max(up_reduced, down_reduced)

    # REDUCED ORDER FILTER DESIGN:
    # Scale from 10 to 3. Because subband signals are already band-limited
    # by the Gammatone filterbank, 3 * max_len provides excellent anti-aliasing
    # while reducing filter taps by 70% (e.g. from 10,001 to 3,001 taps).
    half_len = 3 * max_len
    n_filt = 2 * half_len + 1

    # 1. Design standard Kaiser FIR filter
    h = signal.firwin(n_filt, 1.0 / max_len, window=('kaiser', 5.0))
    h *= up_reduced

    # 2. Perfect replication of SciPy's zero-phase centering padding
    n_pre_pad = (down_reduced - half_len % down_reduced) % down_reduced
    h_padded = np.pad(h, (n_pre_pad, 0))

    # 3. Calculate cropping offset
    n_pre_remove = (half_len + n_pre_pad) // down_reduced

    return h_padded, up_reduced, down_reduced, n_pre_remove


def fast_resample_poly(x: np.ndarray, up: int, down: int, axis: int = -1) -> np.ndarray:
    if up == down:
        return x.copy()

    h_padded, up_reduced, down_reduced, n_pre_remove = get_resample_filter(up, down)

    in_len = x.shape[axis]
    out_len = int(np.ceil(in_len * up_reduced / down_reduced))

    # Run high-speed upfirdn
    y = signal.upfirdn(h_padded, x, up_reduced, down_reduced, axis=axis)

    # Slice the output to keep only the centered portion of out_len samples
    keep = [slice(None)] * y.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove + out_len)

    return y[tuple(keep)]
