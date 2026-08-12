"""
PEASS Auditory Package - Hohmann 2002 Gammatone Filterbank

Implements the complex-valued Gammatone Filterbank providing physical modeling
of frequency analysis, delay/phase alignment, and synthesis reconstruction.
"""

import math
from functools import lru_cache

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


    @numba.njit(cache=True, fastmath=True)
    def _numba_polyphase_decimate(
            reversed_filter: np.ndarray,
            padded_signal: np.ndarray,
            output: np.ndarray,
            down: int,
            n_pre_remove: int,
            out_len: int,
            num_taps: int
    ) -> np.ndarray:
        """Pure decimation (up == 1): one stride-1 dot product per output sample.

        The signal is pre-padded so the dot product has a constant trip count with
        no bounds arithmetic, and both operands are contiguous float64 so numba
        lowers `np.dot` to BLAS `ddot` (measured 2.1x over the hand-rolled tap loop
        at MKL_NUM_THREADS=1, i.e. the win is AVX vectorization, not BLAS
        threading). Complex input is handled by `_numba_polyphase_decimate_complex`
        on split real/imag buffers -- `ddot` needs real operands. The padding is
        load-bearing, not tidiness.

        `output` is supplied by the caller (see `fast_resample_poly`'s `out=`); it
        may be WIDER than `out_len`, and only columns [0, out_len) are written, so
        a caller can scatter straight into a zero-filled destination without the
        filter tail leaking past the region it owns. Every dot product reads only
        `padded_signal`, so the destination's row stride cannot perturb a value.

        The taps and the products are identical to SciPy's; `fastmath`/`ddot`
        reassociate the accumulation. See USE_NUMBA_RESAMPLER for the accuracy
        trade-off and the opt-out.
        """
        num_rows = padded_signal.shape[0]
        for row_idx in range(num_rows):
            row = padded_signal[row_idx]
            for sample_idx in range(out_len):
                base = (sample_idx + n_pre_remove) * down
                output[row_idx, sample_idx] = np.dot(reversed_filter, row[base:base + num_taps])
        return output


    @numba.njit(cache=True, fastmath=True)
    def _numba_polyphase_decimate_complex(
            reversed_filter: np.ndarray,
            padded_real: np.ndarray,
            padded_imag: np.ndarray,
            output: np.ndarray,
            down: int,
            n_pre_remove: int,
            out_len: int,
            num_taps: int
    ) -> np.ndarray:
        """Complex twin of `_numba_polyphase_decimate`: two `ddot`s per output
        sample on split real/imag buffers, writing complex128 directly so the
        wrapper never recombines. `output` is caller-supplied and only its first
        `out_len` columns are written."""
        num_rows = padded_real.shape[0]
        for row_idx in range(num_rows):
            row_real = padded_real[row_idx]
            row_imag = padded_imag[row_idx]
            for sample_idx in range(out_len):
                base = (sample_idx + n_pre_remove) * down
                output[row_idx, sample_idx] = complex(
                    np.dot(reversed_filter, row_real[base:base + num_taps]),
                    np.dot(reversed_filter, row_imag[base:base + num_taps]),
                )
        return output


    @numba.njit(cache=True, fastmath=True)
    def _numba_polyphase_interpolate(
            branches_transposed: np.ndarray,
            padded_signal: np.ndarray,
            output: np.ndarray,
            up: int,
            n_pre_remove: int,
            out_len: int,
            pad_left: int,
            branch_length: int
    ) -> np.ndarray:
        """Interpolation (down == 1) via per-phase polyphase branches.

        y[n] = sum_j branch[n % up][j] * x[n // up - branch_length + 1 + j]

        Loop order is inverted relative to that formula: for each input base index
        q = n // up the kernel runs the tap loop j outside and the phase loop p
        innermost, accumulating into a small L1-resident buffer of length `up`.
        The phase loop walks `branches_transposed[j, :]` (contiguous) with a single
        broadcast input sample -- an AXPY that LLVM fully vectorizes, unlike the
        original per-output-sample reduction over strided branch taps (measured
        ~0.42 GMAC/s scalar vs a ~2.3 GMAC/s vectorized ceiling; same FLOPs).
        First/last base indices may cover output samples outside
        [n_pre_remove, n_pre_remove + out_len); those phases are computed and
        discarded (at most one branch worth of work per row).

        `output` is caller-supplied (see `fast_resample_poly`'s `out=`) and may be
        wider than `out_len`; the p_lo/p_hi clamp already restricts the stores to
        columns [0, out_len), so nothing outside that window is ever touched. The
        accumulation happens entirely in the local `accumulator`, so the
        destination's shape cannot change a single rounding.
        """
        num_rows = padded_signal.shape[0]
        accumulator = np.empty(up, dtype=np.float64)
        base_start = n_pre_remove // up
        base_end = (n_pre_remove + out_len - 1) // up
        for row_idx in range(num_rows):
            row = padded_signal[row_idx]
            for q in range(base_start, base_end + 1):
                for p in range(up):
                    accumulator[p] = 0.0
                start = pad_left + q - branch_length + 1
                for tap_idx in range(branch_length):
                    x_val = row[start + tap_idx]
                    for p in range(up):
                        accumulator[p] += branches_transposed[tap_idx, p] * x_val
                n0 = q * up
                p_lo = max(n_pre_remove - n0, 0)
                p_hi = min(n_pre_remove + out_len - n0, up)
                for p in range(p_lo, p_hi):
                    output[row_idx, n0 + p - n_pre_remove] = accumulator[p]
        return output


    @numba.njit(cache=True, fastmath=True)
    def _numba_polyphase_interpolate_complex(
            branches_transposed: np.ndarray,
            padded_real: np.ndarray,
            padded_imag: np.ndarray,
            output: np.ndarray,
            up: int,
            n_pre_remove: int,
            out_len: int,
            pad_left: int,
            branch_length: int
    ) -> np.ndarray:
        """Complex twin of `_numba_polyphase_interpolate`: the same AXPY loop
        inversion run on split real/imag buffers (two real AXPYs per tap), writing
        complex128 directly so the wrapper never recombines. `output` is
        caller-supplied and only its first `out_len` columns are written."""
        num_rows = padded_real.shape[0]
        accumulator_real = np.empty(up, dtype=np.float64)
        accumulator_imag = np.empty(up, dtype=np.float64)
        base_start = n_pre_remove // up
        base_end = (n_pre_remove + out_len - 1) // up
        for row_idx in range(num_rows):
            row_real = padded_real[row_idx]
            row_imag = padded_imag[row_idx]
            for q in range(base_start, base_end + 1):
                for p in range(up):
                    accumulator_real[p] = 0.0
                    accumulator_imag[p] = 0.0
                start = pad_left + q - branch_length + 1
                for tap_idx in range(branch_length):
                    x_real = row_real[start + tap_idx]
                    for p in range(up):
                        accumulator_real[p] += branches_transposed[tap_idx, p] * x_real
                    x_imag = row_imag[start + tap_idx]
                    for p in range(up):
                        accumulator_imag[p] += branches_transposed[tap_idx, p] * x_imag
                n0 = q * up
                p_lo = max(n_pre_remove - n0, 0)
                p_hi = min(n_pre_remove + out_len - n0, up)
                for p in range(p_lo, p_hi):
                    output[row_idx, n0 + p - n_pre_remove] = complex(
                        accumulator_real[p], accumulator_imag[p]
                    )
        return output


except ImportError:
    _HAS_NUMBA = False


def calculate_equivalent_rectangular_bandwidth(center_frequency_hz: float) -> float:
    """
    Computes the Equivalent Rectangular Bandwidth of auditory filters.

    This is the Glasberg & Moore form used by the MATLAB reference's `erbBW.m`. It is
    used ONLY to pick the per-band decimation factor in the analysis filterbank
    (`myPemoAnalysisFilterBank.m:53`), NOT to build the gammatone filter coefficients.
    See GAMMATONE_ERB_INTERCEPT_HZ below for the (deliberately different) form used by
    the filter constructor.
    """
    return 24.7 * (0.00437 * center_frequency_hz + 1.0)


# Gammatone filter bandwidth constants, from `gammatone/Gfb_set_constants.m` (GFB_L and
# GFB_Q), i.e. equation (17) in Hohmann 2002. The filter constructor
# (`gammatone/Gfb_Filter_new.m:61`) computes its audiological ERB as
#     (GFB_L + fc / GFB_Q) * bandwidth_factor
#
# This is the SAME empirical fit as `calculate_equivalent_rectangular_bandwidth` above,
# re-parameterized with one constant rounded: 1 / (24.7 * 0.00437) = 9.264488..., which
# Hohmann rounds to 9.265. The two therefore differ by ~5.5e-5 relative on the slope
# term. The MATLAB reference deliberately keeps both forms and applies each in a
# different place -- Hohmann's rounded form for the filter coefficients, Glasberg &
# Moore's form for the decimation factors -- so do NOT collapse them onto one formula.
GAMMATONE_ERB_INTERCEPT_HZ = 24.7  # GFB_L
GAMMATONE_ERB_QUALITY_FACTOR = 9.265  # GFB_Q


def calculate_audiological_equivalent_rectangular_bandwidth(
        center_frequency_hz: float,
        bandwidth_factor: float = 1.0
) -> float:
    """
    Computes the audiological ERB used to derive gammatone filter coefficients.

    Mirrors `audiological_erb` in `gammatone/Gfb_Filter_new.m:61` (Hohmann 2002 eq. 13
    and 17). Intentionally distinct from `calculate_equivalent_rectangular_bandwidth`;
    see the constants above.
    """
    return (GAMMATONE_ERB_INTERCEPT_HZ +
            center_frequency_hz / GAMMATONE_ERB_QUALITY_FACTOR) * bandwidth_factor


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

        # Hohmann's rounded form (Gfb_Filter_new.m:61), NOT the erbBW form used for decimation
        audiological_bandwidth = calculate_audiological_equivalent_rectangular_bandwidth(
            center_frequency_hz, bandwidth_factor
        )
        gamma_constant = (np.pi * math.factorial(2 * filter_order - 2) * (2.0 ** -(2 * filter_order - 2)) /
                          (math.factorial(filter_order - 1) ** 2))
        decay_constant = audiological_bandwidth / gamma_constant

        self.lambda_decay_factor: float = np.exp(-2.0 * np.pi * decay_constant / sampling_frequency_hz)
        self.frequency_phase_step: float = 2.0 * np.pi * center_frequency_hz / sampling_frequency_hz

        self.complex_filter_coefficient: complex = self.lambda_decay_factor * np.exp(1j * self.frequency_phase_step)
        self.normalization_factor: float = 2.0 * (1.0 - np.abs(self.complex_filter_coefficient)) ** filter_order

        # commented out because these aren't used, maybe they were meant to be a cache?
        # # Expand the cascade of N 1st-order poles into a single Nth-order denominator polynomial
        # self.numerator_coefficients = np.array([self.normalization_factor], dtype=complex)
        # self.denominator_coefficients = np.poly([self.complex_filter_coefficient] * self.filter_order)

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

        # Save instantiation parameters as instance attributes to prevent re-instantiation mismatches
        self.lower_cutoff_frequency_hz: float = lower_cutoff_frequency_hz
        self.specified_center_frequency_hz: float = specified_center_frequency_hz
        self.upper_cutoff_frequency_hz: float = upper_cutoff_frequency_hz
        self.filters_per_equivalent_rectangular_bandwidth: float = filters_per_equivalent_rectangular_bandwidth
        self.filter_order: int = filter_order
        self.bandwidth_factor: float = bandwidth_factor

        self.center_frequencies: np.ndarray = get_equivalent_rectangular_bandwidth_center_frequencies(
            filters_per_equivalent_rectangular_bandwidth,
            lower_cutoff_frequency_hz,
            specified_center_frequency_hz,
            upper_cutoff_frequency_hz
        )
        self.filters: list[GammatoneFilter] = [
            GammatoneFilter(sampling_frequency_hz, freq, filter_order, bandwidth_factor)
            for freq in self.center_frequencies
        ]
        # erbBW form: these bandwidths drive the decimation factors, not the filter coefficients
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
        lower_cutoff_frequency_hz: float,
        specified_center_frequency_hz: float,
        upper_cutoff_frequency_hz: float,
        filters_per_equivalent_rectangular_bandwidth: float,
        filter_order: int,
        bandwidth_factor: float,
        target_delay_samples: int,
        center_frequencies_tuple: tuple
) -> tuple:
    # Reconstruct the analyzer with identical parameter signatures
    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=sampling_frequency_hz,
        lower_cutoff_frequency_hz=lower_cutoff_frequency_hz,
        specified_center_frequency_hz=specified_center_frequency_hz,
        upper_cutoff_frequency_hz=upper_cutoff_frequency_hz,
        filters_per_equivalent_rectangular_bandwidth=filters_per_equivalent_rectangular_bandwidth,
        filter_order=filter_order,
        bandwidth_factor=bandwidth_factor
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
        analyzer.lower_cutoff_frequency_hz,
        analyzer.specified_center_frequency_hz,
        analyzer.upper_cutoff_frequency_hz,
        analyzer.filters_per_equivalent_rectangular_bandwidth,
        analyzer.filter_order,
        analyzer.bandwidth_factor,
        target_delay_samples,
        tuple(analyzer.center_frequencies)
    )


@lru_cache
def _get_mixer_gains_cached(
        sampling_rate: float,
        lower_cutoff_frequency_hz: float,
        specified_center_frequency_hz: float,
        upper_cutoff_frequency_hz: float,
        filters_per_equivalent_rectangular_bandwidth: float,
        filter_order: int,
        bandwidth_factor: float,
        center_frequencies_tuple: tuple,
        sample_delays_tuple: tuple,
        phase_alignment_factors_tuple: tuple,
        optimization_iterations: int
) -> np.ndarray:
    # Reconstruct the analyzer with identical parameter signatures
    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=sampling_rate,
        lower_cutoff_frequency_hz=lower_cutoff_frequency_hz,
        specified_center_frequency_hz=specified_center_frequency_hz,
        upper_cutoff_frequency_hz=upper_cutoff_frequency_hz,
        filters_per_equivalent_rectangular_bandwidth=filters_per_equivalent_rectangular_bandwidth,
        filter_order=filter_order,
        bandwidth_factor=bandwidth_factor
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
        analyzer.lower_cutoff_frequency_hz,
        analyzer.specified_center_frequency_hz,
        analyzer.upper_cutoff_frequency_hz,
        analyzer.filters_per_equivalent_rectangular_bandwidth,
        analyzer.filter_order,
        analyzer.bandwidth_factor,
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


# Anti-aliasing FIR half-length as a multiple of the up/down ratio. 10 matches
# SciPy's resample_poly default (and the MATLAB reference) for near bit-exact
# agreement. Lower values trade accuracy for speed: empirically 3x costs ~6% of
# each component's energy and drops MATLAB correlation to ~0.99 (the reduction is
# NOT free even for the band-limited subband resamples, contrary to a prior note).
DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR = 10

# Use the Numba polyphase kernels below instead of SciPy's `upfirdn` for the subband
# resampling. Worth ~1.13x on the whole decomposition, because SciPy runs its tap
# loop scalar while these vectorize (see `_numba_polyphase_decimate`).
#
# TRADE-OFF: the kernels are compiled with `fastmath=True`, which is what permits
# LLVM to split the reduction into several partial accumulators -- that IS the
# vectorization. It also reassociates the sum, so results differ from SciPy by a few
# ULP (~1e-15 per subband sample). The decomposition amplifies that to ~2e-11
# absolute, i.e. ~1e-10 of full scale: the per-frame least-squares systems are
# ill-conditioned (regularized at only 1e-15), and `artifacts` is a difference of
# comparable quantities so cancellation shrinks the denominator. Correlation against
# the previous output is 1.0 to all 15 digits and the four quality features agree to
# 10 significant figures.
#
# TO UNDO: set this to False (or `peass.backend_numpy.gammatone.USE_NUMBA_RESAMPLER
# = False` before the first call). That restores the SciPy path, which is bitwise
# identical to the pre-optimization resampler, and takes the end-to-end difference
# from 2.3e-11 to 1.9e-15 -- not to zero, because the block-diagonal projection
# matmul in perform_least_squares_projection also reassociates by ~1 ULP. Revert that
# hunk too if you need exactly zero. Do NOT instead set `fastmath=False` on the
# kernels: that is also bit-exact but ~5% SLOWER than SciPy, so it loses on both
# counts, and it needs the Numba on-disk cache cleared to take effect.
#
# Note this backend was never bitwise reproducible across machines anyway -- any
# BLAS/CPU that reassociates differently gets the same ~1e-10 amplification.
USE_NUMBA_RESAMPLER = True


@lru_cache(maxsize=256)
def get_resample_filter(up: int, down: int, half_length_factor: int = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR) -> tuple:
    g = math.gcd(up, down)
    up_reduced = up // g
    down_reduced = down // g

    max_len = max(up_reduced, down_reduced)

    half_len = half_length_factor * max_len
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


@lru_cache(maxsize=256)
def get_polyphase_branches(up: int, down: int,
                           half_length_factor: int = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR) -> tuple:
    """Reversed full filter (for decimation) and reversed per-phase branches (for
    interpolation), derived from the same Kaiser design as get_resample_filter.

    The branches are returned TRANSPOSED, shape (branch_length, up_reduced) and
    contiguous, so the interpolate kernel's innermost phase loop walks a contiguous
    row per tap (the AXPY layout it vectorizes over)."""
    h_padded, up_reduced, down_reduced, n_pre_remove = get_resample_filter(up, down, half_length_factor)
    num_taps = len(h_padded)
    branch_length = -(-num_taps // up_reduced)
    branches = np.zeros((up_reduced, branch_length))
    for phase in range(up_reduced):
        column = h_padded[phase::up_reduced]
        # right-align so the zero padding lands on the reversed branch's leading taps
        branches[phase, branch_length - len(column):] = column[::-1]
    return (np.ascontiguousarray(h_padded[::-1]), np.ascontiguousarray(branches.T),
            up_reduced, down_reduced, n_pre_remove, num_taps, branch_length)


def _scipy_resample_poly(x, up_reduced, down_reduced, n_pre_remove, h_padded, axis, out_len):
    y = signal.upfirdn(h_padded, x, up_reduced, down_reduced, axis=axis)
    keep = [slice(None)] * y.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove + out_len)
    return y[tuple(keep)]


def resample_output_length(in_len: int, up: int, down: int) -> int:
    """Length `fast_resample_poly` produces along the resampled axis.

    Exposed so a caller can size (or bounds-check) a destination buffer *before*
    the call -- `run_auditory_synthesis_filterbank` uses it to decide whether the
    result fits inside the region it owns and may therefore be written in place.
    `fast_resample_poly` calls this too, so the two cannot drift apart.
    """
    if up == down:
        return int(in_len)
    divisor = math.gcd(int(up), int(down))
    return int(np.ceil(int(in_len) * (int(up) // divisor) / (int(down) // divisor)))


def _resolve_resample_out(x: np.ndarray, out: np.ndarray, axis: int, out_len: int) -> np.ndarray:
    """Validate a caller-supplied `out=` destination and return `out[..., :out_len]`.

    The contract is deliberately narrow, because the point of `out=` is to let a
    caller keep ownership of the samples PAST `out_len`: only the last axis may be
    resampled, `out` must be C-contiguous (so the numba kernels can write into it
    with the same layout they would allocate), and its dtype must be exactly what
    the call would naturally return. Anything else raises rather than silently
    falling back, so a caller cannot lose the in-place write without noticing.
    """
    if axis not in (-1, x.ndim - 1):
        raise ValueError("fast_resample_poly: out= is only supported for axis=-1")
    if out.ndim != x.ndim or out.shape[:-1] != x.shape[:-1]:
        raise ValueError(
            f"fast_resample_poly: out shape {out.shape} incompatible with input shape {x.shape}"
        )
    if out.shape[-1] < out_len:
        raise ValueError(
            f"fast_resample_poly: out last axis is {out.shape[-1]}, needs at least {out_len}"
        )
    expected_dtype = np.dtype(np.complex128) if np.iscomplexobj(x) else np.dtype(np.float64)
    if out.dtype != expected_dtype:
        raise ValueError(
            f"fast_resample_poly: out dtype must be {expected_dtype}, got {out.dtype}"
        )
    if not out.flags.c_contiguous:
        raise ValueError("fast_resample_poly: out must be C-contiguous")
    return out[..., :out_len]


def fast_resample_poly(
        x: np.ndarray,
        up: int,
        down: int,
        axis: int = -1,
        half_length_factor: int = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR,
        out: np.ndarray | None = None
) -> np.ndarray:
    """Polyphase resample of `x` by `up/down` along `axis`.

    With `out=None` a fresh array of length `resample_output_length(...)` is
    returned. With `out` supplied the result is written into `out[..., :out_len]`
    and THAT VIEW is returned; `out` may be longer than the result along the last
    axis and everything from `out_len` onwards is left exactly as the caller left
    it. That is the whole point: it lets a caller scatter into a zero-filled
    buffer whose tail must stay zero. See `_resolve_resample_out` for the
    (deliberately strict) requirements on `out`.
    """
    if up == down:
        if out is None:
            return x.copy()
        destination = _resolve_resample_out(x, out, axis, x.shape[axis])
        destination[...] = x
        return destination

    (reversed_filter, branches_transposed, up_reduced, down_reduced,
     n_pre_remove, num_taps, branch_length) = get_polyphase_branches(up, down, half_length_factor)

    in_len = x.shape[axis]
    out_len = resample_output_length(in_len, up, down)
    destination = None if out is None else _resolve_resample_out(x, out, axis, out_len)

    # Defer to SciPy when Numba is unavailable, when the caller has opted out via
    # USE_NUMBA_RESAMPLER, or for mixed small ratios (e.g. 3/2, 2/3) whose inner loop
    # is too short to pay for the polyphase bookkeeping -- SciPy's Cython loop wins
    # those. This branch is bitwise identical to the pre-optimization implementation.
    if not _HAS_NUMBA or not USE_NUMBA_RESAMPLER or (up_reduced > 1 and down_reduced > 1):
        h_padded = get_resample_filter(up, down, half_length_factor)[0]
        resampled = _scipy_resample_poly(x, up_reduced, down_reduced, n_pre_remove, h_padded, axis, out_len)
        if destination is None:
            return resampled
        # SciPy's upfirdn allocates its own output, so this path cannot write in
        # place; it keeps the copy route instead. `out=` stays honoured (and the
        # tail past out_len untouched) so the two paths remain interchangeable.
        destination[...] = resampled
        return destination

    moved = np.moveaxis(x, axis, -1)
    moved_shape = moved.shape
    flat = moved.reshape(-1, in_len)
    is_complex = np.iscomplexobj(flat)

    if up_reduced == 1:
        pad_left = num_taps - 1
        required = max(pad_left + in_len, (out_len - 1 + n_pre_remove) * down_reduced + num_taps)
    else:
        pad_left = branch_length - 1
        max_position = ((out_len - 1 + n_pre_remove) * down_reduced) // up_reduced
        required = pad_left + max(in_len, max_position + 1)

    def _make_padded(plane):
        # np.empty + zeroed pad edges instead of np.zeros: skips a full memset of
        # the (large) middle region that is immediately overwritten anyway.
        buf = np.empty((flat.shape[0], required), dtype=np.float64)
        buf[:, :pad_left] = 0.0
        buf[:, pad_left + in_len:] = 0.0
        buf[:, pad_left:pad_left + in_len] = plane
        return buf

    # Destination the kernels write into. With `out=` this is the caller's buffer
    # reshaped to (rows, full_width) -- C-contiguity was checked, so the reshape is
    # a view, and the kernels are handed the SAME array layout they would have
    # allocated, only wider. They bound every store by `out_len`, so the caller's
    # columns from `out_len` onwards survive untouched.
    if out is None:
        kernel_output = np.empty((flat.shape[0], out_len),
                                 dtype=np.complex128 if is_complex else np.float64)
    else:
        kernel_output = out.reshape(-1, out.shape[-1])

    # The kernels run on float64 buffers only (that is what lets them lower to
    # ddot / vectorized AXPY); complex input is split into real/imag planes and the
    # complex kernels write complex128 output directly -- no recombine pass here.
    if is_complex:
        padded_real = _make_padded(flat.real)
        padded_imag = _make_padded(flat.imag)
        if up_reduced == 1:
            _numba_polyphase_decimate_complex(
                reversed_filter, padded_real, padded_imag, kernel_output,
                down_reduced, n_pre_remove, out_len, num_taps
            )
        else:
            _numba_polyphase_interpolate_complex(
                branches_transposed, padded_real, padded_imag, kernel_output,
                up_reduced, n_pre_remove, out_len, pad_left, branch_length
            )
    else:
        padded = _make_padded(flat)
        if up_reduced == 1:
            _numba_polyphase_decimate(
                reversed_filter, padded, kernel_output,
                down_reduced, n_pre_remove, out_len, num_taps
            )
        else:
            _numba_polyphase_interpolate(
                branches_transposed, padded, kernel_output,
                up_reduced, n_pre_remove, out_len, pad_left, branch_length
            )

    if destination is not None:
        return destination
    return np.moveaxis(kernel_output.reshape(moved_shape[:-1] + (out_len,)), -1, axis)
