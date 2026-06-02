"""
PEASS Auditory Package - Dau 1996/1997 Psychoacoustic Ear Model

Simulates the transduction process of the inner hair cells and the temporal
adaptation (forward masking) of the auditory nerve. Uses Numba if available,
and fails over gracefully to a SciPy/NumPy native vectorization.
"""

import math

import numpy as np
import scipy.signal as signal

from .config import ModulationProcessingType
from .gammatone import GammatoneAnalyzer
from .gammatone import fast_resample_poly

# -----------------------------------------------------------------------------
# NUMBA JIT COMPILATION (WITH SAFE IMPORT FALLBACK)
# -----------------------------------------------------------------------------
try:
    import numba

    _HAS_NUMBA = True


    @numba.njit(cache=True)
    def _numba_fused_auditory_kernel(
            subband_signals: np.ndarray,
            sampling_frequency_hz: float,
            haircell_filter_gain: float,
            adaptation_bandwidths: np.ndarray,
            absolute_hearing_threshold: float
    ) -> np.ndarray:
        """
        Fused JIT kernel: Half-wave rectification, haircell lowpass,
        and 5-stage non-linear adaptation executing natively in a single pass.
        """
        num_bands, num_samples = subband_signals.shape
        output_signals = np.empty_like(subband_signals)

        stage_thresholds = np.empty(5, dtype=np.float64)
        stage_gains = np.empty(5, dtype=np.float64)

        current_threshold = absolute_hearing_threshold
        for stage_idx in range(5):
            current_threshold = math.sqrt(current_threshold)
            stage_thresholds[stage_idx] = current_threshold
            stage_gains[stage_idx] = math.exp(-math.pi * adaptation_bandwidths[stage_idx] / sampling_frequency_hz)

        haircell_factor = 1.0 - haircell_filter_gain

        adaptation_factors = np.empty_like(stage_thresholds)
        for band_idx in range(num_bands):
            last_haircell_state = 0.0
            for stage_idx in range(5):
                adaptation_factors[stage_idx] = stage_thresholds[stage_idx]
            for sample_idx in range(num_samples):
                # 1. Half-wave rectification
                current_value = subband_signals[band_idx, sample_idx]
                if current_value < 0.0:
                    current_value = 0.0

                # 2. 1 kHz first-order lowpass filter (haircell transduction)
                current_value = haircell_filter_gain * last_haircell_state + haircell_factor * current_value
                last_haircell_state = current_value

                # Minimum hearing threshold floor
                if current_value < absolute_hearing_threshold:
                    current_value = absolute_hearing_threshold

                # 3. Unrolled 5-stage non-linear adaptation loops
                for stage_idx in range(5):
                    gain_value = stage_gains[stage_idx]
                    threshold_value = stage_thresholds[stage_idx]
                    active_factor = adaptation_factors[stage_idx]

                    compressed_value = current_value / active_factor

                    adaptation_factors[stage_idx] = max(
                        (1.0 - gain_value) * compressed_value + gain_value * active_factor,
                        threshold_value
                    )
                    current_value = compressed_value

                output_signals[band_idx, sample_idx] = current_value

        return output_signals


    @numba.njit(cache=True)
    def _numba_haircell_transduction_kernel(
            subband_signals: np.ndarray,
            sampling_frequency_hz: float
    ) -> np.ndarray:
        """
        Dedicated JIT-compiled kernel for standalone haircell transduction.
        """
        num_bands, num_samples = subband_signals.shape
        output_signals = np.empty_like(subband_signals)
        haircell_filter_gain = math.exp(-math.pi * 2000.0 / sampling_frequency_hz)
        haircell_factor = 1.0 - haircell_filter_gain

        for band_idx in range(num_bands):
            last_haircell_state = 0.0
            for sample_idx in range(num_samples):
                current_value = subband_signals[band_idx, sample_idx]
                if current_value < 0.0:
                    current_value = 0.0
                current_value = haircell_filter_gain * last_haircell_state + haircell_factor * current_value
                last_haircell_state = current_value
                output_signals[band_idx, sample_idx] = current_value
        return output_signals


    @numba.njit(cache=True)
    def _numba_adaptation_loops_kernel(
            subband_signals: np.ndarray,
            sampling_frequency_hz: float,
            adaptation_bandwidths: np.ndarray,
            absolute_hearing_threshold: float
    ) -> np.ndarray:
        """
        Dedicated JIT-compiled kernel for standalone adaptation loops.
        """
        num_bands, num_samples = subband_signals.shape
        output_signals = np.empty_like(subband_signals)

        stage_thresholds = np.empty(5, dtype=np.float64)
        stage_gains = np.empty(5, dtype=np.float64)

        current_threshold = absolute_hearing_threshold
        for stage_idx in range(5):
            current_threshold = math.sqrt(current_threshold)
            stage_thresholds[stage_idx] = current_threshold
            stage_gains[stage_idx] = math.exp(-math.pi * adaptation_bandwidths[stage_idx] / sampling_frequency_hz)

        adaptation_factors = np.empty_like(stage_thresholds)
        for band_idx in range(num_bands):
            for stage_idx in range(5):
                adaptation_factors[stage_idx] = stage_thresholds[stage_idx]
            for sample_idx in range(num_samples):
                current_value = subband_signals[band_idx, sample_idx]
                if current_value < absolute_hearing_threshold:
                    current_value = absolute_hearing_threshold

                for stage_idx in range(5):
                    gain_value = stage_gains[stage_idx]
                    threshold_value = stage_thresholds[stage_idx]
                    active_factor = adaptation_factors[stage_idx]
                    compressed_value = current_value / active_factor

                    adaptation_factors[stage_idx] = max(
                        (1.0 - gain_value) * compressed_value + gain_value * active_factor,
                        threshold_value
                    )
                    current_value = compressed_value

                output_signals[band_idx, sample_idx] = current_value

        return output_signals

except ImportError:
    _HAS_NUMBA = False


# -----------------------------------------------------------------------------
# PURE PYTHON/SCIPY FALLBACKS
# -----------------------------------------------------------------------------
def _fallback_adaptation_loops(
        subband_signals: np.ndarray,
        sampling_frequency_hz: float,
        adaptation_bandwidths: np.ndarray,
        absolute_hearing_threshold: float
) -> np.ndarray:
    """
    Pure NumPy fallback for the nonlinear adaptation loops.
    Vectorizes across the frequency bands to mitigate Python loop overhead.
    """
    num_samples = subband_signals.shape[1]
    adapted_signals = np.maximum(subband_signals, absolute_hearing_threshold)
    stage_threshold = absolute_hearing_threshold

    for stage_idx in range(5):
        adaptation_gain = math.exp(-math.pi * adaptation_bandwidths[stage_idx] / sampling_frequency_hz)
        stage_threshold = math.sqrt(stage_threshold)
        divisor_factors = np.full(subband_signals.shape[0], stage_threshold, dtype=np.float64)

        for sample_idx in range(num_samples):
            current_values = adapted_signals[:, sample_idx] / divisor_factors
            adapted_signals[:, sample_idx] = current_values

            divisor_factors = np.maximum(
                (1.0 - adaptation_gain) * current_values + adaptation_gain * divisor_factors,
                stage_threshold
            )

    return adapted_signals


def _fallback_fused_auditory_kernel(
        subband_signals: np.ndarray,
        sampling_frequency_hz: float,
        haircell_filter_gain: float,
        adaptation_bandwidths: np.ndarray,
        absolute_hearing_threshold: float
) -> np.ndarray:
    """
    Pure SciPy/NumPy fallback executing identical math utilizing C-backends.
    """
    # 1. Half-wave rectification
    rectified_signals = np.maximum(subband_signals, 0.0)

    # 2. Haircell 1 kHz first-order lowpass filter
    numerator_coefficients = np.array([1.0 - haircell_filter_gain])
    denominator_coefficients = np.array([1.0, -haircell_filter_gain])
    transduced_signals = signal.lfilter(numerator_coefficients, denominator_coefficients, rectified_signals, axis=-1)

    return _fallback_adaptation_loops(
        transduced_signals, sampling_frequency_hz, adaptation_bandwidths, absolute_hearing_threshold
    )


# -----------------------------------------------------------------------------
# EXPOSED API (STRICT PEP-484 TYPING)
# -----------------------------------------------------------------------------
def simulate_inner_haircell_transduction(
        subband_signals: np.ndarray,
        sampling_frequency_hz: float
) -> np.ndarray:
    """Models the nonlinear mechanical-to-neural transduction of the inner hair cells."""
    if _HAS_NUMBA:
        return _numba_haircell_transduction_kernel(subband_signals, sampling_frequency_hz)
    else:
        rectified_signals = np.maximum(subband_signals, 0.0)
        haircell_filter_gain = math.exp(-math.pi * 2000.0 / sampling_frequency_hz)
        numerator_coefficients = np.array([1.0 - haircell_filter_gain])
        denominator_coefficients = np.array([1.0, -haircell_filter_gain])
        return signal.lfilter(numerator_coefficients, denominator_coefficients, rectified_signals, axis=-1)


def simulate_auditory_nerve_adaptation(
        subband_signals: np.ndarray,
        sampling_frequency_hz: float
) -> np.ndarray:
    """Simulates the physiological adaptive properties of the auditory nerve."""
    decibel_range = 100.0
    absolute_hearing_threshold = 10.0 ** (-decibel_range / 20.0)
    adaptation_loop_bandwidths = 1.0 / (np.pi * np.array([0.005, 0.05, 0.129, 0.253, 0.5]))

    if _HAS_NUMBA:
        adapted_signals = _numba_adaptation_loops_kernel(
            subband_signals, sampling_frequency_hz, adaptation_loop_bandwidths, absolute_hearing_threshold
        )
    else:
        adapted_signals = _fallback_adaptation_loops(
            subband_signals, sampling_frequency_hz, adaptation_loop_bandwidths, absolute_hearing_threshold
        )

    final_threshold = absolute_hearing_threshold
    for _ in range(5):
        final_threshold = math.sqrt(final_threshold)

    return (decibel_range / (1.0 - final_threshold)) * (adapted_signals - final_threshold)


def generate_auditory_internal_representation(
        signal_data: np.ndarray,
        sampling_frequency_hz: float,
        modulation_processing_type: ModulationProcessingType = ModulationProcessingType.LOWPASS
) -> tuple[np.ndarray, float]:
    """Generates the 3D internal auditory representation of a signal."""
    if len(signal_data.shape) > 1:
        if signal_data.shape[0] < signal_data.shape[1]:
            signal_data = signal_data.T
        signal_data = signal_data.ravel()

    # Model input scaling (1.0 becomes 100 dB SPL)
    scaled_signal_data = 10.0 * signal_data

    minimum_frequency = 235.0
    maximum_frequency = min(0.5 * sampling_frequency_hz, 14500.0)

    # Decimate using polyphase FIR (avoids global FFT memory spikes)
    if sampling_frequency_hz < 3.0 * maximum_frequency:
        new_sampling_frequency = int(round(1.5 * sampling_frequency_hz))
        scaled_signal_data = fast_resample_poly(
            scaled_signal_data, new_sampling_frequency, int(sampling_frequency_hz)
        )
        sampling_frequency_hz = float(new_sampling_frequency)

    # 1. Gammatone Analysis Filterbank
    analyzer = GammatoneAnalyzer(sampling_frequency_hz, minimum_frequency, 1000.0, maximum_frequency, 1.0)
    subbands = np.real(analyzer.process(scaled_signal_data))

    # 2 & 3. Fused IHC Transduction and Nerve Adaptation
    haircell_filter_gain = math.exp(-math.pi * 2000.0 / sampling_frequency_hz)
    decibel_range = 100.0
    absolute_hearing_threshold = 10.0 ** (-decibel_range / 20.0)
    adaptation_loop_bandwidths = 1.0 / (np.pi * np.array([0.005, 0.05, 0.129, 0.253, 0.5]))

    if _HAS_NUMBA:
        adapted_signals = _numba_fused_auditory_kernel(
            subbands, sampling_frequency_hz, haircell_filter_gain,
            adaptation_loop_bandwidths, absolute_hearing_threshold
        )
    else:
        adapted_signals = _fallback_fused_auditory_kernel(
            subbands, sampling_frequency_hz, haircell_filter_gain,
            adaptation_loop_bandwidths, absolute_hearing_threshold
        )

    # Global dB offset scaling
    final_threshold = absolute_hearing_threshold
    for _ in range(5):
        final_threshold = math.sqrt(final_threshold)

    adapted_signals = (decibel_range / (1.0 - final_threshold)) * (adapted_signals - final_threshold)

    # 4. Modulation Filtering & Polyphase Decimation
    if modulation_processing_type == ModulationProcessingType.FILTERBANK:
        downsampled_adapted = fast_resample_poly(adapted_signals, 800, int(sampling_frequency_hz), axis=-1)
        sampling_frequency_hz = 800.0
        modulation_center_frequencies = np.concatenate(([0.0, 5.0], 10.0 * (5.0 / 3.0) ** np.arange(6)))
        modulation_bandwidths = np.concatenate(([5.0, 5.0], 5.0 * (5.0 / 3.0) ** np.arange(6)))
    elif modulation_processing_type == ModulationProcessingType.LOWPASS:
        downsampled_adapted = fast_resample_poly(adapted_signals, 100, int(sampling_frequency_hz), axis=-1)
        sampling_frequency_hz = 100.0
        modulation_center_frequencies = np.array([0.0])
        modulation_bandwidths = np.array([15.92])
    else:
        raise RuntimeError(f'Unknown {modulation_processing_type=}')

    num_bands = adapted_signals.shape[0]
    num_modulations = len(modulation_center_frequencies)
    num_samples = downsampled_adapted.shape[1]

    internal_representation = np.zeros((num_bands, num_samples, num_modulations), dtype=complex)

    for mod_idx in range(num_modulations):
        filter_gain = math.exp(-math.pi * modulation_bandwidths[mod_idx] / sampling_frequency_hz)
        numerator_coeffs = np.array([1.0 - filter_gain])
        denominator_coeffs = np.array([
            1.0,
            -filter_gain * np.exp(2j * np.pi * modulation_center_frequencies[mod_idx] / sampling_frequency_hz)
        ])

        # Offloaded to SciPy C-backend
        internal_representation[:, :, mod_idx] = signal.lfilter(
            numerator_coeffs, denominator_coeffs, downsampled_adapted, axis=-1
        )

    channels_above_10_hz = (modulation_center_frequencies > 10.0)
    internal_representation[:, :, ~channels_above_10_hz] = np.real(
        internal_representation[:, :, ~channels_above_10_hz]
    )
    internal_representation[:, :, channels_above_10_hz] = np.abs(
        internal_representation[:, :, channels_above_10_hz]
    )

    # Cast to real float64 since all imaginary parts have been discarded
    return np.real(internal_representation), sampling_frequency_hz
