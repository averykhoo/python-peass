"""
PEASS Decomposition Package - Least-Squares Distortion Decomposer

Decomposes the separation error of a source estimate into Target distortion,
Interference, and Artifacts. Refactored using stride tricks, LAPACK posv solves,
and zero-copy arrays.
"""
import pathlib
import warnings
from functools import lru_cache

import numpy as np
import scipy.linalg as linalg
import scipy.signal as signal
import soundfile as sf
from scipy.linalg import LinAlgWarning

from peass.config import DecomposedFilePaths
from peass.config import DecomposedWaveforms
from peass.config import DecompositionConfiguration
from peass.config import DecompositionResult
from .gammatone import DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
from .gammatone import GammatoneAnalyzer
from .gammatone import GammatoneSynthesizer
from .gammatone import calculate_equivalent_rectangular_bandwidth
from .gammatone import fast_resample_poly


def validate_and_normalize_audio(
        data: np.ndarray,
        sampling_frequency_hz: float,
        name: str = "audio_data"
) -> np.ndarray:
    """
    Validates and normalizes 1D/2D NumPy audio arrays to strictly enforce
    the SciPy/NumPy layout convention: (samples, channels).

    Enforces:
    - Standard 2D shape (samples, channels).
    - Maximum of 32 channels.
    - Minimum duration of 50ms to safely satisfy the boundary shading
      windows and subband frame decimation limits.
    """
    # Force at least 2D array representation
    normalized = np.atleast_2d(data)

    # Standardize 1D signals (1, samples) to column vectors (samples, 1)
    if data.ndim == 1:
        normalized = normalized.T

    num_samples, num_channels = normalized.shape

    # 1. Enforce strict spatial channel limit (< 32 channels)
    if num_channels > 32:
        raise ValueError(
            f"Layout violation for '{name}'. Expected (samples, channels) "
            f"with channels <= 32. Detected shape: {data.shape}. "
            f"If your signal is (channels, samples), please transpose your input array."
        )

    # Dynamically enforce a minimum physical duration of 50 ms
    min_duration_ms = 50
    min_samples = int(min_duration_ms * sampling_frequency_hz / 1000)
    if num_samples < min_samples:
        raise ValueError(
            f"Signal duration for '{name}' is too short ({num_samples} samples). "
            f"PEASS requires a minimum of {min_samples} samples to perform "
            f"the subband least-squares and overlap-add decomposition safely."
        )

    return normalized


def perform_least_squares_projection(
        source_estimates: np.ndarray,
        true_sources: np.ndarray,
        filter_half_length: int,
        analysis_window: np.ndarray
) -> np.ndarray:
    r"""
    Weighted least-squares projection of source estimate onto source subspaces.
    Executes entirely in BLAS/LAPACK without explicitly allocating massive memory blocks.
    """
    filter_length = 2 * filter_half_length + 1
    num_sources = true_sources.shape[1]
    num_samples = source_estimates.shape[0]

    # --- SILENCE BYPASS OPTIMIZATION ---
    # If reference sources are silent in this frame, bypass the solver entirely
    if np.iscomplexobj(true_sources):
        source_energy = np.sum(true_sources.real ** 2 + true_sources.imag ** 2)
    else:
        source_energy = np.sum(true_sources ** 2)
    if source_energy < 1e-13:
        return np.zeros((num_samples, source_estimates.shape[1], num_sources), dtype=source_estimates.dtype)

    # Stride tricks for zero-copy view of the Toeplitz bands
    strided_views = []
    for source_idx in range(num_sources):
        source_signal = true_sources[:, source_idx]
        shape = (num_samples, filter_length)
        strides = (source_signal.strides[0], source_signal.strides[0])

        view = np.lib.stride_tricks.as_strided(source_signal, shape=shape, strides=strides, writeable=False)
        # Reverse each row to match toeplitz(col, row) semantics perfectly
        strided_views.append(view[:, ::-1])

    # Direct horizontal stack to C-contiguous buffer
    toeplitz_matrix = np.hstack(strided_views)

    # --- SYSTEM BENCHMARK: OLD METHOD (Commented out for future comparison) ---
    # weighted_sources = analysis_window[:, np.newaxis] * toeplitz_matrix
    # weighted_estimates = analysis_window[:, np.newaxis] * source_estimates
    # gram_matrix = weighted_sources.conj().T @ weighted_sources
    # rhs_vector = weighted_sources.conj().T @ weighted_estimates

    # --- SYSTEM BENCHMARK: NEW METHOD (Memory-efficient Gram calculation) ---
    # Avoids allocating massive temporary weighted_sources / weighted_estimates matrices in memory
    window_sq = (analysis_window ** 2)[:, np.newaxis]
    gram_matrix = toeplitz_matrix.conj().T @ (window_sq * toeplitz_matrix)
    rhs_vector = toeplitz_matrix.conj().T @ (window_sq * source_estimates)

    # In-place diagonal regularization
    regularization_lambda = 10.0 ** -15
    gram_matrix.flat[::gram_matrix.shape[0] + 1] += regularization_lambda

    # Promote LinAlgWarning to an exception locally so the fallback is triggered
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        # Offload directly to LAPACK posv (Cholesky solve in compiled C/Fortran)
        try:
            projection_weights = linalg.solve(gram_matrix, rhs_vector, assume_a='pos')
        except (linalg.LinAlgError, ValueError, LinAlgWarning):
            # Fallback to pseudo-inverse if singular or highly ill-conditioned
            weighted_toeplitz = toeplitz_matrix * analysis_window[:, np.newaxis]
            weighted_estimates = source_estimates * analysis_window[:, np.newaxis]
            projection_weights = linalg.pinv(weighted_toeplitz) @ weighted_estimates

    projections = np.zeros(
        (num_samples, source_estimates.shape[1], num_sources),
        dtype=source_estimates.dtype
    )
    weighted_diagonal = analysis_window[:, np.newaxis]
    for source_idx in range(num_sources):
        projections[:, :, source_idx] = weighted_diagonal * (
                toeplitz_matrix[:, source_idx * filter_length: (source_idx + 1) * filter_length] @
                projection_weights[source_idx * filter_length: (source_idx + 1) * filter_length, :]
        )

    return projections


def perform_time_varying_least_squares_projection(
        source_estimates: np.ndarray,
        true_sources: np.ndarray,
        filter_length: int,
        window_length: int,
        hop_size: int
) -> np.ndarray:
    r"""
    Time-varying least-squares subband decomposer.
    """
    filter_half_length = (filter_length - 1) // 2
    if (filter_length - 1) % 2 != 0:
        raise ValueError("Filter length must be an odd integer.")

    pad_length = filter_length - 1 + window_length - 1
    true_sources = np.pad(true_sources, ((0, pad_length), (0, 0)), mode='constant')
    source_estimates = np.pad(source_estimates, ((0, pad_length), (0, 0)), mode='constant')

    total_samples, num_sources = true_sources.shape
    num_channels = source_estimates.shape[1]

    hann_window = signal.windows.hann(window_length, sym=False)
    analysis_window = np.sqrt(np.flipud(hann_window))
    synthesis_window = np.sqrt(np.flipud(hann_window))

    synthesis_weights = np.zeros((window_length, num_channels, num_sources))
    for channel_idx in range(num_channels):
        for source_idx in range(num_sources):
            synthesis_weights[:, channel_idx, source_idx] = synthesis_window

    window_begin = 0
    window_end = window_begin + window_length

    projections_accumulation = np.zeros(
        (total_samples, num_channels, num_sources),
        dtype=true_sources.dtype
    )
    window_gain_accumulation = np.zeros((total_samples, 1))

    while window_end - window_length / 2.0 <= projections_accumulation.shape[0] - window_length + 1:
        frame_estimates = source_estimates[window_begin:window_end, :]

        source_window_start = window_begin - filter_half_length
        source_window_end = window_end + filter_half_length
        pad_left = max(0, -source_window_start)
        pad_right = max(0, source_window_end - true_sources.shape[0])
        slice_start = max(0, source_window_start)
        slice_end = min(true_sources.shape[0], source_window_end)

        frame_sources_slice = true_sources[slice_start:slice_end, :]
        frame_sources = np.vstack([
            np.zeros((pad_left, num_sources), dtype=true_sources.dtype),
            frame_sources_slice,
            np.zeros((pad_right, num_sources), dtype=true_sources.dtype)
        ])

        frame_projections = perform_least_squares_projection(
            frame_estimates, frame_sources, filter_half_length, analysis_window
        )

        projections_accumulation[window_begin:window_end, :, :] += (
                frame_projections[:window_length, :, :] * synthesis_weights
        )
        window_gain_accumulation[window_begin:window_end, 0] += synthesis_window * analysis_window

        window_begin += hop_size
        window_end += hop_size

    valid_indices = (window_gain_accumulation[:, 0] != 0)
    for source_idx in range(num_sources):
        projections_accumulation[valid_indices, :, source_idx] /= window_gain_accumulation[valid_indices, :]

    return projections_accumulation[:-(window_length - 1), :, :]


def extract_target_spatial_distortion_interference_artifacts(
        true_sources: np.ndarray,
        source_estimates: np.ndarray,
        filter_length: int,
        window_length: int,
        hop_size: int,
        use_two_stage_projection: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Splits multi-source signal mixtures into physical sub-components.
    """
    total_samples, num_channels, num_sources = true_sources.shape
    num_estimates = source_estimates.shape[2] if len(source_estimates.shape) > 2 else 1
    if len(source_estimates.shape) == 2:
        source_estimates = source_estimates[:, :, np.newaxis]

    sources_reshaped = true_sources.reshape((total_samples, num_sources * num_channels), order='F')
    estimates_reshaped = source_estimates.reshape((total_samples, num_estimates * num_channels), order='F')

    projections_all = perform_time_varying_least_squares_projection(
        estimates_reshaped, sources_reshaped, filter_length, window_length, hop_size
    )

    projected_signals = np.zeros(
        (total_samples, num_channels * num_estimates, num_sources),
        dtype=true_sources.dtype
    )
    for source_idx in range(num_sources):
        start_channel_idx = source_idx * num_channels
        end_channel_idx = (source_idx + 1) * num_channels
        projected_signals[:, :, source_idx] = np.sum(
            projections_all[:total_samples, :, start_channel_idx:end_channel_idx], axis=2
        )

    spatial_distortion = np.zeros((total_samples, num_estimates * num_channels), dtype=source_estimates.dtype)
    if use_two_stage_projection:
        for estimate_idx in range(num_estimates):
            start_estimate_idx = estimate_idx * num_channels
            end_estimate_idx = (estimate_idx + 1) * num_channels
            spatial_projection = perform_time_varying_least_squares_projection(
                estimates_reshaped[:, start_estimate_idx:end_estimate_idx],
                sources_reshaped[:, :num_channels],
                filter_length, window_length, hop_size
            )
            spatial_distortion[:, start_estimate_idx:end_estimate_idx] = np.sum(
                spatial_projection[:total_samples, :, :], axis=2
            )

    true_reference = np.zeros((total_samples, num_channels * num_estimates), dtype=true_sources.dtype)
    for estimate_idx in range(num_estimates):
        start_estimate_idx = estimate_idx * num_channels
        end_estimate_idx = (estimate_idx + 1) * num_channels
        true_reference[:, start_estimate_idx:end_estimate_idx] = sources_reshaped[:, :num_channels]

    if use_two_stage_projection:
        spatial_distortion = spatial_distortion - true_reference
    else:
        spatial_distortion = (
                projected_signals[:, :, :num_estimates].reshape((total_samples, num_estimates * num_channels),
                                                                order='F') -
                true_reference
        )

    interference = np.sum(projected_signals, axis=2) - spatial_distortion - true_reference
    artifacts = estimates_reshaped - true_reference - spatial_distortion - interference

    true_reference_3d = true_reference.reshape((total_samples, num_channels, num_estimates), order='F')
    spatial_distortion_3d = spatial_distortion.reshape((total_samples, num_channels, num_estimates), order='F')
    interference_3d = interference.reshape((total_samples, num_channels, num_estimates), order='F')
    artifacts_3d = artifacts.reshape((total_samples, num_channels, num_estimates), order='F')

    return true_reference_3d, spatial_distortion_3d, interference_3d, artifacts_3d


def run_auditory_analysis_filterbank(
        signal_waveform: np.ndarray,
        sampling_frequency_hz: float,
        modulation_matrix: np.ndarray | None = None,
        half_length_factor: int = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
) -> tuple[list[np.ndarray], GammatoneAnalyzer, np.ndarray]:
    """Helper executing Gammatone Analysis subband decomposition."""
    minimum_frequency = 20.0
    maximum_frequency = sampling_frequency_hz / 2.0
    base_frequency = 1000.0
    filters_per_erb = 1.0

    original_fs = sampling_frequency_hz
    # The PEASS reference always upsamples the analysis signal by 1.5x so the
    # Gammatone filters near the original Nyquist are well resolved. The original
    # guard `fs/2 < 1.5*(fs/2)` is a tautology (always true), so this is
    # unconditional; kept explicit here for clarity.
    new_fs = int(round(1.5 * sampling_frequency_hz))
    signal_waveform = fast_resample_poly(
        signal_waveform, new_fs, int(sampling_frequency_hz), half_length_factor=half_length_factor
    )
    sampling_frequency_hz = new_fs

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz, minimum_frequency, base_frequency, maximum_frequency, filters_per_erb
    )
    analyzer.original_sampling_frequency_hz = original_fs

    subbands_output = analyzer.process(signal_waveform)
    num_bands = subbands_output.shape[0]

    if modulation_matrix is None:
        time_steps = np.arange(subbands_output.shape[1])
        center_frequencies = analyzer.center_frequencies[:, np.newaxis]
        modulation_matrix = np.exp(-2j * np.pi / sampling_frequency_hz * center_frequencies * time_steps)

    subbands_output = subbands_output * modulation_matrix

    equivalent_bandwidths = calculate_equivalent_rectangular_bandwidth(analyzer.center_frequencies)
    decimation_alpha = 2.0
    decimation_factors = np.maximum(
        1, np.floor(sampling_frequency_hz / (equivalent_bandwidths * decimation_alpha))
    ).astype(int)

    analyzer.decimation_factors = decimation_factors
    analyzer.sampling_frequency_hz = sampling_frequency_hz
    analyzer.bandwidths = equivalent_bandwidths

    # --- VECTORIZED 2D BLOCK RESAMPLING (NEW METHOD) ---
    # Group bands with identical decimation factors and process them in contiguous 2D blocks
    decimated_bands = [None] * num_bands
    unique_factors = np.unique(decimation_factors)

    for factor in unique_factors:
        band_indices = np.where(decimation_factors == factor)[0]
        block = subbands_output[band_indices, :]

        # Vectorized 2D resampling along the last axis (-1) in a single C-backend execution pass
        resampled_block = fast_resample_poly(block, 1, factor, axis=-1, half_length_factor=half_length_factor)

        for idx, band_idx in enumerate(band_indices):
            decimated_bands[band_idx] = resampled_block[idx, :]

    return decimated_bands, analyzer, modulation_matrix


# -----------------------------------------------------------------------------
# SYNTHESIS MODULATION MATRIX CACHE (Using functools.lru_cache)
# -----------------------------------------------------------------------------

# Bounded cache: keyed on signal length, so each distinct length holds a full
# (num_bands x max_samples) complex matrix. Cap it so long-lived processes that
# score many different-length signals don't accumulate unbounded memory.
@lru_cache(maxsize=8)
def _get_synthesis_modulation_matrix_cached(
        sampling_frequency: float,
        max_samples_length: int,
        center_frequencies_tuple: tuple
) -> np.ndarray:
    center_frequencies = np.array(center_frequencies_tuple)
    time_steps = np.arange(max_samples_length)
    return np.exp(2j * np.pi / sampling_frequency * center_frequencies[:, np.newaxis] * time_steps)


def get_synthesis_modulation_matrix(
        sampling_frequency: float,
        max_samples_length: int,
        center_frequencies: np.ndarray
) -> np.ndarray:
    """Helper to lazily compute and cache the synthesis modulation matrix."""
    return _get_synthesis_modulation_matrix_cached(
        sampling_frequency,
        max_samples_length,
        tuple(center_frequencies)
    )


def run_auditory_synthesis_filterbank(
        subband_list: list,
        analyzer: GammatoneAnalyzer,
        half_length_factor: int = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
) -> tuple[np.ndarray, GammatoneSynthesizer]:
    """Helper executing Gammatone synthesis reconstruction."""
    num_bands = len(subband_list)
    sampling_frequency = analyzer.sampling_frequency_hz

    max_samples_length = max(
        len(subband_list[band_idx]) * analyzer.decimation_factors[band_idx] for band_idx in range(num_bands)
    )
    processed_subbands = np.zeros((num_bands, max_samples_length), dtype=complex)

    # --- VECTORIZED 2D BLOCK UPSAMPLING ---
    unique_factors = np.unique(analyzer.decimation_factors)

    for factor in unique_factors:
        band_indices = np.where(analyzer.decimation_factors == factor)[0]
        # Faster stacking than vstack
        block = np.array([subband_list[b] for b in band_indices])

        # Vectorized 2D upsampling along axis=-1
        upsampled_block = fast_resample_poly(block, factor, 1, axis=-1, half_length_factor=half_length_factor)

        for idx, band_idx in enumerate(band_indices):
            target_length = len(subband_list[band_idx]) * factor
            upsampled_subband = upsampled_block[idx, :]
            curr_len = len(upsampled_subband)

            # Avoid np.pad allocations; processed_subbands is already zero-initialized
            if curr_len == target_length:
                processed_subbands[band_idx, :target_length] = upsampled_subband
            elif curr_len > target_length:
                processed_subbands[band_idx, :target_length] = upsampled_subband[:target_length]
            else:
                processed_subbands[band_idx, :curr_len] = upsampled_subband

    # Retrieve cached modulation matrix to bypass np.exp re-calculations
    processed_subbands = processed_subbands * get_synthesis_modulation_matrix(
        sampling_frequency, max_samples_length, analyzer.center_frequencies
    )

    desired_delay_seconds = 1000.0 / sampling_frequency
    synthesizer = GammatoneSynthesizer(analyzer, desired_delay_seconds)

    reconstructed_signal = synthesizer.process(processed_subbands)

    original_sampling_frequency = analyzer.original_sampling_frequency_hz
    reconstructed_signal = fast_resample_poly(
        reconstructed_signal,
        int(original_sampling_frequency),
        int(sampling_frequency),
        half_length_factor=half_length_factor
    )
    delay_offset_samples = int(round(desired_delay_seconds * original_sampling_frequency))
    reconstructed_signal = reconstructed_signal[delay_offset_samples:]

    return reconstructed_signal, synthesizer


def decompose_distortion_components(
        source_files: list[str | pathlib.Path | np.ndarray],
        estimate_file: str | pathlib.Path | np.ndarray,
        configuration: DecompositionConfiguration | None = None,
        sampling_frequency_hz: float | None = None
) -> DecompositionResult:
    r"""
    Decomposes an estimated source signal into physical distortion components.
    """
    if configuration is None:
        configuration = DecompositionConfiguration()

    if not source_files:
        raise ValueError("source_files list cannot be empty.")

    is_file_mode = isinstance(estimate_file, str | pathlib.Path)

    if is_file_mode:
        # File-based mode (handled by soundfile, which defaults to samples-first)
        estimate_audio_data, sampling_frequency_hz = sf.read(estimate_file)
        # Normalize file inputs
        estimate_audio_data = validate_and_normalize_audio(
            estimate_audio_data, sampling_frequency_hz, name="estimate_file"
        )

        source_data_list = []
        for idx, source_path in enumerate(source_files):
            if not isinstance(source_path, (str, pathlib.Path)):
                raise ValueError("All source inputs must be file paths in file-based mode.")
            data, source_fs = sf.read(source_path)
            if source_fs != sampling_frequency_hz:
                raise ValueError("Sampling rates of all files must match.")

            data = validate_and_normalize_audio(
                data, sampling_frequency_hz, name=f"source_files[{idx}]"
            )
            source_data_list.append(data)
    else:
        # Array-based mode (enforces strict NumPy/SciPy convention)
        if sampling_frequency_hz is None:
            raise ValueError("In-memory mode requires explicit sampling rate 'sampling_frequency_hz'.")

        estimate_audio_data = validate_and_normalize_audio(
            estimate_file, sampling_frequency_hz, name="estimate_file"
        )

        source_data_list = []
        for idx, source_array in enumerate(source_files):
            if isinstance(source_array, (str, pathlib.Path)):
                raise ValueError("All source inputs must be numpy arrays in array-based mode.")

            data = validate_and_normalize_audio(
                source_array, sampling_frequency_hz, name=f"source_files[{idx}]"
            )
            source_data_list.append(data)

    number_of_sources = len(source_data_list)
    original_samples_length = estimate_audio_data.shape[0]
    number_of_channels = estimate_audio_data.shape[1]

    for source_data in source_data_list:
        if source_data.shape != estimate_audio_data.shape:
            raise ValueError("All source signals must be of matching dimensions.")

    def apply_window_shading(sig: np.ndarray, fs: float, shade_in: float, shade_out: float) -> np.ndarray:
        shaded_signal = sig.copy()
        num_samples = shaded_signal.shape[0]

        fade_in_samples = int(round(shade_in / 1000.0 * fs)) if shade_in > 0 else 0
        fade_out_samples = int(round(shade_out / 1000.0 * fs)) if shade_out > 0 else 0

        # Explicitly validate signal length against configured shading windows
        if fade_in_samples + fade_out_samples > num_samples:
            raise ValueError(
                f"Combined shading length ({fade_in_samples + fade_out_samples} samples) "
                f"exceeds the signal length ({num_samples} samples)."
            )

        if fade_in_samples > 1:
            time_steps = np.arange(fade_in_samples)
            shade_in_window = 0.5 - 0.5 * np.cos(np.pi * time_steps / (fade_in_samples - 1))
            # Vectorized channel multiplication
            shaded_signal[:fade_in_samples, :] *= shade_in_window[:, np.newaxis]

        if fade_out_samples > 1:
            time_steps = np.arange(fade_out_samples)
            shade_out_window = 0.5 + 0.5 * np.cos(np.pi * time_steps / (fade_out_samples - 1))
            # Vectorized channel multiplication
            shaded_signal[-fade_out_samples:, :] *= shade_out_window[:, np.newaxis]

        return shaded_signal

    shaded_sources = [
        apply_window_shading(
            src, sampling_frequency_hz, configuration.shade_in_milliseconds, configuration.shade_out_milliseconds
        ) for src in source_data_list
    ]
    shaded_estimate = apply_window_shading(
        estimate_audio_data,
        sampling_frequency_hz,
        configuration.shade_in_milliseconds,
        configuration.shade_out_milliseconds
    )

    subband_source_signals = [[None for _ in range(number_of_channels)] for _ in range(number_of_sources)]
    modulation_matrix = None
    analyzer_instance = None
    resample_factor = configuration.resample_filter_half_length_factor

    for source_idx in range(number_of_sources):
        for channel_idx in range(number_of_channels):
            subband_source_signals[source_idx][channel_idx], analyzer_instance, modulation_matrix = (
                run_auditory_analysis_filterbank(
                    shaded_sources[source_idx][:, channel_idx], sampling_frequency_hz, modulation_matrix,
                    half_length_factor=resample_factor
                )
            )

    subband_estimate_signals = [None for _ in range(number_of_channels)]
    for channel_idx in range(number_of_channels):
        subband_estimate_signals[channel_idx], analyzer_instance, _ = run_auditory_analysis_filterbank(
            shaded_estimate[:, channel_idx], sampling_frequency_hz, modulation_matrix,
            half_length_factor=resample_factor
        )

    number_of_bands = len(subband_source_signals[0][0])
    subband_sources_composite = []
    subband_estimates_composite = []

    for band_idx in range(number_of_bands):
        # Fully vectorized block construction using transpose, stacking, and list comprehensions
        estimates_block = np.array([subband_estimate_signals[c][band_idx] for c in range(number_of_channels)]).T[
            :, :, np.newaxis]
        sources_block = np.array([
            [subband_source_signals[s][c][band_idx] for s in range(number_of_sources)]
            for c in range(number_of_channels)
        ]).transpose(2, 0, 1)

        subband_sources_composite.append(sources_block)
        subband_estimates_composite.append(estimates_block)

    reference_frequency = 1000.0
    reference_frame_length = configuration.frame_length_seconds
    reference_hop_length = reference_frame_length / 4.0
    f_ref_idx = np.argmin(np.abs(analyzer_instance.center_frequencies - reference_frequency))
    reference_bandwidth = analyzer_instance.bandwidths[f_ref_idx]

    decimated_sampling_frequency = analyzer_instance.sampling_frequency_hz / analyzer_instance.decimation_factors

    reference_filter_length = min(
        configuration.filter_length_seconds, reference_frame_length / number_of_channels / number_of_sources / 3.0
    )

    bandwidth_ratios = reference_bandwidth / analyzer_instance.bandwidths * decimated_sampling_frequency
    filter_lengths = np.maximum(3, 2 * np.round((reference_filter_length * bandwidth_ratios - 1) / 2.0) + 1).astype(int)
    window_lengths = np.maximum(3, np.round(reference_frame_length * bandwidth_ratios)).astype(int)
    hop_sizes = np.maximum(1, np.round(reference_hop_length * bandwidth_ratios)).astype(int)

    decomposed_subband_true = []
    decomposed_subband_target_distortion = []
    decomposed_subband_interference = []
    decomposed_subband_artifacts = []
    for band_idx in range(number_of_bands):
        true_b, target_dist_b, interference_b, artifacts_b = (
            extract_target_spatial_distortion_interference_artifacts(
                subband_sources_composite[band_idx],
                subband_estimates_composite[band_idx],
                filter_lengths[band_idx],
                window_lengths[band_idx],
                hop_sizes[band_idx],
                use_two_stage_projection=configuration.use_two_stage_projection
            )
        )
        decomposed_subband_true.append(true_b)
        decomposed_subband_target_distortion.append(target_dist_b)
        decomposed_subband_interference.append(interference_b)
        decomposed_subband_artifacts.append(artifacts_b)

    # Clean and vectorized subband list reformatting
    reformatted_subband_true = [
        [decomposed_subband_true[b][:, c, 0] for b in range(number_of_bands)]
        for c in range(number_of_channels)
    ]
    reformatted_subband_target_distortion = [
        [decomposed_subband_target_distortion[b][:, c, 0] for b in range(number_of_bands)]
        for c in range(number_of_channels)
    ]
    reformatted_subband_interference = [
        [decomposed_subband_interference[b][:, c, 0] for b in range(number_of_bands)]
        for c in range(number_of_channels)
    ]
    reformatted_subband_artifacts = [
        [decomposed_subband_artifacts[b][:, c, 0] for b in range(number_of_bands)]
        for c in range(number_of_channels)
    ]

    synthesized_true_target = np.zeros((original_samples_length, number_of_channels))
    synthesized_target_distortion = np.zeros((original_samples_length, number_of_channels))
    synthesized_interference = np.zeros((original_samples_length, number_of_channels))
    synthesized_artifacts = np.zeros((original_samples_length, number_of_channels))

    def clip_or_pad_to_target_length(signal_array: np.ndarray, target_length: int) -> np.ndarray:
        if len(signal_array) >= target_length:
            return signal_array[:target_length]
        return np.pad(signal_array, (0, target_length - len(signal_array)), mode='constant')

    for channel_idx in range(number_of_channels):
        synth_t, _ = run_auditory_synthesis_filterbank(
            reformatted_subband_true[channel_idx], analyzer_instance, half_length_factor=resample_factor
        )
        synth_td, _ = run_auditory_synthesis_filterbank(
            reformatted_subband_target_distortion[channel_idx], analyzer_instance, half_length_factor=resample_factor
        )
        synth_i, _ = run_auditory_synthesis_filterbank(
            reformatted_subband_interference[channel_idx], analyzer_instance, half_length_factor=resample_factor
        )
        synth_a, _ = run_auditory_synthesis_filterbank(
            reformatted_subband_artifacts[channel_idx], analyzer_instance, half_length_factor=resample_factor
        )

        synthesized_true_target[:, channel_idx] = clip_or_pad_to_target_length(synth_t, original_samples_length)
        synthesized_target_distortion[:, channel_idx] = (
            clip_or_pad_to_target_length(synth_td, original_samples_length)
        )
        synthesized_interference[:, channel_idx] = clip_or_pad_to_target_length(synth_i, original_samples_length)
        synthesized_artifacts[:, channel_idx] = clip_or_pad_to_target_length(synth_a, original_samples_length)

    waveforms = DecomposedWaveforms(
        true_target=synthesized_true_target,
        target_distortion=synthesized_target_distortion,
        interference=synthesized_interference,
        artifacts=synthesized_artifacts
    )

    if is_file_mode:
        destination_path = pathlib.Path(configuration.destination_directory)
        destination_path.mkdir(parents=True, exist_ok=True)
        estimate_stem = pathlib.Path(estimate_file).stem
        out_filenames = DecomposedFilePaths(
            true_target=str(destination_path / f"{estimate_stem}_true.wav"),
            target_distortion=str(destination_path / f"{estimate_stem}_eTarget.wav"),
            interference=str(destination_path / f"{estimate_stem}_eInterf.wav"),
            artifacts=str(destination_path / f"{estimate_stem}_eArtif.wav")
        )
        sf.write(out_filenames.true_target, synthesized_true_target, int(sampling_frequency_hz))
        sf.write(out_filenames.target_distortion, synthesized_target_distortion, int(sampling_frequency_hz))
        sf.write(out_filenames.interference, synthesized_interference, int(sampling_frequency_hz))
        sf.write(out_filenames.artifacts, synthesized_artifacts, int(sampling_frequency_hz))
        return DecompositionResult(waveforms=waveforms, file_paths=out_filenames)

    return DecompositionResult(waveforms=waveforms, file_paths=None)
