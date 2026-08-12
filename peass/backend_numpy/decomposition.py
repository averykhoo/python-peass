"""
PEASS Decomposition Package - Least-Squares Distortion Decomposer

Decomposes the separation error of a source estimate into Target distortion,
Interference, and Artifacts. Refactored using stride tricks, LAPACK posv solves,
and zero-copy arrays.
"""
import pathlib
from functools import lru_cache

import numpy as np
import scipy.linalg as linalg
import scipy.signal as signal
import soundfile as sf

from peass.config import DecomposedFilePaths
from peass.config import DecomposedWaveforms
from peass.config import DecompositionConfiguration
from peass.config import DecompositionResult
from .gammatone import DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
from .gammatone import GammatoneAnalyzer
from .gammatone import GammatoneSynthesizer
from .gammatone import calculate_equivalent_rectangular_bandwidth
from .gammatone import fast_resample_poly
from .gammatone import resample_output_length


@lru_cache(maxsize=8)
def _get_posv(dtype):
    """Cached LAPACK ?posv handle for the given complex/real dtype."""
    probe = np.empty((1, 1), dtype=dtype)
    return linalg.get_lapack_funcs(('posv',), (probe, probe))[0]


def matlab_shade_length(shade_milliseconds: float, sampling_frequency_hz: float) -> int:
    """
    Length in samples of a PEASS shade window, matching MATLAB
    `extractDistortionComponents.m` (v2.0.1, lines ~155-165).

    MATLAB builds a periodic Hann of length ``N = 2*round(ms/1000*fs + 1)`` and keeps
    ``w(2:end/2)``, i.e. ``N/2 - 1 = round(ms/1000*fs)`` samples. MATLAB's ``round``
    breaks ties away from zero, whereas Python/NumPy round half to even, so the tie
    rule is spelled out here (it matters at e.g. fs=44100 with shadeMs=5 or 25, where
    ms/1000*fs lands exactly on .5).
    """
    return int(np.floor(shade_milliseconds / 1000.0 * sampling_frequency_hz + 0.5))


def matlab_shade_window(fade_samples: int) -> np.ndarray:
    """
    MATLAB-exact PEASS shade-in window of ``fade_samples`` (= R) samples.

    Reference: `extractDistortionComponents.m` (v2.0.1)::

        wShadeIn = hann(2*round(options.shadeInMs/1000*fs+1),'periodic');
        wShadeIn = wShadeIn(2:end/2);

    With ``hann(N,'periodic')[k] = 0.5*(1 - cos(2*pi*(k-1)/N))`` for 1-based k=1..N and
    ``N = 2*(R+1)``, the slice keeps k = 2..N/2, which collapses to

        w[n] = 0.5 * (1 - cos(pi*(n+1)/(R+1)))      for n = 0..R-1

    This is the strict INTERIOR of a Hann rise: MATLAB drops both the leading zero
    (k=1) and the unity midpoint (k=N/2+1), so the window never reaches exactly 0 or
    exactly 1. Do NOT "simplify" it back to a full 0->1 ramp (``pi*n/(R-1)``) - that
    deviates from the reference by ~1.3e-3 at 44.1 kHz and ~7e-3 at 8 kHz, and it is
    undefined for R=1 whereas this form is well defined for every R >= 1.

    The shade-out window is this window reversed (MATLAB applies ``flipud``).
    """
    time_steps = np.arange(1, fade_samples + 1, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(np.pi * time_steps / (fade_samples + 1)))


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

    This is the single-frame reference statement of the projection. The decomposition
    itself no longer calls it per frame -- ``perform_time_varying_least_squares_projection``
    below runs the identical arithmetic a batch of frames at a time -- but it is kept as
    the readable definition, and the torch backend is diffed against it frame by frame in
    ``tests/regression/test_differential_numpy_vs_torch.py``. Any change here has to be
    mirrored there, which
    ``test_batched_projection_matches_single_frame_bitwise`` enforces.

    Solver note: this calls LAPACK ``?posv`` directly rather than
    ``scipy.linalg.solve(..., assume_a='pos')``. It is the same Cholesky solve and is
    bitwise identical, but SciPy's per-call input validation and condition-number
    estimate cost several times more than the ~42x42 solve itself (11x measured over
    2000 solves).

    This does narrow the pseudo-inverse fallback. Previously a ``LinAlgWarning`` was
    promoted to an error, so a merely ill-conditioned (but positive-definite) frame
    also fell back to ``pinv``; now only a genuine non-positive-definite factorization
    (``info != 0``) does. That matches MATLAB's ``[R, flag] = chol(...)`` test, which is
    what this port mirrors. On the reference example the fallback fires 0 times in
    2337 frames either way.
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
    # Build the conjugate transpose once; it was previously materialised twice.
    conjugate_transpose = toeplitz_matrix.conj().T
    gram_matrix = conjugate_transpose @ (window_sq * toeplitz_matrix)
    rhs_vector = conjugate_transpose @ (window_sq * source_estimates)

    # In-place diagonal regularization
    regularization_lambda = 10.0 ** -15
    gram_matrix.flat[::gram_matrix.shape[0] + 1] += regularization_lambda

    # Call LAPACK ?posv straight through: identical Cholesky solve, but without
    # SciPy's per-call validation, condition-number estimate and warnings machinery,
    # which cost several times more than the 42x42 solve itself. A nonzero `info`
    # means "not positive definite", mirroring MATLAB's chol() flag test.
    posv = _get_posv(gram_matrix.dtype)
    _, projection_weights, info = posv(gram_matrix, rhs_vector, lower=False,
                                       overwrite_a=False, overwrite_b=False)
    if info != 0:
        # Fallback to pseudo-inverse if singular or highly ill-conditioned
        weighted_toeplitz = toeplitz_matrix * analysis_window[:, np.newaxis]
        weighted_estimates = source_estimates * analysis_window[:, np.newaxis]
        projection_weights = linalg.pinv(weighted_toeplitz) @ weighted_estimates

    # Assemble all per-source projections with ONE matmul against a block-diagonal
    # weight matrix. num_sources separate (num_samples x filter_length) matmuls are
    # dominated by per-call overhead at these sizes, so trading a few extra FLOPs for
    # a single BLAS call is a net win (~1.03x end-to-end).
    #
    # The extra terms are exact zeros, so this is an exact equivalence, but a
    # num_sources*filter_length GEMM accumulates in a different blocked order than
    # num_sources small ones -- worth ~1 ULP (1.9e-15) against the previous output.
    # This is the residual difference left over when USE_NUMBA_RESAMPLER is disabled;
    # revert this hunk as well for bitwise-identical results. See the README section
    # "NumPy backend performance and numerical reproducibility".
    num_channels = source_estimates.shape[1]
    block_diagonal_weights = np.zeros(
        (num_sources * filter_length, num_sources * num_channels), dtype=projection_weights.dtype
    )
    for source_idx in range(num_sources):
        block_diagonal_weights[
            source_idx * filter_length: (source_idx + 1) * filter_length,
            source_idx * num_channels: (source_idx + 1) * num_channels
        ] = projection_weights[source_idx * filter_length: (source_idx + 1) * filter_length, :]

    stacked = toeplitz_matrix @ block_diagonal_weights
    stacked *= analysis_window[:, np.newaxis]
    return stacked.reshape(num_samples, num_sources, num_channels).transpose(0, 2, 1)


# Frames solved per batched Gram/RHS build. The stacked Toeplitz buffer is
# BATCH x window_length x (num_sources*filter_length) complex; what matters is that
# this makes its footprint independent of clip length, since without it a 60 s clip
# would stack ~3000 frames per band at once. Measured on the two-source reference
# example the stack peaks at 12.6 MB, or ~38 MB counting the transient conjugate and
# windowed copies that feed the two matmuls, and it scales with num_sources.
LEAST_SQUARES_FRAME_BATCH = 256


def perform_time_varying_least_squares_projection(
        source_estimates: np.ndarray,
        true_sources: np.ndarray,
        filter_length: int,
        window_length: int,
        hop_size: int
) -> np.ndarray:
    r"""
    Time-varying least-squares subband decomposer.

    Frames are built and solved in batches rather than one at a time. Every frame in a
    band shares ``window_length`` and ``filter_length``, so the band's whole Toeplitz
    stack is a single strided gather and the Gram/RHS products collapse into two
    batched ``matmul`` calls; only the small Cholesky solves stay in a Python loop.
    Worth ~1.15x on the decomposition, and the win is *not* the matmul -- measured per
    frame, roughly two thirds of the cost was per-frame Python and NumPy call overhead
    (the ``as_strided``/``hstack`` Toeplitz build alone was ~24%), which batching
    removes.

    This is **bitwise** identical to solving frame by frame: stacked ``matmul``
    dispatches the same per-frame GEMM shapes and memory layouts, so nothing is
    reassociated. ``perform_least_squares_projection`` above remains the single-frame
    statement of the same arithmetic -- it is the reference implementation and the
    oracle the torch backend is diffed against -- and
    ``test_batched_projection_matches_single_frame_bitwise`` pins the two together.
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

    projections_accumulation = np.zeros(
        (total_samples, num_channels, num_sources),
        dtype=true_sources.dtype
    )
    window_gain_accumulation = np.zeros((total_samples, 1))

    # Pad the source array once up front. Previously every frame rebuilt its own
    # support with np.vstack plus two zero allocations, even though the padding is
    # empty for every interior frame; now each frame is a plain view.
    edge_padding = np.zeros((filter_half_length, num_sources), dtype=true_sources.dtype)
    padded_sources = np.vstack([edge_padding, true_sources, edge_padding])
    frame_source_length = window_length + 2 * filter_half_length

    window_begin = 0
    window_end = window_begin + window_length
    frame_starts = []
    while window_end - window_length / 2.0 <= projections_accumulation.shape[0] - window_length + 1:
        frame_starts.append(window_begin)
        window_begin += hop_size
        window_end += hop_size

    if not frame_starts:
        return projections_accumulation[:-(window_length - 1), :, :]

    window_sq = (analysis_window ** 2)[:, np.newaxis]
    # The per-frame loop recomputed this product on every iteration.
    frame_window_gain = synthesis_window * analysis_window
    stacked_width = num_sources * filter_length

    # Per-source contiguous columns, so the stacked Toeplitz is a pure stride trick:
    # view[frame, sample, tap] = column[frame*hop_size + sample + tap]. Gathering from
    # a column of `padded_sources` directly would work too, just with a wider stride;
    # the copied values -- and therefore every downstream rounding -- are the same.
    source_columns = [np.ascontiguousarray(padded_sources[:, idx]) for idx in range(num_sources)]
    element_stride = source_columns[0].strides[0]
    # Flat C-order view of the same buffer, used to evaluate the silence bypass for a
    # whole batch in one reduction.
    flat_sources = padded_sources.reshape(-1)

    for batch_start in range(0, len(frame_starts), LEAST_SQUARES_FRAME_BATCH):
        batch_starts = frame_starts[batch_start:batch_start + LEAST_SQUARES_FRAME_BATCH]
        num_frames = len(batch_starts)
        first_start = batch_starts[0]

        # --- SILENCE BYPASS OPTIMIZATION (batched) ---
        # Each row below is one frame's (frame_source_length x num_sources) source
        # block laid out in C order, i.e. exactly the buffer the per-frame code reduced
        # with `np.sum(true_sources.real**2 + true_sources.imag**2)`. Same elements in
        # the same order, so the pairwise summation -- and hence the threshold decision
        # -- is unchanged.
        source_blocks = np.lib.stride_tricks.as_strided(
            flat_sources[first_start * num_sources:],
            shape=(num_frames, frame_source_length * num_sources),
            strides=(hop_size * num_sources * element_stride, element_stride),
            writeable=False
        )
        if np.iscomplexobj(source_blocks):
            frame_energy = np.sum(source_blocks.real ** 2 + source_blocks.imag ** 2, axis=1)
        else:
            frame_energy = np.sum(source_blocks ** 2, axis=1)
        frame_is_live = frame_energy >= 1e-13

        if not frame_is_live.any():
            # Every frame in this batch projects to exact zeros; only the window gain
            # still has to be accumulated.
            for window_begin in batch_starts:
                window_gain_accumulation[window_begin:window_begin + window_length, 0] += frame_window_gain
            continue

        toeplitz_stack = np.empty(
            (num_frames, window_length, stacked_width), dtype=padded_sources.dtype
        )
        for source_idx in range(num_sources):
            strided_view = np.lib.stride_tricks.as_strided(
                source_columns[source_idx][first_start:],
                shape=(num_frames, window_length, filter_length),
                strides=(hop_size * element_stride, element_stride, element_stride),
                writeable=False
            )
            # Reverse each row to match toeplitz(col, row) semantics perfectly
            toeplitz_stack[:, :, source_idx * filter_length:(source_idx + 1) * filter_length] = (
                strided_view[:, :, ::-1]
            )

        estimate_frames = np.lib.stride_tricks.as_strided(
            source_estimates[first_start:],
            shape=(num_frames, window_length, num_channels),
            strides=(hop_size * source_estimates.strides[0],
                     source_estimates.strides[0], source_estimates.strides[1]),
            writeable=False
        )

        # Build the conjugate transpose once; it feeds both the Gram and the RHS.
        conjugate_transpose = toeplitz_stack.conj().transpose(0, 2, 1)
        gram_matrices = conjugate_transpose @ (window_sq * toeplitz_stack)
        rhs_vectors = conjugate_transpose @ (window_sq * estimate_frames)

        # In-place diagonal regularization, one diagonal per stacked matrix
        regularization_lambda = 10.0 ** -15
        gram_matrices.reshape(num_frames, -1)[:, ::stacked_width + 1] += regularization_lambda

        posv = _get_posv(gram_matrices.dtype)
        block_diagonal_weights = np.zeros(
            (num_frames, stacked_width, num_sources * num_channels), dtype=gram_matrices.dtype
        )
        for frame_idx in range(num_frames):
            if not frame_is_live[frame_idx]:
                continue
            _, projection_weights, info = posv(
                gram_matrices[frame_idx], rhs_vectors[frame_idx],
                lower=False, overwrite_a=False, overwrite_b=False
            )
            if info != 0:
                # Fallback to pseudo-inverse if singular or highly ill-conditioned
                weighted_toeplitz = toeplitz_stack[frame_idx] * analysis_window[:, np.newaxis]
                weighted_estimates = estimate_frames[frame_idx] * analysis_window[:, np.newaxis]
                projection_weights = linalg.pinv(weighted_toeplitz) @ weighted_estimates
            for source_idx in range(num_sources):
                block_diagonal_weights[
                    frame_idx,
                    source_idx * filter_length: (source_idx + 1) * filter_length,
                    source_idx * num_channels: (source_idx + 1) * num_channels
                ] = projection_weights[source_idx * filter_length: (source_idx + 1) * filter_length, :]

        frame_projections = toeplitz_stack @ block_diagonal_weights
        # A bypassed frame contributes a literal `np.zeros(...)` in the per-frame code.
        # Leaving it to `toeplitz @ 0` would be numerically equal but can yield -0.0
        # (0.0 * negative), so write the zeros explicitly and keep even the sign of
        # zero identical.
        frame_projections[~frame_is_live] = 0.0
        frame_projections *= analysis_window[:, np.newaxis]
        # Second, separate multiply by the synthesis window. The per-frame code applies
        # the analysis window inside the projection and the synthesis window during the
        # overlap-add; collapsing them into one `analysis*synthesis` factor would round
        # differently, so the two multiplies stay distinct.
        frame_projections *= synthesis_window[:, np.newaxis]
        frame_projections = frame_projections.reshape(
            num_frames, window_length, num_sources, num_channels
        )

        for frame_idx, window_begin in enumerate(batch_starts):
            projections_accumulation[window_begin:window_begin + window_length, :, :] += (
                frame_projections[frame_idx].transpose(0, 2, 1)
            )
            window_gain_accumulation[window_begin:window_begin + window_length, 0] += frame_window_gain

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
        # Cached, and shared with synthesis: `get_synthesis_modulation_matrix` reuses
        # this matrix by conjugation rather than running np.exp a second time.
        modulation_matrix = get_analysis_modulation_matrix(
            sampling_frequency_hz, subbands_output.shape[1], analyzer.center_frequencies
        )

    # In-place: `process` hands back a freshly allocated complex buffer that nothing
    # else holds a reference to, so this saves an allocation and a full copy of a
    # (bands x samples) complex128 array without changing a single rounding.
    subbands_output *= modulation_matrix

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
# MODULATION MATRIX CACHE
# -----------------------------------------------------------------------------
#
# Analysis demodulates each band by exp(-i*w*t) and synthesis remodulates it by
# exp(+i*w*t) at the same centre frequencies and rate, so the two matrices are exact
# conjugates -- and *bitwise* so, because `-2j*pi/fs*f*t` is the exact IEEE negation of
# `2j*pi/fs*f*t` and cos/sin are exactly even/odd about zero. Synthesis needs a few
# hundred columns more than analysis (each band is padded up to a whole number of
# decimated samples), so it conjugates the analysis matrix and only calls np.exp on the
# overhang. On a 5 s clip that turns a ~0.35 s np.exp over a (bands x samples) grid into
# a ~0.03 s conjugation; the matrix build was 11.5% of the decomposition.
#
# The cache is a plain dict rather than lru_cache because the synthesis lookup has to
# find an analysis entry of a *different* length, which lru_cache cannot express. It is
# bounded for the same reason the old lru_cache was: every entry is a full
# (num_bands x num_samples) complex matrix, so a long-lived process scoring many
# different-length signals must not accumulate them.
_MODULATION_MATRIX_CACHE: dict[tuple, np.ndarray] = {}
_MODULATION_MATRIX_CACHE_SIZE = 4


def _store_modulation_matrix(cache_key: tuple, matrix: np.ndarray) -> np.ndarray:
    """Insert into the bounded modulation cache, evicting in insertion order."""
    while len(_MODULATION_MATRIX_CACHE) >= _MODULATION_MATRIX_CACHE_SIZE:
        _MODULATION_MATRIX_CACHE.pop(next(iter(_MODULATION_MATRIX_CACHE)))
    _MODULATION_MATRIX_CACHE[cache_key] = matrix
    return matrix


def get_analysis_modulation_matrix(
        sampling_frequency: float,
        max_samples_length: int,
        center_frequencies: np.ndarray
) -> np.ndarray:
    """Cached ``exp(-2j*pi*f/fs * t)`` for ``t = 0 .. max_samples_length - 1``."""
    center_frequencies = np.asarray(center_frequencies)
    cache_key = ('analysis', sampling_frequency, tuple(center_frequencies))
    cached = _MODULATION_MATRIX_CACHE.get(cache_key)
    # Only an exact length hit is reused: slicing a longer matrix would hand back a
    # non-contiguous view, and the caller multiplies a whole subband block by it.
    if cached is not None and cached.shape[1] == max_samples_length:
        return cached

    time_steps = np.arange(max_samples_length)
    matrix = np.exp(-2j * np.pi / sampling_frequency * center_frequencies[:, np.newaxis] * time_steps)
    return _store_modulation_matrix(cache_key, matrix)


def get_synthesis_modulation_matrix(
        sampling_frequency: float,
        max_samples_length: int,
        center_frequencies: np.ndarray
) -> np.ndarray:
    """
    Cached ``exp(+2j*pi*f/fs * t)`` for ``t = 0 .. max_samples_length - 1``, derived
    from the analysis matrix by conjugation wherever the two overlap (see above).
    """
    center_frequencies = np.asarray(center_frequencies)
    frequencies_key = tuple(center_frequencies)
    cache_key = ('synthesis', sampling_frequency, frequencies_key)
    cached = _MODULATION_MATRIX_CACHE.get(cache_key)
    if cached is not None and cached.shape[1] == max_samples_length:
        return cached

    matrix = np.empty((center_frequencies.size, max_samples_length), dtype=complex)
    analysis_matrix = _MODULATION_MATRIX_CACHE.get(('analysis', sampling_frequency, frequencies_key))
    shared_length = 0 if analysis_matrix is None else min(max_samples_length, analysis_matrix.shape[1])
    if shared_length > 0:
        np.conjugate(analysis_matrix[:, :shared_length], out=matrix[:, :shared_length])
        # t = 0 is the one column conjugation gets "wrong". Both directions evaluate
        # exp(0+0j) = 1+0j there -- the sign of the exponent has been multiplied away --
        # so conjugating yields 1-0j: equal in value, different bit pattern from the
        # +0.0 a direct np.exp leaves behind. Every t >= 1 column really is an exact
        # conjugate, because the two arguments are exact IEEE negations and cos/sin are
        # exactly even/odd.
        #
        # To be clear about why this recompute exists: it is *not* needed for
        # correctness. Measured, feeding the -0.0 column straight through leaves all
        # four output components byte-identical and every score unchanged -- the
        # +-0.0 cross terms in the complex multiply are absorbed by addition, and
        # nothing downstream divides by these values or takes a branch cut (angle/log/
        # complex sqrt) where a zero's sign would survive. It exists so that
        # "bit-identical" stays literally true and checkable by a one-line byte
        # comparison, instead of degrading to "value-identical modulo signed zeros",
        # which no test can assert cheaply and which would have to be re-argued by hand
        # every time a new consumer of this matrix appears. It costs one np.exp over a
        # single column per cache miss.
        matrix[:, :1] = np.exp(
            2j * np.pi / sampling_frequency * center_frequencies[:, np.newaxis] * np.arange(1)
        )
    if shared_length < max_samples_length:
        # `arange(a, b)` and `arange(b)[a:]` hold the same exactly-representable
        # integers, so the tail rounds identically to a full-length np.exp.
        time_steps = np.arange(shared_length, max_samples_length)
        matrix[:, shared_length:] = np.exp(
            2j * np.pi / sampling_frequency * center_frequencies[:, np.newaxis] * time_steps
        )
    return _store_modulation_matrix(cache_key, matrix)


def _can_scatter_upsampling_in_place(
        block: np.ndarray,
        band_indices: np.ndarray,
        subband_list: list,
        factor: int,
        processed_subbands: np.ndarray
) -> bool:
    """Whether a synthesis upsampling block may be written straight into its rows.

    Four things have to hold, and all four are about not corrupting the parts of
    `processed_subbands` this block does not own:

    * the block is a plain rectangular 2D array (ragged input would have produced
      an object array, which the resampler cannot handle in place);
    * its destination rows are one contiguous slab, so `processed_subbands[lo:hi]`
      is a genuine C-contiguous view -- fancy indexing would hand back a copy and
      the write would be silently thrown away;
    * `out=`'s dtype contract is met (complex input needs a complex128 buffer);
    * and, the load-bearing one, the upsampled length is <= EVERY band's
      target_length, so writing columns [0, out_len) can never reach past
      target_length into the zeros the copy route deliberately leaves behind.

    In the real filterbank all four always hold: decimation factors fall
    monotonically with band index so each factor's bands are consecutive, and every
    band sharing a factor has the same subband length, which makes out_len exactly
    equal to target_length. The guard exists so that an unusual caller degrades to
    the copy route instead of to wrong output.
    """
    if block.ndim != 2 or block.dtype == object:
        return False
    if np.iscomplexobj(block) != np.iscomplexobj(processed_subbands):
        return False
    if int(band_indices[-1]) - int(band_indices[0]) + 1 != band_indices.size:
        return False
    upsampled_length = resample_output_length(block.shape[-1], factor, 1)
    return all(upsampled_length <= len(subband_list[b]) * factor for b in band_indices)


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
        factor = int(factor)
        band_indices = np.where(analyzer.decimation_factors == factor)[0]
        # Faster stacking than vstack
        block = np.array([subband_list[b] for b in band_indices])

        if _can_scatter_upsampling_in_place(block, band_indices, subband_list, factor, processed_subbands):
            # Resample straight into the rows we own instead of allocating an
            # upsampled block and copying it in. `out=` writes only the first
            # `resample_output_length` columns of each row, and the guard above has
            # already established that this is <= every band's target_length, so the
            # zeros this loop is supposed to leave behind -- both the short-write gap
            # below target_length and the whole tail past it -- are never written
            # over. The truncating case (`curr_len > target_length`) is deliberately
            # NOT served here: writing in place would have to lay down samples past
            # target_length before discarding them, so it falls through to the copy
            # route, which is the only branch that can throw samples away.
            fast_resample_poly(
                block, factor, 1, axis=-1, half_length_factor=half_length_factor,
                out=processed_subbands[band_indices[0]:band_indices[-1] + 1]
            )
            continue

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

    # Retrieve cached modulation matrix to bypass np.exp re-calculations. Multiplied
    # in place -- `processed_subbands` is the local zero-filled buffer from above, and
    # the cached matrix is only ever read, so the cache entry is not disturbed.
    processed_subbands *= get_synthesis_modulation_matrix(
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

        fade_in_samples = matlab_shade_length(shade_in, fs) if shade_in > 0 else 0
        fade_out_samples = matlab_shade_length(shade_out, fs) if shade_out > 0 else 0

        # Explicitly validate signal length against configured shading windows
        if fade_in_samples + fade_out_samples > num_samples:
            raise ValueError(
                f"Combined shading length ({fade_in_samples + fade_out_samples} samples) "
                f"exceeds the signal length ({num_samples} samples)."
            )

        if fade_in_samples > 0:
            shade_in_window = matlab_shade_window(fade_in_samples)
            # Vectorized channel multiplication
            shaded_signal[:fade_in_samples, :] *= shade_in_window[:, np.newaxis]

        if fade_out_samples > 0:
            # MATLAB obtains the fade-out via flipud() of the same window; deriving it
            # by reversal (rather than a separate cosine) keeps the two exactly consistent.
            shade_out_window = matlab_shade_window(fade_out_samples)[::-1]
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
