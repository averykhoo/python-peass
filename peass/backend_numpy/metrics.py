"""
PEASS Metrics Package - Auditory Features & Similarity Metrics

Computes perceptual features and linear/energy ratio calculations
using extreme N-dimensional vectorized broadcasting.
"""

import numpy as np

from .auditory_model import generate_auditory_internal_representation


def calculate_bss_eval_energy_ratios(
        true_source: np.ndarray,
        target_distortion: np.ndarray,
        interference: np.ndarray,
        artifacts: np.ndarray
) -> tuple[float, float, float, float]:
    r"""
    Computes standard BSS Eval energy ratio metrics from physically decomposed components.

    :param true_source: Extracted target physical waveform.
    :type true_source: numpy.ndarray
    :param target_distortion: Distorted spatial features waveform.
    :type target_distortion: numpy.ndarray
    :param interference: Background overlapping source leakages.
    :type interference: numpy.ndarray
    :param artifacts: Non-linear processing artifacts.
    :type artifacts: numpy.ndarray
    :return: A tuple of (ISR, SIR, SAR, SDR) in Decibels (dB).
    :rtype: tuple[float, float, float, float]
    """
    flat_true_source = true_source.ravel()
    flat_target_distortion = target_distortion.ravel()
    flat_interference = interference.ravel()
    flat_artifacts = artifacts.ravel()

    # Epsilon bounds to prevent division by zero in log scale
    eps = np.finfo(np.float64).eps

    # Pre-calculate vector sums to avoid redundant additions
    sum_target_dist = flat_true_source + flat_target_distortion
    sum_interf = sum_target_dist + flat_interference

    # Compute sum-of-squares via highly optimized self-dot products
    energy_true = np.dot(flat_true_source, flat_true_source)
    energy_dist = np.dot(flat_target_distortion, flat_target_distortion)
    energy_interf = np.dot(flat_interference, flat_interference)
    energy_artif = np.dot(flat_artifacts, flat_artifacts)

    energy_true_dist = np.dot(sum_target_dist, sum_target_dist)
    energy_sum_all = np.dot(sum_interf, sum_interf)

    # Compute overall distortion (eTarget + eInterf + eArtif)
    sum_all_distortions = flat_target_distortion + flat_interference + flat_artifacts
    energy_total_distortion = np.dot(sum_all_distortions, sum_all_distortions)

    source_to_spatial_distortion_ratio = 10.0 * np.log10(
        np.maximum(energy_true, eps) / np.maximum(energy_dist, eps)
    )
    source_to_interference_ratio = 10.0 * np.log10(
        np.maximum(energy_true_dist, eps) / np.maximum(energy_interf, eps)
    )
    source_to_artifacts_ratio = 10.0 * np.log10(
        np.maximum(energy_sum_all, eps) / np.maximum(energy_artif, eps)
    )
    source_to_distortion_ratio = 10.0 * np.log10(
        np.maximum(energy_true, eps) / np.maximum(energy_total_distortion, eps)
    )

    return (
        source_to_spatial_distortion_ratio,
        source_to_interference_ratio,
        source_to_artifacts_ratio,
        source_to_distortion_ratio
    )


def calculate_auditory_similarity_metric(
        internal_reference_representation: np.ndarray,
        internal_test_representation: np.ndarray,
        representation_sampling_frequency: float
) -> float:
    r"""
    Compares two internal representations to produce an auditory similarity metric.
    Computes cross-correlation heavily vectorized over the temporal framing dimensions.
    """
    num_bands, num_samples, num_modulations = internal_reference_representation.shape

    internal_reference_representation = np.real(internal_reference_representation)
    internal_test_representation = np.real(internal_test_representation).copy()
    assimilation_mask = (np.abs(internal_test_representation) < np.abs(internal_reference_representation))
    internal_test_representation[assimilation_mask] = (
            0.25 * internal_reference_representation[assimilation_mask] +
            0.75 * internal_test_representation[assimilation_mask]
    )

    frame_length = int(min(num_samples, 0.1 * representation_sampling_frequency))
    num_frames = int(np.floor(num_samples / frame_length))
    num_samples = num_frames * frame_length

    # Ensure shape aligns cleanly onto boundaries
    ref_view = internal_reference_representation[:, :num_samples, :]
    test_view = internal_test_representation[:, :num_samples, :]

    # Reshape from (nband, nfram * flen, nmod) to (nfram, nband * flen, nmod)
    ref_frames = ref_view.reshape(num_bands, num_frames, frame_length, num_modulations)
    ref_frames = ref_frames.transpose(1, 0, 2, 3).reshape(num_frames, num_bands * frame_length, num_modulations)

    test_frames = test_view.reshape(num_bands, num_frames, frame_length, num_modulations)
    test_frames = test_frames.transpose(1, 0, 2, 3).reshape(num_frames, num_bands * frame_length, num_modulations)

    # Vectorized similarity measure components
    eps = np.finfo(np.float64).eps
    ref_mean = np.mean(ref_frames, axis=1, keepdims=True)
    lref = ref_frames - ref_mean

    ltest_orig = test_frames
    lnms = np.sum(ltest_orig ** 2, axis=1)

    ltest_mean = np.mean(ltest_orig, axis=1, keepdims=True)
    ltest = ltest_orig - ltest_mean

    denominator = np.sqrt(np.sum(lref ** 2, axis=1) * np.sum(ltest ** 2, axis=1))

    lpsm = np.sum(lref * ltest, axis=1) / np.maximum(denominator, eps)
    local_perceptual_similarity_measures = np.sum(lpsm * lnms, axis=1) / np.maximum(np.sum(lnms, axis=1), eps)

    # From local to global similarity
    integration_length = int(1.0 * representation_sampling_frequency)
    squared_test_representation = np.sum(test_view ** 2, axis=(0, 2))

    # O(1) cumulative prefix sums - Fully vectorized to eliminate Python loop overhead
    cum_sum = np.concatenate(([0.0], np.cumsum(squared_test_representation)))

    frame_indices = np.arange(num_frames)
    frame_centers = (frame_indices + 0.5) * frame_length

    start_indices = np.maximum(0, frame_centers - 0.5 * integration_length).astype(int)
    end_indices = np.minimum(num_samples, frame_centers + 0.5 * integration_length).astype(int)
    slice_lengths = end_indices - start_indices

    # As proved, slice_lengths is guaranteed to be strictly >= 1 for any valid inputs,
    # so we can perform the division safely without conditional branching.
    root_mean_square_energy = (cum_sum[end_indices] - cum_sum[start_indices]) / slice_lengths

    sorted_indices = np.argsort(local_perceptual_similarity_measures)
    sorted_similarity_measures = local_perceptual_similarity_measures[sorted_indices]
    sorted_rms_energy = root_mean_square_energy[sorted_indices]
    cumulative_rms_energy = np.cumsum(sorted_rms_energy)

    cutoff_energy = 0.5 * cumulative_rms_energy[-1]
    matching_indices = np.where(cumulative_rms_energy >= cutoff_energy)[0]

    return float(sorted_similarity_measures[matching_indices[0]]) if len(matching_indices) > 0 else 0.0


def calculate_auditory_quality_features(
        decomposition_signals: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        sampling_frequency_hz: float = 16000.0
) -> tuple[float, float, float, float]:
    """Computes quality features by sending decomposed signals through the internal auditory model."""
    true_target, target_distortion, interference, artifacts = decomposition_signals

    if len(true_target.shape) == 1:
        true_target = true_target[:, np.newaxis]
        target_distortion = target_distortion[:, np.newaxis]
        interference = interference[:, np.newaxis]
        artifacts = artifacts[:, np.newaxis]

    composite_estimate = true_target + target_distortion + interference + artifacts
    num_channels = true_target.shape[1]

    channel_target_quality = np.zeros(num_channels)
    channel_interference_quality = np.zeros(num_channels)
    channel_artifacts_quality = np.zeros(num_channels)
    channel_global_quality = np.zeros(num_channels)

    # --- FUTURE WORK: Potential Parallelization Bottleneck ---
    # These sequential generate_auditory_internal_representation calls are independent.
    # To optimize multi-core throughput, offload these tasks to a concurrent.futures.ThreadPoolExecutor.
    for channel_idx in range(num_channels):
        mtest, fr = generate_auditory_internal_representation(composite_estimate[:, channel_idx], sampling_frequency_hz)

        mref_t, _ = generate_auditory_internal_representation(
            true_target[:, channel_idx] + interference[:, channel_idx] + artifacts[:, channel_idx],
            sampling_frequency_hz
        )
        channel_target_quality[channel_idx] = calculate_auditory_similarity_metric(mref_t, mtest, fr)

        mref_i, _ = generate_auditory_internal_representation(
            true_target[:, channel_idx] + target_distortion[:, channel_idx] + artifacts[:, channel_idx],
            sampling_frequency_hz
        )
        channel_interference_quality[channel_idx] = calculate_auditory_similarity_metric(mref_i, mtest, fr)

        mref_a, _ = generate_auditory_internal_representation(
            true_target[:, channel_idx] + target_distortion[:, channel_idx] + interference[:, channel_idx],
            sampling_frequency_hz
        )
        channel_artifacts_quality[channel_idx] = calculate_auditory_similarity_metric(mref_a, mtest, fr)

        mref_g, _ = generate_auditory_internal_representation(true_target[:, channel_idx], sampling_frequency_hz)
        channel_global_quality[channel_idx] = calculate_auditory_similarity_metric(mref_g, mtest, fr)

    return (
        float(np.min(channel_target_quality)),
        float(np.min(channel_interference_quality)),
        float(np.min(channel_artifacts_quality)),
        float(np.min(channel_global_quality))
    )
