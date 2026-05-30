"""
PEASS Predictor Package - Multi-Criteria Neural Network Regressor

Maps raw auditory similarity scores to Predicted Perceptual Scores
(OPS, TPS, IPS, APS) on a scale from 0 to 100.
"""

import os
import pathlib
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import soundfile as sf

from .config import DecompositionConfiguration
from .config import PerceptualSeparationScores
from .decomposition import decompose_distortion_components
from .metrics import calculate_auditory_quality_features
from .metrics import calculate_bss_eval_energy_ratios


def evaluate_neural_network_mapping(
        features: np.ndarray,
        hidden_layer_weights: np.ndarray,
        hidden_layer_bias: np.ndarray,
        output_layer_weights: np.ndarray,
        output_layer_bias: np.ndarray
) -> float:
    r"""
    Evaluates forward propagation through the two-layer perceptron.
    """
    if len(features.shape) == 1:
        features = features[:, np.newaxis]

    hidden_layer_activation = hidden_layer_weights @ features + hidden_layer_bias
    hidden_layer_output = 1.0 / (1.0 + np.exp(-hidden_layer_activation))

    output_layer_activation = output_layer_weights.T @ hidden_layer_output + output_layer_bias
    final_score = 100.0 / (1.0 + np.exp(-output_layer_activation))

    return float(final_score[0, 0])


def predict_perceptual_evaluation_scores(
        original_files: List[Union[str, np.ndarray]],
        estimate_file: Union[str, np.ndarray],
        configuration: Optional[DecompositionConfiguration] = None,
        sampling_frequency_hz: Optional[float] = None,
        return_decomposition: bool = False
) -> PerceptualSeparationScores:
    r"""
    Performs least-squares decomposition, generates auditory features,
    and predicts Perceptual Evaluation scores on a 0-100 scale.
    """
    if configuration is None:
        configuration = DecompositionConfiguration()

    decomposition_result = decompose_distortion_components(
        source_files=original_files,
        estimate_file=estimate_file,
        configuration=configuration,
        sampling_frequency_hz=sampling_frequency_hz
    )
    waveforms = decomposition_result.waveforms

    if sampling_frequency_hz is None:
        if isinstance(estimate_file, (str, pathlib.Path)):
            _, sampling_frequency_hz = sf.read(estimate_file)
        else:
            sampling_frequency_hz = 16000.0

    (
        source_to_spatial_distortion_ratio,
        source_to_interference_ratio,
        source_to_artifacts_ratio,
        source_to_distortion_ratio
    ) = calculate_bss_eval_energy_ratios(
        waveforms.true_target,
        waveforms.target_distortion,
        waveforms.interference,
        waveforms.artifacts
    )

    decomposition_tuple = (
        waveforms.true_target,
        waveforms.target_distortion,
        waveforms.interference,
        waveforms.artifacts
    )
    q_target, q_interf, q_artif, q_global = calculate_auditory_quality_features(
        decomposition_tuple, sampling_frequency_hz
    )

    auditory_quality_features = np.array([q_global, q_target, q_interf, q_artif])
    log_mapped_quality_features = np.clip(
        np.log((1.0 + auditory_quality_features) / np.maximum(1.0 - auditory_quality_features, np.finfo(float).eps)),
        -5.5,
        5.5
    )

    scores = np.zeros(4)
    package_directory = os.path.dirname(os.path.realpath(__file__))

    for task_idx in range(4):
        parameters_path = os.path.join(package_directory, "parameters", f"paramTask{task_idx + 1}.npz")
        parameters_data = np.load(parameters_path)

        W = parameters_data['W']
        b = parameters_data['b']
        v = parameters_data['v']
        a = parameters_data['a']
        selected_feature_indices = parameters_data['selec']

        scores[task_idx] = evaluate_neural_network_mapping(
            log_mapped_quality_features[selected_feature_indices], W, b, v, a
        )

    perceptual_scores = PerceptualSeparationScores(
        overall_perceptual_score=float(scores[0]),
        target_perceptual_score=float(scores[1]),
        interference_perceptual_score=float(scores[2]),
        artifact_perceptual_score=float(scores[3]),
        source_to_distortion_ratio=source_to_distortion_ratio,
        source_to_spatial_distortion_ratio=source_to_spatial_distortion_ratio,
        source_to_interference_ratio=source_to_interference_ratio,
        source_to_artifacts_ratio=source_to_artifacts_ratio
    )

    if return_decomposition:
        perceptual_scores.decomposition_waveforms = waveforms
        perceptual_scores.decomposition_files = decomposition_result.file_paths

    return perceptual_scores


# -----------------------------------------------------------------------------
# LEGACY BACKWARD-COMPATIBILITY ALIASES
# -----------------------------------------------------------------------------
def predict_peass_scores(
        original_files: List[Union[str, np.ndarray]],
        estimate_file: Union[str, np.ndarray],
        options: Optional[dict] = None,
        sampling_frequency: Optional[float] = None,
        return_decomposition: bool = False
) -> PerceptualSeparationScores:
    """Legacy compatibility wrapper for predict_perceptual_evaluation_scores."""
    config = DecompositionConfiguration()
    if options is not None:
        if 'destDir' in options:
            config.destination_directory = options['destDir']
        if 'FLAG_2PROJ' in options:
            config.use_two_stage_projection = options['FLAG_2PROJ']
        if 'frameLength' in options:
            config.frame_length_seconds = options['frameLength']
        if 'filterLength' in options:
            config.filter_length_seconds = options['filterLength']
        if 'shadeInMs' in options:
            config.shade_in_milliseconds = options['shadeInMs']
        if 'shadeOutMs' in options:
            config.shade_out_milliseconds = options['shadeOutMs']
        if 'segmentationFactor' in options:
            config.segmentation_factor = options['segmentationFactor']

    return predict_perceptual_evaluation_scores(
        original_files=original_files,
        estimate_file=estimate_file,
        configuration=config,
        sampling_frequency_hz=sampling_frequency,
        return_decomposition=return_decomposition
    )
