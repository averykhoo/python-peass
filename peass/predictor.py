"""
PEASS Predictor Package - Multi-Criteria Neural Network Regressor

Maps raw auditory similarity scores to Predicted Perceptual Scores
(OPS, TPS, IPS, APS) on a scale from 0 to 100.
"""

import os
import pathlib
from functools import lru_cache

import numpy as np
import soundfile as sf

from .config import DecompositionConfiguration
from .config import PerceptualSeparationScores
from .decomposition import decompose_distortion_components
from .metrics import calculate_auditory_quality_features
from .metrics import calculate_bss_eval_energy_ratios

MODEL_WEIGHTS_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "parameters")


@lru_cache
def _get_model_parameters(task_idx: int) -> dict:
    parameters_path = os.path.join(MODEL_WEIGHTS_DIRECTORY, f"paramTask{task_idx + 1}.npz")
    with np.load(parameters_path) as parameters_data:
        return {
            'w':     parameters_data['W'],  # hidden_layer_weights
            'b':     parameters_data['b'],  # hidden_layer_bias
            'v':     parameters_data['v'],  # output_layer_weights
            'a':     parameters_data['a'],  # output_layer_bias
            'selec': parameters_data['selec']
        }


def evaluate_neural_network_mapping(
        features: np.ndarray,
        w: np.ndarray,  # hidden_layer_weights
        b: np.ndarray,  # hidden_layer_bias
        v: np.ndarray,  # output_layer_weights
        a: np.ndarray,  # output_layer_bias
) -> float:
    """
    Evaluates forward propagation through the two-layer perceptron.
    """
    if len(features.shape) == 1:
        features = features[:, np.newaxis]

    hidden_layer_activation = w @ features + b
    hidden_layer_output = 1.0 / (1.0 + np.exp(-hidden_layer_activation))

    output_layer_activation = v.T @ hidden_layer_output + a
    final_score = 100.0 / (1.0 + np.exp(-output_layer_activation))

    return float(final_score.item())


def predict_perceptual_evaluation_scores(
        original_files: list[str | np.ndarray],
        estimate_file: str | np.ndarray,
        configuration: DecompositionConfiguration | None = None,
        sampling_frequency_hz: float | None = None,
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

    # Protect against floating-point boundary violations before log-mapping
    eps = np.finfo(float).eps
    clamped_quality_features = np.clip(auditory_quality_features, -1.0 + eps, 1.0 - eps)
    log_mapped_quality_features = np.clip(
        np.log((1.0 + clamped_quality_features) / (1.0 - clamped_quality_features)),
        -5.5,
        5.5
    )

    scores = np.zeros(4)

    for task_idx in range(4):
        params = _get_model_parameters(task_idx)
        selected_features = log_mapped_quality_features[params['selec']]

        # Filter out 'selec' and unpack the rest (w, b, v, a) as keyword arguments
        model_weights = {k: v for k, v in params.items() if k != 'selec'}
        scores[task_idx] = evaluate_neural_network_mapping(selected_features, **model_weights)

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
