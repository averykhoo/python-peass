"""
PEASS Predictor Package - Multi-Criteria Neural Network Regressor [1]

This module maps raw auditory similarity scores (qTarget, qInterf, qArtif, qGlobal)
to Predicted Perceptual Scores (OPS, TPS, IPS, APS) on a scale from 0 to 100
using modern .npz parameter loading [1].
"""

import os
import pathlib
from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import soundfile as sf

from .decomposition import extract_distortion_components
from .metrics import audio_quality_features
from .metrics import calculate_energy_ratios


def my_mapping(features: np.ndarray, weights: np.ndarray, bias: np.ndarray, output_weights: np.ndarray,
               output_bias: np.ndarray) -> float:
    """
    Evaluates forward propagation through the two-layer perceptron.
    Replaces myMapping.m [1].
    """
    if len(features.shape) == 1:
        features = features[:, np.newaxis]

    # Hidden layer
    s1 = weights @ features + bias
    o1 = 1.0 / (1.0 + np.exp(-s1))

    # Output layer
    s2 = output_weights.T @ o1 + output_bias
    y = 100.0 / (1.0 + np.exp(-s2))

    return float(y[0, 0])


def predict_peass_scores(
        original_files: list[Union[str, np.ndarray]],
        estimate_file: Union[str, np.ndarray],
        options: dict = None,
        sampling_frequency: float = None,
        return_decomposition: bool = False
) -> Dict[str, Any]:
    """
    Wrapper entry point. Performs least-squares decomposition, generates auditory features,
    and predicts Perceptual Evaluation scores [1].

    Replaces PEASS_ObjectiveMeasure.m and map2SubjScale.m [1].

    Args:
        original_files: List of file paths or NumPy arrays of reference sources.
        estimate_file: File path or NumPy array of the separated estimate.
        options: Algorithmic tuning parameters dictionary.
        sampling_frequency: Rate in Hz (required for in-memory array arrays).
        return_decomposition: If True, returns the calculated waveform arrays/saved filepaths.

    Returns:
        dict: Containing OPS, TPS, IPS, APS and decibel criteria.
              If return_decomposition=True, includes "decomposition_arrays" (and
              "decomposition_files" if inputting file paths).
    """
    # 1. Physical Decomposition
    file_paths, decomposed_arrays = extract_distortion_components(original_files, estimate_file, options,
                                                                  sampling_frequency)
    s_true, e_target, e_interf, e_artif = decomposed_arrays

    if sampling_frequency is None:
        if isinstance(estimate_file, (str, pathlib.Path)):
            _, sampling_frequency = sf.read(estimate_file)
        else:
            sampling_frequency = 16000.0

    # 2. Traditional Energy Ratios
    ISR, SIR, SAR, SDR = calculate_energy_ratios(s_true, e_target, e_interf, e_artif)

    # 3. Auditory Feature Extraction
    q_target, q_interf, q_artif, q_global = audio_quality_features(decomposed_arrays, sampling_frequency)

    # 4. Neural Network Scoring Regressions
    q_features = np.array([q_global, q_target, q_interf, q_artif])
    q_mapped = np.clip(np.log((1.0 + q_features) / (1.0 - q_features)), -5.5, 5.5)

    scores = np.zeros(4)
    # Dynamically locate local parameters folder absolute path
    pkg_dir = os.path.dirname(os.path.realpath(__file__))

    for nTask in range(4):
        npz_path = os.path.join(pkg_dir, "parameters", f"paramTask{nTask + 1}.npz")
        mat_data = np.load(npz_path)

        W = mat_data['W']
        b = mat_data['b']
        v = mat_data['v']
        a = mat_data['a']
        selec = mat_data['selec']

        scores[nTask] = my_mapping(q_mapped[selec], W, b, v, a)

    return {
        "OPS": float(scores[0]),  # Overall Perceptual Score
        "TPS": float(scores[1]),  # Target-related Perceptual Score
        "IPS": float(scores[2]),  # Interference-related Perceptual Score
        "APS": float(scores[3]),  # Artifact-related Perceptual Score
        "SDR": SDR,
        "ISR": ISR,
        "SIR": SIR,
        "SAR": SAR
    }

    if return_decomposition:
        results["decomposition_arrays"] = {
            "true_target":       s_true,
            "target_distortion": e_target,
            "interference":      e_interf,
            "artifacts":         e_artif
        }
        if file_paths:
            results["decomposition_files"] = {
                "true_target":       file_paths[0],
                "target_distortion": file_paths[1],
                "interference":      file_paths[2],
                "artifacts":         file_paths[3]
            }

    return results
