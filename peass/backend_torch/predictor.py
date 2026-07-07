"""
PEASS PyTorch MLP Predictor
File path: peass/backend_torch/predictor.py
"""
import os
import pathlib

import numpy as np
import soundfile as sf
import torch

from .decomposition import decompose_distortion_components
from .metrics import calculate_auditory_quality_features
from .metrics import calculate_bss_eval_energy_ratios
from ..config import DecompositionConfiguration
from ..config import PerceptualSeparationScores

MODEL_WEIGHTS_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "parameters"
)


def _get_model_parameters_torch(task_idx: int, device, dtype) -> dict:
    parameters_path = os.path.join(MODEL_WEIGHTS_DIRECTORY, f"paramTask{task_idx + 1}.npz")
    with np.load(parameters_path) as data:
        return {
            'w':     torch.tensor(data['W'], device=device, dtype=dtype),
            'b':     torch.tensor(data['b'], device=device, dtype=dtype),
            'v':     torch.tensor(data['v'], device=device, dtype=dtype),
            'a':     torch.tensor(data['a'], device=device, dtype=dtype),
            'selec': torch.tensor(data['selec'], device=device, dtype=torch.long)
        }


def evaluate_neural_network_mapping_torch(features, w, b, v, a):
    if features.dim() == 1:
        features = features.unsqueeze(1)
    hidden_activation = w @ features + b
    hidden_output = torch.sigmoid(hidden_activation)
    output_activation = v.T @ hidden_output + a
    return (100.0 * torch.sigmoid(output_activation)).squeeze()


def predict_perceptual_evaluation_scores(
        original_files,
        estimate_file,
        configuration=None,
        sampling_frequency_hz=None,
        return_decomposition=False
):
    """
    Performs least-squares decomposition, generates auditory features,
    and predicts Perceptual Evaluation scores natively on target hardware.
    """
    if configuration is None:
        configuration = DecompositionConfiguration()

    # Step 1: Run subband decomposition on PyTorch backend
    decomp_result = decompose_distortion_components(
        source_files=original_files,
        estimate_file=estimate_file,
        configuration=configuration,
        sampling_frequency_hz=sampling_frequency_hz
    )
    waveforms = decomp_result.waveforms
    # Derive device/dtype from the decomposed waveforms rather than the input:
    # estimate_file may be a file path (not a tensor), and the internal pipeline
    # always promotes to float64, so this keeps the MLP weights consistent.
    device = waveforms.true_target.device
    dtype = waveforms.true_target.dtype

    # Recover the sampling rate for file-based inputs (mirrors the NumPy backend).
    if sampling_frequency_hz is None:
        if isinstance(estimate_file, (str, pathlib.Path)):
            sampling_frequency_hz = float(sf.info(estimate_file).samplerate)
        else:
            sampling_frequency_hz = 16000.0

    # Calculate actual energy ratios
    isr, sir, sar, sdr = calculate_bss_eval_energy_ratios(
        waveforms.true_target,
        waveforms.target_distortion,
        waveforms.interference,
        waveforms.artifacts
    )

    # Compute actual auditory features
    signals = (waveforms.true_target, waveforms.target_distortion, waveforms.interference, waveforms.artifacts)
    q_target, q_interf, q_artif, q_global = calculate_auditory_quality_features(signals, sampling_frequency_hz)

    aud_features = torch.stack([
        q_global if isinstance(q_global, torch.Tensor) else torch.tensor(q_global, device=device, dtype=dtype),
        q_target if isinstance(q_target, torch.Tensor) else torch.tensor(q_target, device=device, dtype=dtype),
        q_interf if isinstance(q_interf, torch.Tensor) else torch.tensor(q_interf, device=device, dtype=dtype),
        q_artif if isinstance(q_artif, torch.Tensor) else torch.tensor(q_artif, device=device, dtype=dtype)
    ])
    clamped = torch.clamp(aud_features, -1.0 + 1e-15, 1.0 - 1e-15)
    log_mapped = torch.clamp(torch.log((1.0 + clamped) / (1.0 - clamped)), -5.5, 5.5)

    # The MLP weights must match the feature dtype (the auditory pipeline runs in
    # float64), independent of the caller's input precision (e.g. a float32 tensor).
    param_dtype = log_mapped.dtype
    scores = []
    for task_idx in range(4):
        params = _get_model_parameters_cached(task_idx, device, param_dtype)
        selected = log_mapped[params['selec']]
        scores.append(
            evaluate_neural_network_mapping_torch(selected, params['w'], params['b'], params['v'], params['a']))

    perceptual_scores = PerceptualSeparationScores(
        overall_perceptual_score=scores[0],
        target_perceptual_score=scores[1],
        interference_perceptual_score=scores[2],
        artifact_perceptual_score=scores[3],
        source_to_distortion_ratio=sdr,
        source_to_spatial_distortion_ratio=isr,
        source_to_interference_ratio=sir,
        source_to_artifacts_ratio=sar
    )

    # Step 4: Attach waveforms if requested
    if return_decomposition:
        perceptual_scores.decomposition_waveforms = waveforms
        perceptual_scores.decomposition_files = decomp_result.file_paths

    return perceptual_scores


# Cache weight transformations to prevent recurring numpy reads
_PARAMS_CACHE = {}


def _get_model_parameters_cached(task_idx: int, device, dtype):
    key = (task_idx, device, dtype)
    if key not in _PARAMS_CACHE:
        _PARAMS_CACHE[key] = _get_model_parameters_torch(task_idx, device, dtype)
    return _PARAMS_CACHE[key]
