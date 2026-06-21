"""
PEASS Execution Dispatcher
File path: peass/core/dispatch.py
"""

import pathlib

import numpy as np

from .base import BaseBackend
from ..config import DecompositionConfiguration
from ..config import DecompositionResult
from ..config import PerceptualSeparationScores


def resolve_backend(estimate_signal: any, source_signals: any = None) -> BaseBackend:
    """
    Inspects input types and resolves the appropriate execution backend.
    Without PyTorch registered yet, this strictly routes to NumpyBackend.
    """
    from ..backend_numpy import NumpyBackend
    return NumpyBackend()


def decompose_distortion_components(
        source_files: list[str | pathlib.Path | np.ndarray],
        estimate_file: str | pathlib.Path | np.ndarray,
        configuration: DecompositionConfiguration | None = None,
        sampling_frequency_hz: float | None = None
) -> DecompositionResult:
    backend = resolve_backend(estimate_file, source_files)
    return backend.decompose_distortion_components(source_files, estimate_file, configuration, sampling_frequency_hz)


def calculate_bss_eval_energy_ratios(
        true_source: np.ndarray,
        target_distortion: np.ndarray,
        interference: np.ndarray,
        artifacts: np.ndarray
) -> tuple[float, float, float, float]:
    backend = resolve_backend(true_source)
    return backend.calculate_bss_eval_energy_ratios(true_source, target_distortion, interference, artifacts)


def calculate_auditory_quality_features(
        decomposition_signals: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        sampling_frequency_hz: float = 16000.0
) -> tuple[float, float, float, float]:
    backend = resolve_backend(decomposition_signals[0])
    return backend.calculate_auditory_quality_features(decomposition_signals, sampling_frequency_hz)


def predict_perceptual_evaluation_scores(
        original_files: list[str | pathlib.Path | np.ndarray],
        estimate_file: str | pathlib.Path | np.ndarray,
        configuration: DecompositionConfiguration | None = None,
        sampling_frequency_hz: float | None = None,
        return_decomposition: bool = False
) -> PerceptualSeparationScores:
    backend = resolve_backend(estimate_file, original_files)
    return backend.predict_perceptual_evaluation_scores(original_files, estimate_file, configuration,
                                                        sampling_frequency_hz, return_decomposition)
