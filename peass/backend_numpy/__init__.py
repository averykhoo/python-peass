"""
PEASS NumPy Execution Backend
File path: peass/backend_numpy/__init__.py
"""

import pathlib

import numpy as np

from .decomposition import decompose_distortion_components as _decomp
from .metrics import calculate_auditory_quality_features as _aud_feat
from .metrics import calculate_bss_eval_energy_ratios as _bss_eval
from .predictor import predict_perceptual_evaluation_scores as _pred
from ..config import DecompositionConfiguration
from ..config import DecompositionResult
from ..config import PerceptualSeparationScores
from ..core.base import BaseBackend


class NumpyBackend(BaseBackend):
    """Concrete execution backend utilizing CPU-optimized NumPy, SciPy, and Numba."""

    def decompose_distortion_components(
            self,
            source_files: list[str | pathlib.Path | np.ndarray],
            estimate_file: str | pathlib.Path | np.ndarray,
            configuration: DecompositionConfiguration | None = None,
            sampling_frequency_hz: float | None = None
    ) -> DecompositionResult:
        return _decomp(source_files, estimate_file, configuration, sampling_frequency_hz)

    def calculate_bss_eval_energy_ratios(
            self,
            true_source: np.ndarray,
            target_distortion: np.ndarray,
            interference: np.ndarray,
            artifacts: np.ndarray
    ) -> tuple[float, float, float, float]:
        return _bss_eval(true_source, target_distortion, interference, artifacts)

    def calculate_auditory_quality_features(
            self,
            decomposition_signals: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            sampling_frequency_hz: float = 16000.0
    ) -> tuple[float, float, float, float]:
        return _aud_feat(decomposition_signals, sampling_frequency_hz)

    def predict_perceptual_evaluation_scores(
            self,
            original_files: list[str | pathlib.Path | np.ndarray],
            estimate_file: str | pathlib.Path | np.ndarray,
            configuration: DecompositionConfiguration | None = None,
            sampling_frequency_hz: float | None = None,
            return_decomposition: bool = False
    ) -> PerceptualSeparationScores:
        return _pred(original_files, estimate_file, configuration, sampling_frequency_hz, return_decomposition)
