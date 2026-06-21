"""
PEASS Abstract Backend Contract
File path: peass/core/base.py
"""

import pathlib
from abc import ABC
from abc import abstractmethod

import numpy as np

from ..config import DecompositionConfiguration
from ..config import DecompositionResult
from ..config import PerceptualSeparationScores


class BaseBackend(ABC):
    """Abstract interface defining required operations for PEASS execution backends."""

    @abstractmethod
    def decompose_distortion_components(
            self,
            source_files: list[str | pathlib.Path | np.ndarray],
            estimate_file: str | pathlib.Path | np.ndarray,
            configuration: DecompositionConfiguration | None = None,
            sampling_frequency_hz: float | None = None
    ) -> DecompositionResult:
        ...

    @abstractmethod
    def calculate_bss_eval_energy_ratios(
            self,
            true_source: np.ndarray,
            target_distortion: np.ndarray,
            interference: np.ndarray,
            artifacts: np.ndarray
    ) -> tuple[float, float, float, float]:
        ...

    @abstractmethod
    def calculate_auditory_quality_features(
            self,
            decomposition_signals: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            sampling_frequency_hz: float = 16000.0
    ) -> tuple[float, float, float, float]:
        ...

    @abstractmethod
    def predict_perceptual_evaluation_scores(
            self,
            original_files: list[str | pathlib.Path | np.ndarray],
            estimate_file: str | pathlib.Path | np.ndarray,
            configuration: DecompositionConfiguration | None = None,
            sampling_frequency_hz: float | None = None,
            return_decomposition: bool = False
    ) -> PerceptualSeparationScores:
        ...
