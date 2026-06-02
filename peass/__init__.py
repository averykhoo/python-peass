"""
python-peass: Perceptual Evaluation methods for Audio Source Separation
A modern, Pythonic port of the PEASS v2.0.1 toolkit.
"""

__version__ = "2.0.1.2"  # matches peass version, with one more segment for me to edit

from .config import DecomposedFilePaths
from .config import DecomposedWaveforms
from .config import DecompositionConfiguration
from .config import DecompositionResult
from .config import ModulationProcessingType
from .config import PerceptualSeparationScores
from .decomposition import decompose_distortion_components
from .metrics import calculate_auditory_quality_features
from .metrics import calculate_bss_eval_energy_ratios
from .predictor import predict_perceptual_evaluation_scores

__all__ = [
    "DecomposedFilePaths",
    "DecomposedWaveforms",
    "DecompositionConfiguration",
    "DecompositionResult",
    "ModulationProcessingType",
    "PerceptualSeparationScores",
    "predict_perceptual_evaluation_scores",
    "decompose_distortion_components",
    "calculate_bss_eval_energy_ratios",
    "calculate_auditory_quality_features",
]
