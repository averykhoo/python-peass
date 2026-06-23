"""
PEASS Configuration and Data Structures
"""

from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import TYPE_CHECKING

import numpy as np

# OPTIONAL DEPENDENCY TYPING:
# Resolves PyCharm warnings without introducing PyTorch import requirements at runtime!
if TYPE_CHECKING:
    import torch

    TensorOrArray = np.ndarray | torch.Tensor
else:
    TensorOrArray = np.ndarray  # = Any


class ModulationProcessingType(Enum):
    """Defines the type of modulation processing used in the auditory model."""
    LOWPASS = auto()
    FILTERBANK = auto()


@dataclass(slots=True)
class DecomposedWaveforms:
    """Holds the in-memory arrays for the decomposed physical components."""
    true_target: TensorOrArray
    target_distortion: TensorOrArray
    interference: TensorOrArray
    artifacts: TensorOrArray


@dataclass(slots=True)
class DecomposedFilePaths:
    """Holds the absolute file paths to the generated WAV files on disk."""
    true_target: str
    target_distortion: str
    interference: str
    artifacts: str


@dataclass(slots=True)
class DecompositionResult:
    """Wrapper holding both the arrays and optional file paths of a decomposition."""
    waveforms: DecomposedWaveforms
    file_paths: DecomposedFilePaths | None = None


@dataclass(slots=True)
class DecompositionConfiguration:
    """Structural configurations for the subband least-squares windowing."""
    destination_directory: str = "./"
    use_two_stage_projection: bool = False
    frame_length_seconds: float = 0.5
    filter_length_seconds: float = 0.04
    shade_in_milliseconds: float = 10.0
    shade_out_milliseconds: float = 10.0
    segmentation_factor: int = 1


@dataclass(slots=True)
class PerceptualSeparationScores:
    """Final assessment metrics representing the predicted subjective evaluation."""
    overall_perceptual_score: float
    target_perceptual_score: float
    interference_perceptual_score: float
    artifact_perceptual_score: float
    source_to_distortion_ratio: float
    source_to_spatial_distortion_ratio: float
    source_to_interference_ratio: float
    source_to_artifacts_ratio: float
    decomposition_waveforms: DecomposedWaveforms | None = None
    decomposition_files: DecomposedFilePaths | None = None
