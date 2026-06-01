"""
PEASS Configuration and Data Structures
"""

import sys
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Optional

# Dynamically enable slots only on Python 3.10+
# TODO: drop py3.9 support and just always include slots
_DATACLASS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}


class ModulationProcessingType(Enum):
    """Defines the type of modulation processing used in the auditory model."""
    LOWPASS = auto()
    FILTERBANK = auto()


@dataclass(**_DATACLASS_KWARGS)
class DecomposedWaveforms:
    """Holds the in-memory NumPy arrays for the decomposed physical components."""
    true_target: np.ndarray
    target_distortion: np.ndarray
    interference: np.ndarray
    artifacts: np.ndarray


@dataclass(**_DATACLASS_KWARGS)
class DecomposedFilePaths:
    """Holds the absolute file paths to the generated WAV files on disk."""
    true_target: str
    target_distortion: str
    interference: str
    artifacts: str


@dataclass(**_DATACLASS_KWARGS)
class DecompositionResult:
    """Wrapper holding both the arrays and optional file paths of a decomposition."""
    waveforms: DecomposedWaveforms
    file_paths: Optional[DecomposedFilePaths] = None


@dataclass(**_DATACLASS_KWARGS)
class DecompositionConfiguration:
    """Structural configurations for the subband least-squares windowing."""
    destination_directory: str = "./"
    use_two_stage_projection: bool = False
    frame_length_seconds: float = 0.5
    filter_length_seconds: float = 0.04
    shade_in_milliseconds: float = 10.0
    shade_out_milliseconds: float = 10.0
    segmentation_factor: int = 1


@dataclass(**_DATACLASS_KWARGS)
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
    decomposition_waveforms: Optional[DecomposedWaveforms] = None
    decomposition_files: Optional[DecomposedFilePaths] = None
