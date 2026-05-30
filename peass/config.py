"""
PEASS Configuration and Data Structures
"""

from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Optional

import numpy as np


class ModulationProcessingType(Enum):
    """Defines the type of modulation processing used in the auditory model."""
    LOWPASS = auto()
    FILTERBANK = auto()


@dataclass
class DecomposedWaveforms:
    """Holds the in-memory NumPy arrays for the decomposed physical components."""
    true_target: np.ndarray
    target_distortion: np.ndarray
    interference: np.ndarray
    artifacts: np.ndarray


@dataclass
class DecomposedFilePaths:
    """Holds the absolute file paths to the generated WAV files on disk."""
    true_target: str
    target_distortion: str
    interference: str
    artifacts: str


@dataclass
class DecompositionResult:
    """Wrapper holding both the arrays and optional file paths of a decomposition."""
    waveforms: DecomposedWaveforms
    file_paths: Optional[DecomposedFilePaths] = None


@dataclass
class DecompositionConfiguration:
    """Structural configurations for the subband least-squares windowing."""
    destination_directory: str = "./"
    use_two_stage_projection: bool = False
    frame_length_seconds: float = 0.5
    filter_length_seconds: float = 0.04
    shade_in_milliseconds: float = 10.0
    shade_out_milliseconds: float = 10.0
    segmentation_factor: int = 1


@dataclass
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

    def __contains__(self, key: str) -> bool:
        """Determines membership safely to prevent index iteration errors."""
        return key in {
            "OPS", "TPS", "IPS", "APS", "SDR", "ISR", "SIR", "SAR",
            "decomposition_arrays", "decomposition_files"
        }

    def __iter__(self):
        """Allows safe verification checks without raising KeyErrors."""
        keys = ["OPS", "TPS", "IPS", "APS", "SDR", "ISR", "SIR", "SAR"]
        if self.decomposition_waveforms is not None:
            keys.append("decomposition_arrays")
        if self.decomposition_files is not None:
            keys.append("decomposition_files")
        return iter(keys)

    def __getitem__(self, key: str):
        """Allows dictionary-like access for full backward-compatibility with legacy tests."""
        mapping = {
            "OPS": self.overall_perceptual_score,
            "TPS": self.target_perceptual_score,
            "IPS": self.interference_perceptual_score,
            "APS": self.artifact_perceptual_score,
            "SDR": self.source_to_distortion_ratio,
            "ISR": self.source_to_spatial_distortion_ratio,
            "SIR": self.source_to_interference_ratio,
            "SAR": self.source_to_artifacts_ratio,
        }
        if key in mapping:
            return mapping[key]
        if key == "decomposition_arrays" and self.decomposition_waveforms is not None:
            return {
                "true_target":       self.decomposition_waveforms.true_target,
                "target_distortion": self.decomposition_waveforms.target_distortion,
                "interference":      self.decomposition_waveforms.interference,
                "artifacts":         self.decomposition_waveforms.artifacts,
            }
        if key == "decomposition_files" and self.decomposition_files is not None:
            return {
                "true_target":       self.decomposition_files.true_target,
                "target_distortion": self.decomposition_files.target_distortion,
                "interference":      self.decomposition_files.interference,
                "artifacts":         self.decomposition_files.artifacts,
            }
        raise KeyError(key)
