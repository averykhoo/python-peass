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
    # NOT IMPLEMENTED: only the default of 1 is accepted; anything else raises in
    # `__post_init__` rather than being silently ignored. The field exists so that
    # MATLAB's `options.segmentationFactor` maps onto a recognizable name instead of
    # an opaque "unexpected keyword argument". MATLAB implements the split path in
    # `extractDistortionComponents.m` (the `segmentationFactor > 1` branch at
    # ~lines 107-110, dispatching to `aux_segmentAndDecompose` / `aux_cutWav` /
    # `aux_mergeWav` at ~lines 270-386): it chops the signal into overlapping
    # segments, decomposes each independently with the shades suppressed on interior
    # edges, and overlap-adds them under a periodic Hann window with normalization by
    # the accumulated window. It is purely a peak-memory relief valve. See ARCHIVE.md
    # for why this was declined rather than ported, plus the full port spec.
    segmentation_factor: int = 1
    # Anti-aliasing FIR half-length as a multiple of the up/down ratio, used for
    # the polyphase resampling inside the decomposition. 10 matches SciPy/MATLAB
    # (near bit-exact reference agreement); lower values (e.g. 3) trade accuracy
    # for speed (~-6% component energy, correlation ~0.99 at 3x). Values below 1
    # are rejected in `__post_init__`; see the check for why.
    resample_filter_half_length_factor: int = 10

    def __post_init__(self) -> None:
        # The fast `_polyphase_decimate` path does not raise below 1 -- it returns
        # finite numbers that disagree with its own `_polyphase_decimate_padded`
        # reference by O(1). Measured 2026-08-17 at hf = 0: deviations 0.39 to 2.13
        # across 33 of the swept (down, in_len, rows, dtype) combinations, against
        # 2.22e-16 for hf >= 1. The grid algebra needs
        # `right_pad >= (hf-1)*down + 1 > 0`, i.e. `hf >= 1`. The gradient path
        # routes to the padded reference instead, so below 1 the no-grad and grad
        # paths silently disagree as well.
        #
        # Raising here is a deliberate public behaviour change: callers who pass 0
        # get a loud error where they used to get silently wrong output. Nothing in
        # the library passes it (default 10, and the comment above contemplates
        # lowering only to ~3), so it has never been hit -- but "nothing in the
        # library passes that" is a statement about the library, not about the API,
        # and this dataclass is public.
        if self.resample_filter_half_length_factor < 1:
            raise ValueError(
                f"resample_filter_half_length_factor="
                f"{self.resample_filter_half_length_factor} is invalid; it must be >= 1. "
                f"Below 1 the polyphase resampler's grid algebra breaks down and the fast "
                f"path returns finite but O(1)-wrong output instead of failing, and the "
                f"gradient path (which uses the padded reference) disagrees with it. "
                f"The default is 10, which matches SciPy/MATLAB; 3 is the lowest value "
                f"that has been characterized as a usable accuracy/speed trade."
            )
        if self.segmentation_factor != 1:
            raise NotImplementedError(
                f"segmentation_factor={self.segmentation_factor} is not supported; only 1 is. "
                f"In MATLAB PEASS this option splits the signal into overlapping segments that "
                f"are decomposed separately and overlap-added, purely to relieve peak memory "
                f"('increase this integer if you experienced out of memory problems'). That path "
                f"was never ported, so the signal is always decomposed in one piece."
            )


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
