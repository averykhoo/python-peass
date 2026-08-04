"""
PEASS Execution Dispatcher
File path: peass/core/dispatch.py
"""

import pathlib
import sys
from typing import Any

import numpy as np

from .base import BaseBackend
from ..config import DecompositionConfiguration
from ..config import DecompositionResult
from ..config import PerceptualSeparationScores


def resolve_backend(estimate_signal: Any, source_signals: Any = None) -> BaseBackend:
    """
    Inspects input types and resolves the appropriate execution backend.
    Enforces device and precision homogeneity for PyTorch tensors.
    """
    # Look torch up in sys.modules rather than importing it. A torch.Tensor cannot
    # exist unless the caller already imported torch to construct it, so an absent
    # entry here is proof the input is not a tensor -- the check is exact, not a
    # heuristic. Importing it instead would drag the torch runtime into every
    # NumPy-only process, which costs ~1s on first dispatch and, on Windows, loads a
    # second Intel OpenMP runtime alongside conda MKL's; that combination aborts the
    # interpreter outright (see "Intel OpenMP conflict on Windows" in README.md).
    # This matches the TYPE_CHECKING-guarded import in peass/config.py: torch is an
    # optional extra and must not be imported at runtime unless actually in use.
    torch = sys.modules.get("torch")

    if torch is not None and isinstance(estimate_signal, torch.Tensor):
        devices = {estimate_signal.device}
        dtypes = {estimate_signal.dtype}

        if source_signals is not None:
            for s in source_signals:
                if not isinstance(s, torch.Tensor):
                    raise TypeError("Mixed inputs: estimate_file is a Tensor, but a source_file is not.")
                devices.add(s.device)
                dtypes.add(s.dtype)

        if len(devices) > 1:
            raise RuntimeError(f"Device mismatch: Tensors must be on the same device. Found {devices}")
        if len(dtypes) > 1:
            raise RuntimeError(f"Precision mismatch: Tensors must have the same dtype. Found {dtypes}")

        from ..backend_torch import TorchBackend
        return TorchBackend()

    # Fallback to NumPy optimized backend
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
