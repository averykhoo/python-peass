"""
PEASS PyTorch Execution Backend
File path: peass/backend_torch/__init__.py
"""

from .decomposition import decompose_distortion_components as _decomp
from .metrics import calculate_auditory_quality_features as _aud_feat
from .metrics import calculate_bss_eval_energy_ratios as _bss_eval
from .predictor import predict_perceptual_evaluation_scores as _pred
from ..core.base import BaseBackend


class TorchBackend(BaseBackend):
    """Execution backend utilizing native PyTorch for GPU acceleration and differentiability."""

    def decompose_distortion_components(self, source_files, estimate_file, configuration=None,
                                        sampling_frequency_hz=None):
        return _decomp(source_files, estimate_file, configuration, sampling_frequency_hz)

    def calculate_bss_eval_energy_ratios(self, true_source, target_distortion, interference, artifacts):
        return _bss_eval(true_source, target_distortion, interference, artifacts)

    def calculate_auditory_quality_features(self, decomposition_signals, sampling_frequency_hz=16000.0):
        return _aud_feat(decomposition_signals, sampling_frequency_hz)

    def predict_perceptual_evaluation_scores(self, original_files, estimate_file, configuration=None,
                                             sampling_frequency_hz=None, return_decomposition=False):
        return _pred(original_files, estimate_file, configuration, sampling_frequency_hz, return_decomposition)
