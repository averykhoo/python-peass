"""
PEASS Test Suite - Backend Dispatcher
File path: tests/unit/test_dispatch.py
"""

import numpy as np
import pytest

from peass.backend_numpy import NumpyBackend
from peass.core.dispatch import resolve_backend

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.unit
def test_resolve_backend_numpy_for_ndarray():
    """NumPy arrays always resolve to the NumPy backend."""
    est = np.zeros((1000, 1))
    assert isinstance(resolve_backend(est), NumpyBackend)
    assert isinstance(resolve_backend(est, [np.zeros((1000, 1))]), NumpyBackend)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch is not installed in this environment.")
def test_resolve_backend_torch_for_tensor():
    """Tensor inputs resolve to the torch backend."""
    from peass.backend_torch import TorchBackend

    est = torch.zeros((1000, 1), dtype=torch.float64)
    src = torch.zeros((1000, 1), dtype=torch.float64)
    assert isinstance(resolve_backend(est, [src]), TorchBackend)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch is not installed in this environment.")
def test_resolve_backend_rejects_mixed_inputs():
    """A tensor estimate with a non-tensor source is a TypeError."""
    est = torch.zeros((1000, 1), dtype=torch.float64)
    with pytest.raises(TypeError):
        resolve_backend(est, [np.zeros((1000, 1))])


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch is not installed in this environment.")
def test_resolve_backend_rejects_dtype_mismatch():
    """Tensors of differing dtype are rejected."""
    est = torch.zeros((1000, 1), dtype=torch.float64)
    src = torch.zeros((1000, 1), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        resolve_backend(est, [src])


@pytest.mark.unit
@pytest.mark.skipif(not (_HAS_TORCH and torch.cuda.is_available()),
                    reason="CUDA not available for device-mismatch check.")
def test_resolve_backend_rejects_device_mismatch():
    """Tensors on differing devices are rejected."""
    est = torch.zeros((1000, 1), dtype=torch.float64, device="cpu")
    src = torch.zeros((1000, 1), dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError):
        resolve_backend(est, [src])
