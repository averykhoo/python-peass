"""
PEASS Test Suite - Backend Dispatcher
File path: tests/unit/test_dispatch.py
"""

import subprocess
import sys
import textwrap

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
@pytest.mark.skipif(not _HAS_TORCH, reason="Nothing to verify when torch is not installed at all.")
def test_numpy_dispatch_does_not_import_torch(project_root):
    """
    Resolving a NumPy input must leave torch unimported, even when it is installed.

    Dispatch answers "is this a tensor?" from `sys.modules`, which is exact: a
    torch.Tensor cannot exist unless the caller already imported torch. Importing it
    instead costs ~1s on first dispatch, and on Windows pulls a second Intel OpenMP
    runtime in alongside conda MKL's, which aborts the interpreter outright
    (OMP: Error #15) -- from pure-NumPy usage that never asked for torch.

    Runs in a subprocess because torch is unavoidably imported in this one.
    """
    script = textwrap.dedent(
        """
        import sys

        import numpy as np

        from peass.backend_numpy import NumpyBackend
        from peass.core.dispatch import resolve_backend

        assert isinstance(resolve_backend(np.zeros((1000, 1))), NumpyBackend)
        assert isinstance(resolve_backend(np.zeros((1000, 1)), [np.zeros((1000, 1))]), NumpyBackend)
        assert "torch" not in sys.modules, "dispatch imported torch to resolve a NumPy input"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
