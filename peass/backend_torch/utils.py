"""
PEASS PyTorch Sub-Utilities
File path: peass/backend_torch/utils.py
"""
import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from scipy.signal import firwin


def smoothmax(x: torch.Tensor, threshold: float | torch.Tensor, k: float = 1000.0) -> torch.Tensor:
    """
    Temperature-scaled SmoothMax (Softplus approximation).
    Provides exact mathematical parity to `max(x, threshold)` while maintaining
    smooth, active gradients for neural network backpropagation.
    """
    return F.softplus(k * (x - threshold)) / k + threshold


# -----------------------------------------------------------------------------
# HIGH-SPEED CACHED FILTER DESIGNER
# -----------------------------------------------------------------------------
@lru_cache(maxsize=256)
def get_resample_filter_torch(up: int, down: int, dtype: torch.dtype, device: torch.device) -> tuple:
    """
    Designs and caches the Kaiser resample filter on the fly to eliminate
    CPU-to-Device transfer overhead.
    """
    g = math.gcd(up, down)
    up_reduced = up // g
    down_reduced = down // g

    max_len = max(up_reduced, down_reduced)
    half_len = 3 * max_len
    n_filt = 2 * half_len + 1

    # Design filter on CPU, push to the correct target device/dtype once
    h_numpy = firwin(n_filt, 1.0 / max_len, window=('kaiser', 5.0)) * up_reduced
    h = torch.tensor(h_numpy, dtype=dtype, device=device)

    n_pre_pad = (down_reduced - half_len % down_reduced) % down_reduced
    h_padded = F.pad(h, (n_pre_pad, 0))
    n_pre_remove = (half_len + n_pre_pad) // down_reduced

    return h_padded, up_reduced, down_reduced, n_pre_remove


def fast_resample_poly_torch(x: torch.Tensor, up: int, down: int, axis: int = -1) -> torch.Tensor:
    """
    Native PyTorch polyphase resampler replicating SciPy's upfirdn using Conv1D.
    Directly pads the input to match SciPy's off-end filter sliding.
    Heavily optimized for batched processing of continuous N-dimensional sequences.
    """
    if up == down:
        return x

    # Fetch pre-calculated and cached filter tensor instantly
    h_padded, up_reduced, down_reduced, n_pre_remove = get_resample_filter_torch(
        up, down, x.dtype, x.device
    )

    in_len = x.shape[axis]
    out_len = math.ceil(in_len * up_reduced / down_reduced)

    # Move target axis to the end for conv1d
    x_moved = x.transpose(axis, -1)
    shape_prefix = x_moved.shape[:-1]

    # Flatten prefix dimensions into Batch
    x_flat = x_moved.reshape(-1, in_len)
    B = x_flat.shape[0]

    # Zero-insertion Upsampling
    x_up = torch.zeros((B, in_len * up_reduced), dtype=x.dtype, device=x.device)
    x_up[:, ::up_reduced] = x_flat

    # Calculate and apply input padding (causal padding at start, remainder at end)
    K = h_padded.shape[0]
    pad_left = K - 1

    required_conv_len = n_pre_remove + out_len
    L_required = (required_conv_len - 1) * down_reduced + K
    pad_right = max(0, L_required - (x_up.shape[1] + pad_left))

    x_up_padded = F.pad(x_up, (pad_left, pad_right))

    # Downsampling via Stride Convolution (Flipped weights for true convolution)
    weights = h_padded.flip(-1).view(1, 1, -1)
    x_conv = F.conv1d(x_up_padded.unsqueeze(1), weights, stride=down_reduced)[..., 0, :]

    # Slice and reconstruct shapes
    y_flat = x_conv[:, n_pre_remove: n_pre_remove + out_len]
    y = y_flat.view(*shape_prefix, out_len).transpose(axis, -1)

    return y
