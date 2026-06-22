"""
PEASS PyTorch Sub-Utilities
File path: peass/backend_torch/utils.py
"""
import math

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


def fast_resample_poly_torch(x: torch.Tensor, up: int, down: int, axis: int = -1) -> torch.Tensor:
    """
    Native PyTorch polyphase resampler replicating SciPy's upfirdn using Conv1D.
    Directly pads the input to match SciPy's off-end filter sliding.
    """
    if up == down:
        return x

    g = math.gcd(up, down)
    up //= g
    down //= g

    max_len = max(up, down)
    half_len = 3 * max_len
    n_filt = 2 * half_len + 1

    # Design Kaiser filter on CPU, push to Target Device
    h_numpy = firwin(n_filt, 1.0 / max_len, window=('kaiser', 5.0)) * up
    h = torch.tensor(h_numpy, dtype=x.dtype, device=x.device)

    n_pre_pad = (down - half_len % down) % down
    h_padded = F.pad(h, (n_pre_pad, 0))
    n_pre_remove = (half_len + n_pre_pad) // down

    in_len = x.shape[axis]
    out_len = math.ceil(in_len * up / down)

    # Move target axis to the end for conv1d
    x_moved = x.transpose(axis, -1)
    shape_prefix = x_moved.shape[:-1]

    # Flatten prefix dimensions into Batch
    x_flat = x_moved.reshape(-1, in_len)

    # Zero-insertion Upsampling
    B = x_flat.shape[0]
    x_up = torch.zeros((B, in_len * up), dtype=x.dtype, device=x.device)
    x_up[:, ::up] = x_flat

    # Calculate and apply input tail padding to prevent early Conv1D truncation
    required_len = n_pre_remove * down + out_len * down + h_padded.shape[0]
    pad_needed = required_len - x_up.shape[1]
    if pad_needed > 0:
        x_up = F.pad(x_up, (0, pad_needed))

    # Downsampling via Stride Convolution
    weights = h_padded.view(1, 1, -1)
    x_conv = F.conv1d(x_up.unsqueeze(1), weights, stride=down).squeeze(1)

    # Slice and reconstruct shapes
    y_flat = x_conv[:, n_pre_remove: n_pre_remove + out_len]
    y = y_flat.view(*shape_prefix, out_len).transpose(axis, -1)

    return y
