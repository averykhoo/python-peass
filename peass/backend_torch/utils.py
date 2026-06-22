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
    Heavily optimized for batched processing of continuous N-dimensional sequences
    using transposed convolutions to mathematically skip zero-insertions.
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
    K = h_padded.shape[0]

    # Standard batched 1D convolution layout: (Batch, Channels, Length)
    x_batched = x_flat.unsqueeze(1)

    if up_reduced == 1 and down_reduced > 1:
        # OPTIMIZATION 1: Native Downsampling via Strided Convolution
        pad_left = K - 1
        required_conv_len = n_pre_remove + out_len
        L_required = (required_conv_len - 1) * down_reduced + K
        pad_right = max(0, L_required - (in_len + pad_left))

        x_padded = F.pad(x_batched, (pad_left, pad_right))

        # FIR filtering corresponds to correlation, which requires flipped weights
        weights_flipped = h_padded.flip(-1).view(1, 1, K)

        y_conv = F.conv1d(x_padded, weights_flipped, stride=down_reduced)
        y_flat = y_conv[:, 0, n_pre_remove: n_pre_remove + out_len]

    else:
        # OPTIMIZATION 2: Native Upsampling via Transposed Convolution
        # Transposed convolution naturally applies the kernel without flipping it,
        # perfectly matching zero-insertion followed by standard FIR convolution.
        weights = h_padded.view(1, 1, K)
        y_upsampled = F.conv_transpose1d(x_batched, weights, stride=up_reduced)

        start_idx = n_pre_remove * down_reduced
        end_idx = (n_pre_remove + out_len) * down_reduced

        pad_needed = max(0, end_idx - y_upsampled.shape[-1])
        if pad_needed > 0:
            y_upsampled = F.pad(y_upsampled, (0, pad_needed))

        y_flat = y_upsampled[:, 0, start_idx: end_idx: down_reduced]

    # Reconstruct original shape
    y = y_flat.view(*shape_prefix, out_len).transpose(axis, -1)

    return y
