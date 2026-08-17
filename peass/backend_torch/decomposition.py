"""
PEASS PyTorch Least-Squares Decomposer
File path: peass/backend_torch/decomposition.py
"""
import math
import pathlib
from functools import lru_cache

import soundfile as sf
import torch
import torch.nn.functional as F

from .gammatone import GammatoneAnalyzerTorch
from .gammatone import GammatoneSynthesizerTorch
from .gammatone import _get_synthesizer_params_torch
from .utils import DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
from .utils import fast_resample_poly_torch
from ..config import DecomposedFilePaths
from ..config import DecomposedWaveforms
from ..config import DecompositionResult


@lru_cache(maxsize=16)
def _get_analysis_mod_matrix_torch(fs: float, max_len: int, cfs_tuple: tuple, device_str: str):
    device = torch.device(device_str)
    cfs = torch.tensor(cfs_tuple, device=device, dtype=torch.float64)
    time_steps = torch.arange(max_len, device=device, dtype=torch.float64)
    return torch.exp(-2j * math.pi / fs * cfs.unsqueeze(-1) * time_steps)


@lru_cache(maxsize=16)
def _get_synthesis_alignment_matrix_torch(fs: float, max_len: int, cfs_tuple: tuple, device_str: str,
                                          delay_sec: float, norms_tuple: tuple, coefs_tuple: tuple):
    """Synthesis modulation matrix pre-multiplied by the synthesizer's phase factors.

    ``exp(+2j*pi*fc*t/fs) * phase_factor[band]`` -- the two per-band constants the
    synthesis path applies back to back. Fusing them turns two full passes over the
    (62 MB mono / 123 MB stereo) subband block into one, and re-associating
    ``(x * mod) * phase`` into ``x * (mod * phase)`` costs ~1 ULP.

    Keyed on the phase factors' full dependency set (``delay_sec``, ``fs``, the center
    frequencies and the filter ``norms``/``coefs``) as well as the modulation's
    (``fs``, ``max_len``, center frequencies, device). Keying it on the modulation's
    arguments alone would silently serve a matrix built for a different delay.

    This *replaces* a separate cache of the bare modulation matrix rather than adding to
    one: the bare matrix had no other caller, so the resident cost is unchanged. The
    analysis-side cache above is untouched, and `_get_synthesizer_params_torch` is only
    read from here -- the product below is a fresh tensor, so neither cache entry is
    disturbed.
    """
    device = torch.device(device_str)
    cfs = torch.tensor(cfs_tuple, device=device, dtype=torch.float64)
    time_steps = torch.arange(max_len, device=device, dtype=torch.float64)
    modulation = torch.exp(2j * math.pi / fs * cfs.unsqueeze(1) * time_steps)

    _, phase_factors, _ = _get_synthesizer_params_torch(
        delay_sec, fs, cfs_tuple, norms_tuple, coefs_tuple, device_str
    )
    return modulation * phase_factors.unsqueeze(-1)


def _solve_hermitian_batch(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    r"""Solve ``gram @ x = rhs`` for a batch of Hermitian positive-semidefinite Grams.

    ``gram`` is ``T^H diag(w^2) T``: Hermitian and positive semidefinite by
    construction, so a Cholesky factorization is the right solver. This used to call
    ``torch.linalg.pinv`` unconditionally, on the grounds that it degrades gracefully
    on rank-deficient frames without a host sync. It does, but it pays an SVD of every
    frame for it. Profiled over the 2338-matrix stack of the mono reference example:
    pinv 409 ms, ``pinv(hermitian=True)`` 318 ms, ``lstsq`` 92 ms, ``ldl`` 28 ms,
    Cholesky 21 ms.

    ``cholesky_ex`` reports failure per matrix in ``info`` rather than raising, which
    is exactly the rank-deficiency test the NumPy backend already performs with LAPACK
    ``?posv`` (``backend_numpy/decomposition.py``). Frames it rejects fall back to
    ``pinv``, so a genuinely singular frame still resolves to a minimum-norm solution.

    The ``info.any()`` test does force a device sync on CUDA. That is one sync per
    band per solve (32 bands), against an SVD per frame; the trade is not close.

    The ``1e-15`` diagonal ridge below is not a NumPy-backend quirk: it is MATLAB PEASS
    v2.0.1's own ``lambda`` in ``LSDecompose.m`` ("regularization parameter (useful when
    a sequence of zeroes occurs in sources s)"), transcribed in
    ``reference/LSDecompose.py`` and mirrored by ``backend_numpy/decomposition.py``. It
    is an UNSCALED absolute constant added to the raw diagonal -- MATLAB writes
    ``chol(gramSw + lambda*eye(size(gramSw)))``, with no trace or max normalization --
    and it goes in *before* the factorization, so the ``info`` test and the fallbacks
    below all see the same regularized matrix.

    It was dropped here on 2026-08-10 on the grounds that against a minimum eigenvalue
    of ~3.4e-10 it perturbs the solution by ~1e-5 relative. That is true of the
    *weights* and does not reach the *output*. Restoring it, measured 2026-08-15 on the
    MATLAB reference clip: ``true_target`` is bit-identical (it never touches the
    solve), and the other three components move by at most 5.9e-8 of their peak on the
    mono channel and 3.5e-7 stereo; correlation against the MATLAB gold WAVs moves by
    at most 3.0e-12 over all four components and both channels, and the worst
    *degradation* among the eight is 6.6e-13 (``artifacts`` CH0, deficit 3.3107e-6
    either way, against a 1e-4 floor); the eight reported scores move by at most 2.2e-7
    on the 0-100 scale and 1.6e-7 dB on the energy ratios, against 1.0 / 0.5
    characterization tolerances. What dropping it cost was correctness on
    rank-deficient input: on the tonal pair of
    ``tests/regression/test_differential_numpy_vs_torch.py``'s ``baseline_signals`` --
    where one gammatone band of a pure tone is a single complex exponential, so its
    Toeplitz block is numerically rank ~1 -- torch-vs-NumPy correlation was 0.016490 on
    ``target_distortion``, 0.016452 on ``interference`` and 0.966525 on ``artifacts``,
    with torch peaking at 3.90e+2 against NumPy's 1.56. With the ridge they are
    0.999991, 0.999995 and 0.998739, and the torch peak is 1.559.

    Written as ``gram + lam*eye`` rather than an in-place diagonal ``add_`` so the
    autograd path stays unambiguous; the eye is at most 42x42 here.

    One consequence, and it is the correct one: a *silent* frame's Gram is identically
    zero, and ``1e-15 * I`` is positive definite, so ``cholesky_ex`` now succeeds on it
    instead of routing it to ``pinv``. That agrees, because ``gram`` and ``rhs`` are
    assembled from the same Toeplitz factor -- ``gram = T^H diag(w^2) T`` and
    ``rhs = T^H diag(w^2) est`` -- so a zero ``T`` forces ``rhs`` to be exactly zero and
    the ridged solve returns bitwise-zero weights, which is what ``pinv(0) @ 0`` gives
    and what MATLAB gives (``chol(0 + lambda*I)`` succeeds there too, and ``Sw'*se`` is
    zero). A frame that is merely rank deficient rather than zero still fails the ridged
    factorization and still takes the ``pinv`` path.

    The one way to reach a zero Gram with a nonzero ``rhs`` is underflow, so do not
    write down that the assembly cannot produce that state: at denormal source scale
    (``T`` ~ 1e-180) the Gram's entries square to ~1e-360 and flush to exactly 0.0 while
    ``rhs`` stays at ~9.9e-180, and the ridged solve duly returns weights ~9.9e-165.
    That is harmless in magnitude rather than impossible -- the projection
    ``T @ weights`` underflows straight back, measured bitwise 0.0 at ``T`` = 1e-180.
    """
    # MATLAB LSDecompose.m: chol(gramSw + lambda*eye(size(gramSw))), lambda = 1e-15.
    gram = gram + 1e-15 * torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    factor, info = torch.linalg.cholesky_ex(gram)
    if not bool(info.any()):
        return torch.cholesky_solve(rhs, factor)

    if gram.dim() == 2:
        # Single-frame twin: nothing to salvage, take the min-norm solve directly.
        return torch.linalg.pinv(gram, hermitian=True) @ rhs

    # A failed Cholesky leaves uninitialized values behind, and solving against them
    # can produce inf/NaN that poisons the autograd graph even though the values are
    # discarded below. Add the identity on exactly the failing frames -- the Gram is
    # PSD, so that is guaranteed positive definite -- and refactor the batch.
    size = gram.shape[-1]
    degenerate = info != 0
    shim = degenerate.to(gram.dtype).view(-1, 1, 1) * torch.eye(
        size, dtype=gram.dtype, device=gram.device
    )
    weights = torch.cholesky_solve(rhs, torch.linalg.cholesky_ex(gram + shim)[0]).clone()
    weights[degenerate] = torch.linalg.pinv(gram[degenerate], hermitian=True) @ rhs[degenerate]
    return weights


def get_real_dtype(dtype: torch.dtype) -> torch.dtype:
    """Helper returning the corresponding real-valued dtype counterpart of any dtype."""
    if dtype in (torch.complex128, torch.complex64):
        return torch.float64 if dtype == torch.complex128 else torch.float32
    return dtype


def validate_and_normalize_audio_torch(
        data: torch.Tensor,
        sampling_frequency_hz: float,
        name: str = "audio_data"
) -> torch.Tensor:
    """Enforces standard 2D (samples, channels) formatting for PyTorch tensors.

    Mirrors the NumPy backend's validation: at most 32 channels and a minimum
    physical duration of 50 ms so the boundary shading and subband frame
    decimation stay well-defined.
    """
    if data.dim() == 1:
        data = data.unsqueeze(1)

    num_samples, num_channels = data.shape
    if num_channels > 32:
        raise ValueError(
            f"Layout violation for '{name}'. Expected (samples, channels) with channels <= 32. "
            f"Found shape {data.shape}. Please transpose."
        )

    min_samples = int(50 * sampling_frequency_hz / 1000)
    if num_samples < min_samples:
        raise ValueError(
            f"Signal duration for '{name}' is too short ({num_samples} samples). "
            f"PEASS requires a minimum of {min_samples} samples to perform "
            f"the subband least-squares and overlap-add decomposition safely."
        )
    return data


def perform_least_squares_projection_torch(
        source_estimates: torch.Tensor,
        true_sources: torch.Tensor,
        filter_half_length: int,
        analysis_window: torch.Tensor
) -> torch.Tensor:
    """Computes weighted least-squares projections natively on GPU.

    Note: the production pipeline uses the fully-batched
    ``perform_time_varying_least_squares_projection_torch`` below; this
    single-frame solver is retained as the direct analogue of the NumPy
    per-frame solver for the differential parity tests.
    """
    filter_length = 2 * filter_half_length + 1
    num_sources = true_sources.shape[1]
    num_samples = source_estimates.shape[0]

    # Silence bypass optimization
    source_energy = torch.sum(
        true_sources.real ** 2 + true_sources.imag ** 2) if true_sources.is_complex() else torch.sum(true_sources ** 2)
    if source_energy < 1e-13:
        return torch.zeros((num_samples, source_estimates.shape[1], num_sources), dtype=source_estimates.dtype,
                           device=source_estimates.device)

    # Build Toeplitz matrix using fast 1D striding (unfolded directly without extra F.pad)
    strided_views = []
    for s_idx in range(num_sources):
        source_signal = true_sources[:, s_idx]
        view = source_signal.unfold(0, filter_length, 1).flip(-1)
        strided_views.append(view)

    toeplitz_matrix = torch.cat(strided_views, dim=1)

    # Compute Gram and RHS matrices with window weighting
    window_sq = (analysis_window ** 2).unsqueeze(1)

    # Complex support for analytic subbands
    Gram = toeplitz_matrix.conj().T @ (window_sq * toeplitz_matrix)
    RHS = toeplitz_matrix.conj().T @ (window_sq * source_estimates)

    # Same Cholesky-with-pinv-fallback solve as the batched path, so the two stay the
    # exact single-frame/batched twins the differential tests compare.
    projection_weights = _solve_hermitian_batch(Gram, RHS)
    projections = torch.zeros((num_samples, source_estimates.shape[1], num_sources), dtype=source_estimates.dtype,
                              device=source_estimates.device)
    weighted_diagonal = analysis_window.unsqueeze(1)

    for s_idx in range(num_sources):
        start = s_idx * filter_length
        end = (s_idx + 1) * filter_length
        projections[:, :, s_idx] = weighted_diagonal * (
                toeplitz_matrix[:, start:end] @ projection_weights[start:end, :]
        )

    return projections


def perform_time_varying_least_squares_projection_torch(
        source_estimates: torch.Tensor,
        true_sources: torch.Tensor,
        filter_length: int,
        window_length: int,
        hop_size: int
) -> torch.Tensor:
    """
    Time-varying subband least-squares solver.
    Fully vectorized with zero time-loops or source-loops.
    """
    filter_half_length = (filter_length - 1) // 2
    pad_length = filter_length - 1 + window_length - 1

    # Match the exact sequence-level padding
    true_sources = F.pad(true_sources, (0, 0, 0, pad_length))
    source_estimates = F.pad(source_estimates, (0, 0, 0, pad_length))

    total_samples, num_sources = true_sources.shape  # Padded length
    num_channels = source_estimates.shape[1]

    # Resolve real-valued dtypes for window calculations
    real_dtype = get_real_dtype(source_estimates.dtype)
    hann_window = torch.hann_window(window_length, periodic=True, device=source_estimates.device, dtype=real_dtype)
    analysis_window = torch.sqrt(torch.flip(hann_window, dims=[0]))
    synthesis_window = torch.sqrt(torch.flip(hann_window, dims=[0]))

    # Closed-form mathematical evaluation of NumFrames matching the sequential loop boundary
    NumFrames = max(0, int(math.floor((total_samples - 1.5 * window_length + 1) / hop_size)) + 1)

    if NumFrames == 0:
        return torch.zeros((total_samples - (window_length - 1), num_channels, num_sources), dtype=true_sources.dtype,
                           device=true_sources.device)

    # 1. Unfold and slice estimates and true sources across the temporal axis
    est_frames = source_estimates.unfold(0, window_length, hop_size).transpose(1, 2)[:NumFrames]

    # Unfolding true sources requires symmetric source overlap mapping.
    # To mathematically match the timeline alignment of sw:
    # We pad the beginning of true_sources with exactly filter_half_length zeros, and then unfold.
    true_sources_pad = F.pad(true_sources, (0, 0, filter_half_length, 0))
    src_frames = true_sources_pad.unfold(0, window_length + filter_length - 1, hop_size).transpose(1, 2)[:NumFrames]

    # 2. Build the batched Toeplitz representations
    src_unfold = src_frames.unfold(1, filter_length, 1).flip(
        -1)  # (NumFrames, window_length, num_sources, filter_length)
    toeplitz_batched = src_unfold.reshape(NumFrames, window_length, num_sources * filter_length)

    # 3. Solve the batched least squares system in a single step
    window_sq = (analysis_window ** 2).view(1, window_length, 1)

    Gram = toeplitz_batched.conj().transpose(1, 2) @ (window_sq * toeplitz_batched)
    RHS = toeplitz_batched.conj().transpose(1, 2) @ (window_sq * est_frames)

    weights = _solve_hermitian_batch(Gram, RHS)

    # weights shape: (NumFrames, num_sources * filter_length, num_channels)
    weights_unflat = weights.view(NumFrames, num_sources, filter_length, num_channels)

    # 4. Reconstruct projections for all sources in one shot (Batch Matrix Multiplication)
    T_s_all = src_unfold.permute(0, 2, 1, 3)  # (NumFrames, num_sources, window_length, filter_length)
    W_s_all = weights_unflat  # (NumFrames, num_sources, filter_length, num_channels)

    # Batched matrix product: (F, S, W, FL) @ (F, S, FL, C) -> (F, S, W, C)
    proj_all = torch.matmul(T_s_all, W_s_all)

    # Permute back to (NumFrames, window_length, num_channels, num_sources)
    projections_batched = proj_all.permute(0, 2, 3, 1) * analysis_window.view(1, window_length, 1, 1)
    projections_batched = projections_batched * synthesis_window.view(1, window_length, 1, 1)

    # 5. High-speed vectorized Overlap-Add via Scatter-Add
    projections_accumulation = torch.zeros(total_samples, num_channels, num_sources, dtype=true_sources.dtype,
                                           device=true_sources.device)
    window_gain_accumulation = torch.zeros(total_samples, 1, dtype=real_dtype, device=source_estimates.device)

    # Temporal indexing grids
    frame_indices = (torch.arange(window_length, device=source_estimates.device).view(1, -1) +
                     torch.arange(NumFrames, device=source_estimates.device).view(-1, 1) * hop_size)

    proj_flat = projections_batched.reshape(NumFrames * window_length, num_channels * num_sources)
    idx_flat = frame_indices.reshape(-1, 1).expand(-1, num_channels * num_sources)
    projections_accumulation.view(-1, num_channels * num_sources).scatter_add_(0, idx_flat, proj_flat)

    win_gain = (analysis_window * synthesis_window).view(1, window_length).expand(NumFrames, -1).reshape(-1, 1)
    idx_gain = frame_indices.reshape(-1, 1)
    window_gain_accumulation.scatter_add_(0, idx_gain, win_gain)

    # Differentiable safe division
    safe_gain = torch.where(window_gain_accumulation > 0, window_gain_accumulation, 1.0)
    projections_accumulation = projections_accumulation / safe_gain.unsqueeze(-1)

    # Return the cropped timeline (matches MATLAB/NumPy Ls timeline)
    return projections_accumulation[:-(window_length - 1), :, :]


def extract_target_spatial_distortion_interference_artifacts_torch(
        true_sources: torch.Tensor,
        source_estimates: torch.Tensor,
        filter_length: int,
        window_length: int,
        hop_size: int,
        use_two_stage_projection: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits signals into physical components in the subband domain."""
    total_samples, num_channels, num_sources = true_sources.shape
    num_estimates = source_estimates.shape[2] if source_estimates.dim() > 2 else 1
    if source_estimates.dim() == 2:
        source_estimates = source_estimates.unsqueeze(2)

    # Flatten channel and source dimensions for multi-channel solve
    sources_reshaped = true_sources.transpose(1, 2).reshape(total_samples, num_sources * num_channels)
    estimates_reshaped = source_estimates.transpose(1, 2).reshape(total_samples, num_estimates * num_channels)

    projections_all = perform_time_varying_least_squares_projection_torch(
        estimates_reshaped, sources_reshaped, filter_length, window_length, hop_size
    )

    # 100% Vectorized C++ aggregation (Replaces the Python For-Loop)
    proj_sliced = projections_all[:-filter_length + 1]
    T_dim, EC_dim = proj_sliced.shape[:2]

    # Reshape and sum across the contiguous channels natively
    projected_signals = proj_sliced.view(T_dim, EC_dim, num_sources, num_channels).sum(dim=-1)

    spatial_distortion = torch.zeros((total_samples, num_estimates * num_channels), dtype=source_estimates.dtype,
                                     device=source_estimates.device)
    if use_two_stage_projection:
        for est_idx in range(num_estimates):
            start = est_idx * num_channels
            end = (est_idx + 1) * num_channels
            spatial_projection = perform_time_varying_least_squares_projection_torch(
                estimates_reshaped[:, start:end],
                sources_reshaped[:, :num_channels],
                filter_length, window_length, hop_size
            )
            spatial_distortion[:, start:end] = torch.sum(spatial_projection[:-filter_length + 1, :, :], dim=2)

    true_ref = torch.zeros((total_samples, num_channels * num_estimates), dtype=true_sources.dtype,
                           device=true_sources.device)
    for est_idx in range(num_estimates):
        start = est_idx * num_channels
        end = (est_idx + 1) * num_channels
        true_ref[:, start:end] = sources_reshaped[:, :num_channels]

    if use_two_stage_projection:
        spatial_distortion = spatial_distortion - true_ref
    else:
        spatial_distortion = projected_signals[:, :, :num_estimates].transpose(1, 2).reshape(total_samples,
                                                                                             num_estimates * num_channels) - true_ref

    interference = torch.sum(projected_signals, dim=2) - spatial_distortion - true_ref
    artifacts = estimates_reshaped - true_ref - spatial_distortion - interference

    # NOTE: the NumPy backend reshapes these buffers with order='F' (estimate-major
    # column layout). torch only has C-order reshape, which coincides with F-order
    # here solely because num_estimates == 1 throughout the current pipeline. If
    # multi-estimate support is ever added, these reshapes must be reworked to
    # preserve the estimate-major layout, or the channels will be scrambled.
    true_ref_3d = true_ref.reshape(total_samples, num_channels, num_estimates)
    spatial_dist_3d = spatial_distortion.reshape(total_samples, num_channels, num_estimates)
    interference_3d = interference.reshape(total_samples, num_channels, num_estimates)
    artifacts_3d = artifacts.reshape(total_samples, num_channels, num_estimates)

    return true_ref_3d, spatial_dist_3d, interference_3d, artifacts_3d


def run_auditory_analysis_filterbank_torch(
        signal_waveform: torch.Tensor,
        fs: float,
        modulation_matrix: torch.Tensor | None = None,
        half_length_factor: int = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
) -> tuple[list[torch.Tensor], GammatoneAnalyzerTorch, torch.Tensor]:
    """Runs parallel analytical complex-valued filterbanks."""

    minimum_frequency = 20.0
    maximum_frequency = fs / 2.0
    base_frequency = 1000.0

    # 1. Apply Anti-Aliasing Polyphase Upsampling.
    # The PEASS reference always upsamples by 1.5x so the Gammatone filters near
    # the original Nyquist are well resolved. The original guard
    # `fs/2 < 1.5*(fs/2)` is a tautology (always true), so this is unconditional.
    original_fs = fs
    new_fs = int(round(1.5 * fs))
    signal_waveform = fast_resample_poly_torch(
        signal_waveform, new_fs, int(fs), axis=-1, half_length_factor=half_length_factor
    )
    fs = float(new_fs)

    analyzer = GammatoneAnalyzerTorch(fs, minimum_frequency, base_frequency, maximum_frequency, 1.0,
                                      signal_waveform.device, signal_waveform.dtype)
    analyzer.original_sampling_frequency_hz = original_fs

    subbands_output = analyzer.process(signal_waveform)
    num_bands = subbands_output.shape[-2]

    if modulation_matrix is None:
        modulation_matrix = _get_analysis_mod_matrix_torch(
            fs, subbands_output.shape[-1], tuple(analyzer.center_frequencies.tolist()), str(signal_waveform.device)
        )

    # -------------------------------------------------------------------------
    # OPTIMIZED: Group bands by unique decimation factors (decimations)
    # -------------------------------------------------------------------------
    decimated_bands = [None] * num_bands
    unique_factors = torch.unique(analyzer.decimations)

    for factor in unique_factors:
        factor_val = factor.item()
        band_indices = torch.where(analyzer.decimations == factor_val)[0]

        # Extract the group block and apply the analysis modulation in the same pass:
        # shape (Batch, NumGroupBands, Time). This replaces a separate full-block
        # `subbands_output = subbands_output * modulation_matrix` pass that used to sit
        # above this loop. `torch.unique` + `torch.where` partition the bands, so every
        # band lands in exactly one group and every element gets exactly one multiply by
        # exactly the row it would have got from the full-block pass. Bit-identical
        # forward and backward -- no reassociation, the same scalar multiply moved --
        # asserted with `torch.equal` on all four decomposed components and on
        # dL/dinput, mono and stereo. This is the analysis mirror of the fused synthesis
        # alignment matrix.
        #
        # It buys memory, not time: it drops one full-block complex128 temporary per
        # call. Measured on the 5 s / 16 kHz / 3-source MATLAB reference clip, where
        # every decimation group is a single band, so the block this fuses into is
        # 1/32 the size of the temporary it replaces:
        #
        #            batch  full block (dropped)  fused group block
        #   mono     3 + 1   184.32 + 61.44 MB      5.76 / 1.92 MB
        #   stereo   6 + 2   368.64 + 122.88 MB    11.52 / 3.84 MB
        #
        # (two calls per decomposition: the sources at batch S*C, the estimate at C.)
        # It is a MEASURED SPEED DUD: 2026-08-12, ten paired interleaved A/B
        # replications, mixed-sign, medians 0.999x / 1.002x. Do NOT re-prototype this
        # expecting a speedup; take it for the temporary alone.
        block = subbands_output[..., band_indices, :] * modulation_matrix[band_indices, :]

        # Resample the 3D block along the last axis in a single parallel call!
        resampled_block = fast_resample_poly_torch(block, 1, factor_val, axis=-1, half_length_factor=half_length_factor)

        # Unpack back into the list
        for idx, band_idx in enumerate(band_indices.tolist()):
            decimated_bands[band_idx] = resampled_block[..., idx, :]

    return decimated_bands, analyzer, modulation_matrix


def run_auditory_synthesis_filterbank_torch(
        subband_list: list,
        analyzer: GammatoneAnalyzerTorch,
        half_length_factor: int = DEFAULT_RESAMPLE_HALF_LENGTH_FACTOR
) -> torch.Tensor:
    """Reconstructs subbands back into a single fullband waveform."""
    num_bands = len(subband_list)
    is_batched = subband_list[0].dim() > 1
    B = subband_list[0].shape[0] if is_batched else 1

    max_len = max(subband_list[b].shape[-1] * int(analyzer.decimations[b]) for b in range(num_bands))

    if is_batched:
        processed = torch.zeros((B, num_bands, max_len), dtype=torch.complex128,
                                device=analyzer.center_frequencies.device)
    else:
        processed = torch.zeros((num_bands, max_len), dtype=torch.complex128, device=analyzer.center_frequencies.device)

    # -------------------------------------------------------------------------
    # OPTIMIZED: Group subbands by unique decimation factors (decimations)
    # -------------------------------------------------------------------------
    unique_factors = torch.unique(analyzer.decimations)

    for factor in unique_factors:
        factor_val = factor.item()
        band_indices = torch.where(analyzer.decimations == factor_val)[0]

        # Stack the subbands for this group: shape (..., GroupBands, T_decimated)
        block = torch.stack([subband_list[b] for b in band_indices.tolist()], dim=-2)

        # Upsample the 3D block in a single parallel call!
        upsampled_block = fast_resample_poly_torch(block, factor_val, 1, axis=-1, half_length_factor=half_length_factor)

        # Unpack back into the processed buffer
        for idx, band_idx in enumerate(band_indices.tolist()):
            upsampled = upsampled_block[..., idx, :]
            curr_len = upsampled.shape[-1]
            if curr_len >= max_len:
                if is_batched:
                    processed[:, band_idx, :] = upsampled[..., :max_len]
                else:
                    processed[band_idx, :] = upsampled[:max_len]
            else:
                if is_batched:
                    processed[:, band_idx, :curr_len] = upsampled
                else:
                    processed[band_idx, :curr_len] = upsampled

    # Re-modulate back to center frequencies and phase-align, in one cached matrix that
    # the synthesizer applies band by band. Multiplying `processed` by the modulation
    # here would cost a full extra pass over a 62-123 MB block for nothing.
    desired_delay_seconds = 1000.0 / analyzer.fs
    alignment = _get_synthesis_alignment_matrix_torch(
        analyzer.fs, max_len, tuple(analyzer.center_frequencies.tolist()),
        str(analyzer.center_frequencies.device), desired_delay_seconds,
        tuple(analyzer.norms.tolist()), tuple(analyzer.coefs.tolist())
    )

    synth = GammatoneSynthesizerTorch(analyzer, desired_delay_seconds)
    reconstructed = synth.process(processed, alignment=alignment)

    # 1. Downsample back to the original frequency to prevent duration expansion
    original_fs = analyzer.original_sampling_frequency_hz
    reconstructed = fast_resample_poly_torch(
        reconstructed, int(original_fs), int(analyzer.fs), axis=-1, half_length_factor=half_length_factor
    )
    delay_samples = int(round(desired_delay_seconds * original_fs))

    return reconstructed[..., delay_samples:]


def matlab_shade_length(shade_milliseconds: float, sampling_frequency_hz: float) -> int:
    """
    Length in samples of a PEASS shade window, matching MATLAB
    `extractDistortionComponents.m` (v2.0.1, lines ~155-165).

    MATLAB builds a periodic Hann of length ``N = 2*round(ms/1000*fs + 1)`` and keeps
    ``w(2:end/2)``, i.e. ``N/2 - 1 = round(ms/1000*fs)`` samples. MATLAB's ``round``
    breaks ties away from zero, whereas Python rounds half to even, so the tie rule is
    spelled out here (it matters at e.g. fs=44100 with shadeMs=5 or 25, where
    ms/1000*fs lands exactly on .5).
    """
    return int(math.floor(shade_milliseconds / 1000.0 * sampling_frequency_hz + 0.5))


def matlab_shade_window_torch(fade_samples: int, device, dtype) -> torch.Tensor:
    """
    MATLAB-exact PEASS shade-in window of ``fade_samples`` (= R) samples.

    Reference: `extractDistortionComponents.m` (v2.0.1)::

        wShadeIn = hann(2*round(options.shadeInMs/1000*fs+1),'periodic');
        wShadeIn = wShadeIn(2:end/2);

    With ``hann(N,'periodic')[k] = 0.5*(1 - cos(2*pi*(k-1)/N))`` for 1-based k=1..N and
    ``N = 2*(R+1)``, the slice keeps k = 2..N/2, which collapses to

        w[n] = 0.5 * (1 - cos(pi*(n+1)/(R+1)))      for n = 0..R-1

    This is the strict INTERIOR of a Hann rise: MATLAB drops both the leading zero
    (k=1) and the unity midpoint (k=N/2+1), so the window never reaches exactly 0 or
    exactly 1. Do NOT "simplify" it back to a full 0->1 ramp (``pi*n/(R-1)``) - that
    deviates from the reference by ~1.3e-3 at 44.1 kHz and ~7e-3 at 8 kHz, and it is
    undefined for R=1 whereas this form is well defined for every R >= 1.

    The shade-out window is this window reversed (MATLAB applies ``flipud``).
    """
    time_steps = torch.arange(1, fade_samples + 1, device=device, dtype=dtype)
    return 0.5 * (1.0 - torch.cos(math.pi * time_steps / (fade_samples + 1)))


def apply_window_shading_torch(sig: torch.Tensor, fs: float, shade_in: float, shade_out: float) -> torch.Tensor:
    shaded = sig.clone()
    num_samples = shaded.shape[0]

    fade_in_samples = matlab_shade_length(shade_in, fs) if shade_in > 0 else 0
    fade_out_samples = matlab_shade_length(shade_out, fs) if shade_out > 0 else 0

    if fade_in_samples + fade_out_samples > num_samples:
        raise ValueError(
            f"Combined shading length ({fade_in_samples + fade_out_samples} samples) "
            f"exceeds the signal length ({num_samples} samples)."
        )

    if fade_in_samples > 0:
        win = matlab_shade_window_torch(fade_in_samples, sig.device, sig.dtype)
        shaded[:fade_in_samples, :] *= win.unsqueeze(1)

    if fade_out_samples > 0:
        # MATLAB obtains the fade-out via flipud() of the same window; deriving it
        # by reversal (rather than a separate cosine) keeps the two exactly consistent.
        win = matlab_shade_window_torch(fade_out_samples, sig.device, sig.dtype).flip(0)
        shaded[-fade_out_samples:, :] *= win.unsqueeze(1)

    return shaded


def decompose_distortion_components(
        source_files: list[str | pathlib.Path | torch.Tensor],
        estimate_file: str | pathlib.Path | torch.Tensor,
        configuration=None,
        sampling_frequency_hz: float | None = None
) -> DecompositionResult:
    """Natively decomposes estimated audio waveforms on PyTorch GPU."""
    if configuration is None:
        from ..config import DecompositionConfiguration
        configuration = DecompositionConfiguration()

    if not source_files:
        raise ValueError("source_files list cannot be empty.")

    is_file_mode = isinstance(estimate_file, str | pathlib.Path)

    if is_file_mode:
        # File-based loading on PyTorch backend
        est_numpy, fs = sf.read(estimate_file)
        sampling_frequency_hz = float(fs)

        # Push to PyTorch (checks if CUDA/MPS is available to choose default hardware accelerator)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        estimate_tensor = torch.tensor(est_numpy, device=device, dtype=torch.float64)

        sources_tensors = []
        for src_path in source_files:
            src_numpy, src_fs = sf.read(src_path)
            if src_fs != fs:
                raise ValueError("Sampling rates of all files must match.")
            sources_tensors.append(torch.tensor(src_numpy, device=device, dtype=torch.float64))
    else:
        estimate_tensor = estimate_file
        sources_tensors = source_files

    # ADDED FIX: Raise explicit error for missing sampling rate in in-memory mode
    if not is_file_mode and sampling_frequency_hz is None:
        raise ValueError("In-memory mode requires explicit sampling rate 'sampling_frequency_hz'.")

    # 1. Validate spatial layouts
    estimate_audio = validate_and_normalize_audio_torch(estimate_tensor, sampling_frequency_hz, "estimate_file")
    sources_audio = [
        validate_and_normalize_audio_torch(s, sampling_frequency_hz, f"source_files[{i}]")
        for i, s in enumerate(sources_tensors)
    ]

    # ADDED FIX: Enforce matching shapes early to prevent internal reshape crashes
    for s in sources_audio:
        if s.shape != estimate_audio.shape:
            raise ValueError("All source signals must be of matching dimensions.")

    N_samples = estimate_audio.shape[0]
    C = estimate_audio.shape[1]
    S = len(sources_audio)

    # 2. Window shading
    shaded_sources = [apply_window_shading_torch(s, sampling_frequency_hz, configuration.shade_in_milliseconds,
                                                 configuration.shade_out_milliseconds) for s in sources_audio]
    shaded_estimate = apply_window_shading_torch(estimate_audio, sampling_frequency_hz,
                                                 configuration.shade_in_milliseconds,
                                                 configuration.shade_out_milliseconds)

    resample_factor = configuration.resample_filter_half_length_factor

    # Batched Gammatone Analysis
    sources_stacked = torch.stack(shaded_sources, dim=0)  # (S, T, C)
    sources_flat = sources_stacked.transpose(1, 2).reshape(S * C, N_samples)

    # Sources and estimate stay two separate passes over the filterbank. Merging them
    # into one wider batch was tried (2026-08-10) on the grounds that the batch is
    # what the FFT parallelises over: measured within noise on mono and slightly
    # negative on stereo, because it also raises the peak frequency-domain footprint.
    # The batching argument had force when resampling was an FFT starved of batch;
    # the polyphase GEMM path in `utils.py` removed that.
    subbands_sources_flat, analyzer, mod_matrix = run_auditory_analysis_filterbank_torch(
        sources_flat, sampling_frequency_hz, half_length_factor=resample_factor
    )

    estimate_flat = shaded_estimate.transpose(0, 1)  # (C, T)
    subband_estimate_flat, _, _ = run_auditory_analysis_filterbank_torch(
        estimate_flat, sampling_frequency_hz, mod_matrix, half_length_factor=resample_factor
    )

    num_bands = len(subbands_sources_flat)

    # 4. Form composite band tensors and solve
    ref_frequency = 1000.0
    f_ref_idx = torch.argmin(torch.abs(analyzer.center_frequencies - ref_frequency)).item()
    ref_bw = analyzer.bandwidths[f_ref_idx]
    decimated_fs = analyzer.fs / analyzer.decimations

    ref_filter_len = min(configuration.filter_length_seconds, configuration.frame_length_seconds / C / S / 3.0)
    bw_ratios = ref_bw / analyzer.bandwidths * decimated_fs

    filter_lengths = torch.clamp(2 * torch.round((ref_filter_len * bw_ratios - 1) / 2.0) + 1, min=3).to(torch.int32)
    window_lengths = torch.clamp(torch.round(configuration.frame_length_seconds * bw_ratios), min=3).to(torch.int32)
    hop_sizes = torch.clamp(torch.round((configuration.frame_length_seconds / 4.0) * bw_ratios), min=1).to(torch.int32)

    decomp_true = []
    decomp_target = []
    decomp_interf = []
    decomp_artifacts = []

    for b in range(num_bands):
        b_src = subbands_sources_flat[b].view(S, C, -1)
        true_b_stacked = b_src.permute(2, 1, 0)

        b_est = subband_estimate_flat[b].view(C, -1)
        est_b_stacked = b_est.transpose(0, 1).unsqueeze(2)

        t_b, td_b, int_b, art_b = extract_target_spatial_distortion_interference_artifacts_torch(
            true_b_stacked, est_b_stacked, filter_lengths[b].item(), window_lengths[b].item(), hop_sizes[b].item(),
            use_two_stage_projection=configuration.use_two_stage_projection
        )
        decomp_true.append(t_b)
        decomp_target.append(td_b)
        decomp_interf.append(int_b)
        decomp_artifacts.append(art_b)

    # Batched Gammatone Synthesis, one pass per distortion component. Concatenating
    # the four into a single 4*C-row batch was measured alongside the analysis merge
    # above and behaved the same way: no reliable gain, higher peak memory.
    components = (decomp_true, decomp_target, decomp_interf, decomp_artifacts)
    subband_true_batched, subband_target_batched, subband_interf_batched, subband_artif_batched = (
        [component[b].squeeze(-1).transpose(0, 1) for b in range(num_bands)]
        for component in components
    )

    def clip_pad(val, target):
        L = val.shape[0]
        if L >= target: return val[:target]
        # PADDING THE FIRST DIMENSION (rows, time) OF TENSOR WITH (0, 0, 0, target - L)
        return F.pad(val, (0, 0, 0, target - L))

    # Synthesis outputs (C, T) -> transpose to (T, C)
    synth_t = run_auditory_synthesis_filterbank_torch(
        subband_true_batched, analyzer, half_length_factor=resample_factor).transpose(0, 1)
    synth_td = run_auditory_synthesis_filterbank_torch(
        subband_target_batched, analyzer, half_length_factor=resample_factor).transpose(0, 1)
    synth_i = run_auditory_synthesis_filterbank_torch(
        subband_interf_batched, analyzer, half_length_factor=resample_factor).transpose(0, 1)
    synth_a = run_auditory_synthesis_filterbank_torch(
        subband_artif_batched, analyzer, half_length_factor=resample_factor).transpose(0, 1)

    waveforms = DecomposedWaveforms(
        true_target=clip_pad(synth_t, N_samples),
        target_distortion=clip_pad(synth_td, N_samples),
        interference=clip_pad(synth_i, N_samples),
        artifacts=clip_pad(synth_a, N_samples)
    )

    # 6. Save WAV files to disk only in file mode (matches the NumPy backend).
    out_paths = None
    if is_file_mode:
        dest_dir = pathlib.Path(configuration.destination_directory)
        dest_dir.mkdir(parents=True, exist_ok=True)

        stem = pathlib.Path(estimate_file).stem
        out_paths = DecomposedFilePaths(
            true_target=str(dest_dir / f"{stem}_true.wav"),
            target_distortion=str(dest_dir / f"{stem}_eTarget.wav"),
            interference=str(dest_dir / f"{stem}_eInterf.wav"),
            artifacts=str(dest_dir / f"{stem}_eArtif.wav")
        )
        sf.write(out_paths.true_target, waveforms.true_target.detach().cpu().numpy(), int(sampling_frequency_hz))
        sf.write(out_paths.target_distortion, waveforms.target_distortion.detach().cpu().numpy(),
                 int(sampling_frequency_hz))
        sf.write(out_paths.interference, waveforms.interference.detach().cpu().numpy(), int(sampling_frequency_hz))
        sf.write(out_paths.artifacts, waveforms.artifacts.detach().cpu().numpy(), int(sampling_frequency_hz))

    return DecompositionResult(waveforms=waveforms, file_paths=out_paths)
