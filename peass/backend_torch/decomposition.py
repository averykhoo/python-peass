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
def _get_synthesis_mod_matrix_torch(fs: float, max_len: int, cfs_tuple: tuple, device_str: str):
    device = torch.device(device_str)
    cfs = torch.tensor(cfs_tuple, device=device, dtype=torch.float64)
    time_steps = torch.arange(max_len, device=device, dtype=torch.float64)
    return torch.exp(2j * math.pi / fs * cfs.unsqueeze(1) * time_steps)


def get_real_dtype(dtype: torch.dtype) -> torch.dtype:
    """Helper returning the corresponding real-valued dtype counterpart of any dtype."""
    if dtype in (torch.complex128, torch.complex64):
        return torch.float64 if dtype == torch.complex128 else torch.float32
    return dtype


def validate_and_normalize_audio_torch(data: torch.Tensor, name: str = "audio_data") -> torch.Tensor:
    """Enforces standard 2D (samples, channels) formatting for PyTorch tensors."""
    if data.dim() == 1:
        data = data.unsqueeze(1)

    num_samples, num_channels = data.shape
    if num_channels > 32:
        raise ValueError(
            f"Layout violation for '{name}'. Expected (samples, channels) with channels <= 32. "
            f"Found shape {data.shape}. Please transpose."
        )
    return data


def perform_least_squares_projection_torch(
        source_estimates: torch.Tensor,
        true_sources: torch.Tensor,
        filter_half_length: int,
        analysis_window: torch.Tensor
) -> torch.Tensor:
    """Computes weighted least-squares projections natively on GPU."""
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

    # Diagonal regularization for positive-definiteness
    Gram.diagonal().add_(10.0 ** -15)

    try:
        projection_weights = torch.linalg.solve(Gram, RHS)
    except torch.linalg.LinAlgError:
        weighted_toeplitz = toeplitz_matrix * analysis_window.unsqueeze(1)
        weighted_estimates = source_estimates * analysis_window.unsqueeze(1)
        projection_weights = torch.linalg.pinv(weighted_toeplitz) @ weighted_estimates

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

    # Diagonal regularization
    diag_idx = torch.arange(Gram.shape[-1], device=Gram.device)
    Gram[:, diag_idx, diag_idx] += 10.0 ** -15

    try:
        weights = torch.linalg.solve(Gram, RHS)
    except torch.linalg.LinAlgError:
        weights = torch.linalg.pinv(Gram) @ RHS

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

    # 6. Apply final gain normalization
    valid_indices = (window_gain_accumulation[:, 0] != 0)
    for s_idx in range(num_sources):
        projections_accumulation[valid_indices, :, s_idx] /= window_gain_accumulation[valid_indices, :]

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

    true_ref_3d = true_ref.reshape(total_samples, num_channels, num_estimates)
    spatial_dist_3d = spatial_distortion.reshape(total_samples, num_channels, num_estimates)
    interference_3d = interference.reshape(total_samples, num_channels, num_estimates)
    artifacts_3d = artifacts.reshape(total_samples, num_channels, num_estimates)

    return true_ref_3d, spatial_dist_3d, interference_3d, artifacts_3d


def run_auditory_analysis_filterbank_torch(
        signal_waveform: torch.Tensor,
        fs: float,
        modulation_matrix: torch.Tensor | None = None
) -> tuple[list[torch.Tensor], GammatoneAnalyzerTorch, torch.Tensor]:
    """Runs parallel analytical complex-valued filterbanks."""

    minimum_frequency = 20.0
    maximum_frequency = fs / 2.0
    base_frequency = 1000.0

    # 1. Apply Anti-Aliasing Polyphase Upsampling
    original_fs = fs
    if fs / 2.0 < 1.5 * maximum_frequency:
        new_fs = int(round(1.5 * fs))
        signal_waveform = fast_resample_poly_torch(signal_waveform, new_fs, int(fs), axis=-1)
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

    subbands_output = subbands_output * modulation_matrix

    # -------------------------------------------------------------------------
    # OPTIMIZED: Group bands by unique decimation factors (decimations)
    # -------------------------------------------------------------------------
    decimated_bands = [None] * num_bands
    unique_factors = torch.unique(analyzer.decimations)

    for factor in unique_factors:
        factor_val = factor.item()
        band_indices = torch.where(analyzer.decimations == factor_val)[0]

        # Extract the entire group block: shape (Batch, NumGroupBands, Time)
        block = subbands_output[..., band_indices, :]

        # Resample the 3D block along the last axis in a single parallel call!
        resampled_block = fast_resample_poly_torch(block, 1, factor_val, axis=-1)

        # Unpack back into the list
        for idx, band_idx in enumerate(band_indices.tolist()):
            decimated_bands[band_idx] = resampled_block[..., idx, :]

    return decimated_bands, analyzer, modulation_matrix


def run_auditory_synthesis_filterbank_torch(
        subband_list: list,
        analyzer: GammatoneAnalyzerTorch
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
        upsampled_block = fast_resample_poly_torch(block, factor_val, 1, axis=-1)

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

    # Re-modulate back to center frequencies prior to synthesis
    mod_matrix = _get_synthesis_mod_matrix_torch(
        analyzer.fs, max_len, tuple(analyzer.center_frequencies.tolist()), str(analyzer.center_frequencies.device)
    )
    processed = processed * mod_matrix

    desired_delay_seconds = 1000.0 / analyzer.fs
    synth = GammatoneSynthesizerTorch(analyzer, desired_delay_seconds)
    reconstructed = synth.process(processed)

    # 1. Downsample back to the original frequency to prevent duration expansion
    original_fs = analyzer.original_sampling_frequency_hz
    reconstructed = fast_resample_poly_torch(reconstructed, int(original_fs), int(analyzer.fs), axis=-1)
    delay_samples = int(round(desired_delay_seconds * original_fs))

    return reconstructed[..., delay_samples:]


def apply_window_shading_torch(sig: torch.Tensor, fs: float, shade_in: float, shade_out: float) -> torch.Tensor:
    shaded = sig.clone()
    num_samples = shaded.shape[0]

    fade_in_samples = int(round(shade_in / 1000.0 * fs)) if shade_in > 0 else 0
    fade_out_samples = int(round(shade_out / 1000.0 * fs)) if shade_out > 0 else 0

    if fade_in_samples > 1:
        t = torch.arange(fade_in_samples, device=sig.device, dtype=sig.dtype)
        win = 0.5 - 0.5 * torch.cos(math.pi * t / (fade_in_samples - 1))
        shaded[:fade_in_samples, :] *= win.unsqueeze(1)

    if fade_out_samples > 1:
        t = torch.arange(fade_out_samples, device=sig.device, dtype=sig.dtype)
        win = 0.5 + 0.5 * torch.cos(math.pi * t / (fade_out_samples - 1))
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
            src_numpy, _ = sf.read(src_path)
            sources_tensors.append(torch.tensor(src_numpy, device=device, dtype=torch.float64))
    else:
        estimate_tensor = estimate_file
        sources_tensors = source_files

    # 1. Validate spatial layouts
    estimate_audio = validate_and_normalize_audio_torch(estimate_tensor, "estimate_file")
    sources_audio = [validate_and_normalize_audio_torch(s, f"source_files[{i}]") for i, s in enumerate(sources_tensors)]

    N_samples = estimate_audio.shape[0]
    C = estimate_audio.shape[1]
    S = len(sources_audio)

    # 2. Window shading
    shaded_sources = [apply_window_shading_torch(s, sampling_frequency_hz, configuration.shade_in_milliseconds,
                                                 configuration.shade_out_milliseconds) for s in sources_audio]
    shaded_estimate = apply_window_shading_torch(estimate_audio, sampling_frequency_hz,
                                                 configuration.shade_in_milliseconds,
                                                 configuration.shade_out_milliseconds)

    # Batched Gammatone Analysis
    sources_stacked = torch.stack(shaded_sources, dim=0)  # (S, T, C)
    sources_flat = sources_stacked.transpose(1, 2).reshape(S * C, N_samples)

    subbands_sources_flat, analyzer, mod_matrix = run_auditory_analysis_filterbank_torch(sources_flat,
                                                                                         sampling_frequency_hz)

    estimate_flat = shaded_estimate.transpose(0, 1)  # (C, T)
    subband_estimate_flat, _, _ = run_auditory_analysis_filterbank_torch(estimate_flat, sampling_frequency_hz,
                                                                         mod_matrix)

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

    # Batched Gammatone Synthesis
    subband_true_batched = [decomp_true[b].squeeze(-1).transpose(0, 1) for b in range(num_bands)]
    subband_target_batched = [decomp_target[b].squeeze(-1).transpose(0, 1) for b in range(num_bands)]
    subband_interf_batched = [decomp_interf[b].squeeze(-1).transpose(0, 1) for b in range(num_bands)]
    subband_artif_batched = [decomp_artifacts[b].squeeze(-1).transpose(0, 1) for b in range(num_bands)]

    def clip_pad(val, target):
        L = val.shape[0]
        if L >= target: return val[:target]
        # PADDING THE FIRST DIMENSION (rows, time) OF TENSOR WITH (0, 0, 0, target - L)
        return F.pad(val, (0, 0, 0, target - L))

    # Synthesis outputs (C, T) -> transpose to (T, C)
    synth_t = run_auditory_synthesis_filterbank_torch(subband_true_batched, analyzer).transpose(0, 1)
    synth_td = run_auditory_synthesis_filterbank_torch(subband_target_batched, analyzer).transpose(0, 1)
    synth_i = run_auditory_synthesis_filterbank_torch(subband_interf_batched, analyzer).transpose(0, 1)
    synth_a = run_auditory_synthesis_filterbank_torch(subband_artif_batched, analyzer).transpose(0, 1)

    waveforms = DecomposedWaveforms(
        true_target=clip_pad(synth_t, N_samples),
        target_distortion=clip_pad(synth_td, N_samples),
        interference=clip_pad(synth_i, N_samples),
        artifacts=clip_pad(synth_a, N_samples)
    )

    # 6. Save WAV files natively to disk if requested in config or running in file mode
    out_paths = None
    if is_file_mode or configuration.destination_directory != "./":
        dest_dir = pathlib.Path(configuration.destination_directory)
        dest_dir.mkdir(parents=True, exist_ok=True)

        stem = pathlib.Path(estimate_file).stem if is_file_mode else "decomposed"
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
