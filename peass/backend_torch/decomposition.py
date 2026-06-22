"""
PEASS PyTorch Least-Squares Decomposer
File path: peass/backend_torch/decomposition.py
"""
import math
import pathlib
import torch
import torch.nn.functional as F
import soundfile as sf

from ..config import DecompositionResult, DecomposedWaveforms, DecomposedFilePaths
from .gammatone import GammatoneAnalyzerTorch, GammatoneSynthesizerTorch
from .utils import fast_resample_poly_torch


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
    source_energy = torch.sum(true_sources.real ** 2 + true_sources.imag ** 2) if true_sources.is_complex() else torch.sum(true_sources ** 2)
    if source_energy < 1e-13:
        return torch.zeros((num_samples, source_estimates.shape[1], num_sources), dtype=source_estimates.dtype, device=source_estimates.device)

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

    projections = torch.zeros((num_samples, source_estimates.shape[1], num_sources), dtype=source_estimates.dtype, device=source_estimates.device)
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
    """Time-varying subband least-squares solver."""
    filter_half_length = (filter_length - 1) // 2
    pad_length = filter_length - 1 + window_length - 1

    true_sources = F.pad(true_sources, (0, 0, 0, pad_length))
    source_estimates = F.pad(source_estimates, (0, 0, 0, pad_length))

    total_samples, num_sources = true_sources.shape
    num_channels = source_estimates.shape[1]

    # Dynamically resolve real-valued dtype to prevent arange complex exception
    real_dtype = get_real_dtype(source_estimates.dtype)

    # Match SciPy's sym=False with periodic=True
    hann_window = torch.hann_window(window_length, periodic=True, device=source_estimates.device, dtype=real_dtype)
    analysis_window = torch.sqrt(torch.flip(hann_window, dims=[0]))
    synthesis_window = torch.sqrt(torch.flip(hann_window, dims=[0]))

    synthesis_weights = synthesis_window.view(window_length, 1, 1).repeat(1, num_channels, num_sources)

    window_begin = 0
    window_end = window_begin + window_length

    projections_accumulation = torch.zeros((total_samples, num_channels, num_sources), dtype=true_sources.dtype, device=true_sources.device)
    window_gain_accumulation = torch.zeros((total_samples, 1), dtype=real_dtype, device=source_estimates.device)

    while window_end - window_length / 2.0 <= projections_accumulation.shape[0] - window_length + 1:
        frame_estimates = source_estimates[window_begin:window_end, :]

        source_window_start = window_begin - filter_half_length
        source_window_end = window_end + filter_half_length
        pad_left = max(0, -source_window_start)
        pad_right = max(0, source_window_end - true_sources.shape[0])
        slice_start = max(0, source_window_start)
        slice_end = min(true_sources.shape[0], source_window_end)

        frame_sources_slice = true_sources[slice_start:slice_end, :]
        frame_sources = torch.cat([
            torch.zeros((pad_left, num_sources), dtype=true_sources.dtype, device=true_sources.device),
            frame_sources_slice,
            torch.zeros((pad_right, num_sources), dtype=true_sources.dtype, device=true_sources.device)
        ], dim=0)

        frame_projections = perform_least_squares_projection_torch(
            frame_estimates, frame_sources, filter_half_length, analysis_window
        )

        projections_accumulation[window_begin:window_end, :, :] += (
            frame_projections[:window_length, :, :] * synthesis_weights
        )
        window_gain_accumulation[window_begin:window_end, 0] += synthesis_window * analysis_window

        window_begin += hop_size
        window_end += hop_size

    valid_indices = (window_gain_accumulation[:, 0] != 0)
    for s_idx in range(num_sources):
        projections_accumulation[valid_indices, :, s_idx] /= window_gain_accumulation[valid_indices, :]

    return projections_accumulation[:-(window_length - 1), :, :]

# Optimize compilation path: Bypass JIT compilation latency on standard CPU runs
# but compile on GPU or when explicitly requested.
_SHOULD_COMPILE = torch.cuda.is_available() and os.environ.get("PEASS_NO_COMPILE") != "1"

if _SHOULD_COMPILE:
    try:
        _compiled_projection_solver = torch.compile(
            perform_time_varying_least_squares_projection_torch,
            mode="reduce-overhead"
        )
    except Exception:
        _compiled_projection_solver = perform_time_varying_least_squares_projection_torch
else:
    _compiled_projection_solver = perform_time_varying_least_squares_projection_torch

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

    projections_all = _compiled_projection_solver(
        estimates_reshaped, sources_reshaped, filter_length, window_length, hop_size
    )

    projected_signals = torch.zeros((total_samples, num_channels * num_estimates, num_sources), dtype=true_sources.dtype, device=true_sources.device)
    for s_idx in range(num_sources):
        start = s_idx * num_channels
        end = (s_idx + 1) * num_channels
        projected_signals[:, :, s_idx] = torch.sum(projections_all[:total_samples, :, start:end], dim=2)

    spatial_distortion = torch.zeros((total_samples, num_estimates * num_channels), dtype=source_estimates.dtype, device=source_estimates.device)
    if use_two_stage_projection:
        for est_idx in range(num_estimates):
            start = est_idx * num_channels
            end = (est_idx + 1) * num_channels
            spatial_projection = perform_time_varying_least_squares_projection_torch(
                estimates_reshaped[:, start:end],
                sources_reshaped[:, :num_channels],
                filter_length, window_length, hop_size
            )
            spatial_distortion[:, start:end] = torch.sum(spatial_projection[:total_samples, :, :], dim=2)

    true_ref = torch.zeros((total_samples, num_channels * num_estimates), dtype=true_sources.dtype, device=true_sources.device)
    for est_idx in range(num_estimates):
        start = est_idx * num_channels
        end = (est_idx + 1) * num_channels
        true_ref[:, start:end] = sources_reshaped[:, :num_channels]

    if use_two_stage_projection:
        spatial_distortion = spatial_distortion - true_ref
    else:
        spatial_distortion = projected_signals[:, :, :num_estimates].transpose(1, 2).reshape(total_samples, num_estimates * num_channels) - true_ref

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
    filters_per_erb = 1.0

    # 1. Apply Anti-Aliasing Polyphase Upsampling
    original_fs = fs
    if fs / 2.0 < 1.5 * maximum_frequency:
        new_fs = int(round(1.5 * fs))
        signal_waveform = fast_resample_poly_torch(signal_waveform, new_fs, int(fs))
        fs = float(new_fs)

    analyzer = GammatoneAnalyzerTorch(
        fs, minimum_frequency, base_frequency, maximum_frequency, filters_per_erb,
        signal_waveform.device, signal_waveform.dtype
    )

    # Save the original sample rate onto the analyzer object to match NumPy behavior
    analyzer.original_sampling_frequency_hz = original_fs

    subbands_output = analyzer.process(signal_waveform)
    num_bands = subbands_output.shape[0]

    if modulation_matrix is None:
        time_steps = torch.arange(subbands_output.shape[1], device=signal_waveform.device, dtype=signal_waveform.dtype)
        center_frequencies = analyzer.center_frequencies.unsqueeze(1)
        # Shift down to baseband (demodulate)
        modulation_matrix = torch.exp(-2j * math.pi / fs * center_frequencies * time_steps)

    subbands_output = subbands_output * modulation_matrix

    decimated_bands = []
    for band_idx in range(num_bands):
        decimated = fast_resample_poly_torch(subbands_output[band_idx], 1, int(analyzer.decimations[band_idx]))
        decimated_bands.append(decimated)

    return decimated_bands, analyzer, modulation_matrix


def run_auditory_synthesis_filterbank_torch(
    subband_list: list,
    analyzer: GammatoneAnalyzerTorch
) -> torch.Tensor:
    """Reconstructs subbands back into a single fullband waveform."""
    num_bands = len(subband_list)
    max_len = max(len(subband_list[b]) * int(analyzer.decimations[b]) for b in range(num_bands))

    processed = torch.zeros((num_bands, max_len), dtype=torch.complex128, device=analyzer.center_frequencies.device)
    for b in range(num_bands):
        factor = int(analyzer.decimations[b])
        upsampled = fast_resample_poly_torch(subband_list[b], factor, 1)
        curr_len = len(upsampled)
        if curr_len >= max_len:
            processed[b] = upsampled[:max_len]
        else:
            processed[b, :curr_len] = upsampled

    # Re-modulate back to center frequencies prior to synthesis
    time_steps = torch.arange(max_len, device=analyzer.center_frequencies.device, dtype=analyzer.center_frequencies.dtype)
    mod_matrix = torch.exp(2j * math.pi / analyzer.fs * analyzer.center_frequencies.unsqueeze(1) * time_steps)
    processed = processed * mod_matrix

    # -------------------------------------------------------------------------
    # FIXED: Replaced hardcoded 0.004 (4ms) delay with 1000.0 / analyzer.fs (41.67ms at 24kHz)
    # -------------------------------------------------------------------------
    desired_delay_seconds = 1000.0 / analyzer.fs
    synth = GammatoneSynthesizerTorch(analyzer, desired_delay_seconds)
    reconstructed = synth.process(processed)

    # 1. Downsample back to the original frequency to prevent duration expansion
    original_fs = analyzer.original_sampling_frequency_hz
    reconstructed = fast_resample_poly_torch(reconstructed, int(original_fs), int(analyzer.fs))

    # 2. Account for synthesizer delay offsets AT the original sampling frequency
    delay_samples = int(round(desired_delay_seconds * original_fs))
    return reconstructed[delay_samples:]


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
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
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
    shaded_sources = [apply_window_shading_torch(s, sampling_frequency_hz, configuration.shade_in_milliseconds, configuration.shade_out_milliseconds) for s in sources_audio]
    shaded_estimate = apply_window_shading_torch(estimate_audio, sampling_frequency_hz, configuration.shade_in_milliseconds, configuration.shade_out_milliseconds)

    # 3. Analyze subbands natively
    subband_sources = [[None for _ in range(C)] for _ in range(S)]
    mod_matrix = None
    analyzer = None

    for s_idx in range(S):
        for c_idx in range(C):
            subband_sources[s_idx][c_idx], analyzer, mod_matrix = run_auditory_analysis_filterbank_torch(
                shaded_sources[s_idx][:, c_idx], sampling_frequency_hz, mod_matrix
            )

    subband_estimate = [None for _ in range(C)]
    for c_idx in range(C):
        subband_estimate[c_idx], _, _ = run_auditory_analysis_filterbank_torch(
            shaded_estimate[:, c_idx], sampling_frequency_hz, mod_matrix
        )

    num_bands = len(subband_sources[0][0])

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
        # Gather signals across sources and channels for this band
        # true_b: (T_dec, C, S), est_b: (T_dec, C, 1)
        true_b_stacked = torch.stack([torch.stack([subband_sources[s][c][b] for c in range(C)], dim=1) for s in range(S)], dim=2)
        est_b_stacked = torch.stack([subband_estimate[c][b] for c in range(C)], dim=1).unsqueeze(2)

        t_b, td_b, int_b, art_b = extract_target_spatial_distortion_interference_artifacts_torch(
            true_b_stacked, est_b_stacked, filter_lengths[b].item(), window_lengths[b].item(), hop_sizes[b].item(),
            use_two_stage_projection=configuration.use_two_stage_projection
        )
        decomp_true.append(t_b)
        decomp_target.append(td_b)
        decomp_interf.append(int_b)
        decomp_artifacts.append(art_b)

    # 5. Format and synthesize back to fullband
    ref_subband_true = [[decomp_true[b][:, c, 0] for b in range(num_bands)] for c in range(C)]
    ref_subband_target = [[decomp_target[b][:, c, 0] for b in range(num_bands)] for c in range(C)]
    ref_subband_interf = [[decomp_interf[b][:, c, 0] for b in range(num_bands)] for c in range(C)]
    ref_subband_artif = [[decomp_artifacts[b][:, c, 0] for b in range(num_bands)] for c in range(C)]

    synth_t = torch.zeros((N_samples, C), dtype=estimate_audio.dtype, device=estimate_audio.device)
    synth_td = torch.zeros((N_samples, C), dtype=estimate_audio.dtype, device=estimate_audio.device)
    synth_i = torch.zeros((N_samples, C), dtype=estimate_audio.dtype, device=estimate_audio.device)
    synth_a = torch.zeros((N_samples, C), dtype=estimate_audio.dtype, device=estimate_audio.device)

    def clip_pad(val, target):
        L = val.shape[0]
        if L >= target:
            return val[:target]
        return F.pad(val, (0, target - L))

    for c in range(C):
        st = run_auditory_synthesis_filterbank_torch(ref_subband_true[c], analyzer)
        std = run_auditory_synthesis_filterbank_torch(ref_subband_target[c], analyzer)
        si = run_auditory_synthesis_filterbank_torch(ref_subband_interf[c], analyzer)
        sa = run_auditory_synthesis_filterbank_torch(ref_subband_artif[c], analyzer)

        synth_t[:, c] = clip_pad(st, N_samples)
        synth_td[:, c] = clip_pad(std, N_samples)
        synth_i[:, c] = clip_pad(si, N_samples)
        synth_a[:, c] = clip_pad(sa, N_samples)

    waveforms = DecomposedWaveforms(
        true_target=synth_t,
        target_distortion=synth_td,
        interference=synth_i,
        artifacts=synth_a
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

        # Cast back to CPU for standard soundfile saving
        sf.write(out_paths.true_target, synth_t.detach().cpu().numpy(), int(sampling_frequency_hz))
        sf.write(out_paths.target_distortion, synth_td.detach().cpu().numpy(), int(sampling_frequency_hz))
        sf.write(out_paths.interference, synth_i.detach().cpu().numpy(), int(sampling_frequency_hz))
        sf.write(out_paths.artifacts, synth_a.detach().cpu().numpy(), int(sampling_frequency_hz))

    return DecompositionResult(waveforms=waveforms, file_paths=out_paths)