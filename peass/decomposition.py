"""
PEASS Decomposition Package - Least-Squares Distortion Decomposer [1]

This module decomposes the separation error of a source estimate into:
  1. Target distortion (filter-induced alterations)
  2. Interference (leakage from overlapping sources)
  3. Artifacts (artificial noise / musical noise components)
"""

import pathlib
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import scipy.linalg as linalg
import scipy.signal as signal
import soundfile as sf

from .gammatone import GammatoneAnalyzer
from .gammatone import GammatoneSynthesizer


def least_squares_decompose(
        source_estimates: np.ndarray,
        true_sources: np.ndarray,
        filter_half_length: int,
        analysis_window: np.ndarray
) -> np.ndarray:
    """
    Weighted least-squares projection of source estimate on the source subspaces.
    Equivalent of LSDecompose.m [1].
    """
    filter_length = 2 * filter_half_length + 1
    num_sources = true_sources.shape[1]
    num_samples = source_estimates.shape[0]

    toeplitz_matrix = np.zeros((num_samples, num_sources * filter_length), dtype=true_sources.dtype)
    for j in range(num_sources):
        col = true_sources[filter_length - 1:, j]
        row = true_sources[filter_length - 1::-1, j]
        toeplitz_matrix[:, j * filter_length: (j + 1) * filter_length] = linalg.toeplitz(col, row)

    weighted_sources = analysis_window[:, np.newaxis] * toeplitz_matrix
    weighted_estimates = analysis_window[:, np.newaxis] * source_estimates

    gram_matrix = weighted_sources.conj().T @ weighted_sources
    reg_lambda = 10.0 ** -15

    try:
        cholesky_factor = linalg.cholesky(gram_matrix + reg_lambda * np.eye(gram_matrix.shape[0]), lower=False)
        test_condition = False
    except (linalg.LinAlgError, ValueError):
        test_condition = True

    if test_condition:
        projection_weights = np.linalg.pinv(weighted_sources) @ weighted_estimates
    else:
        b = weighted_sources.conj().T @ weighted_estimates
        tmp = linalg.solve_triangular(cholesky_factor.conj().T, b, lower=True)
        projection_weights = linalg.solve_triangular(cholesky_factor, tmp, lower=False)

    projections = np.zeros((num_samples, source_estimates.shape[1], num_sources), dtype=source_estimates.dtype)
    weighted_diag = analysis_window[:, np.newaxis]
    for j in range(num_sources):
        projections[:, :, j] = weighted_diag * (toeplitz_matrix[:, j * filter_length: (j + 1) * filter_length] @
                                                projection_weights[j * filter_length: (j + 1) * filter_length, :])

    return projections


def least_squares_decompose_time_varying(
        source_estimates: np.ndarray,
        true_sources: np.ndarray,
        filter_length: int,
        window_length: int,
        hop_size: int
) -> np.ndarray:
    """
    Time-varying least-squares subband decomposer.
    Equivalent of LSDecompose_tv.m [1].
    """
    filter_half_length = (filter_length - 1) // 2
    if (filter_length - 1) % 2 != 0:
        raise ValueError("Filter length must be an odd integer.")

    pad_length = filter_length - 1 + window_length - 1
    true_sources = np.pad(true_sources, ((0, pad_length), (0, 0)), mode='constant')
    source_estimates = np.pad(source_estimates, ((0, pad_length), (0, 0)), mode='constant')

    total_samples, num_sources = true_sources.shape
    num_channels = source_estimates.shape[1]

    # Periodic Hann windows
    hann_win = signal.windows.hann(window_length, sym=False)
    analysis_window = np.sqrt(np.flipud(hann_win))
    synthesis_window = np.sqrt(np.flipud(hann_win))

    synthesis_weights = np.zeros((window_length, num_channels, num_sources))
    for chan in range(num_channels):
        for j in range(num_sources):
            synthesis_weights[:, chan, j] = synthesis_window

    w_begin = 0
    w_end = w_begin + window_length

    projections_accum = np.zeros((total_samples, num_channels, num_sources), dtype=true_sources.dtype)
    window_accum = np.zeros((total_samples, 1))

    while w_end - window_length / 2.0 <= projections_accum.shape[0] - window_length + 1:
        frame_estimates = source_estimates[w_begin:w_end, :]

        sw_start = w_begin - filter_half_length
        sw_end = w_end + filter_half_length
        pad_left = max(0, -sw_start)
        pad_right = max(0, sw_end - true_sources.shape[0])
        slice_start = max(0, sw_start)
        slice_end = min(true_sources.shape[0], sw_end)

        frame_sources_slice = true_sources[slice_start:slice_end, :]
        frame_sources = np.vstack([
            np.zeros((pad_left, num_sources), dtype=true_sources.dtype),
            frame_sources_slice,
            np.zeros((pad_right, num_sources), dtype=true_sources.dtype)
        ])

        frame_projections = least_squares_decompose(frame_estimates, frame_sources, filter_half_length, analysis_window)

        projections_accum[w_begin:w_end, :, :] += frame_projections[:window_length, :, :] * synthesis_weights
        window_accum[w_begin:w_end, 0] += synthesis_window * analysis_window

        w_begin += hop_size
        w_end += hop_size

    valid_indices = (window_accum[:, 0] != 0)
    for j in range(num_sources):
        projections_accum[valid_indices, :, j] /= window_accum[valid_indices, :]

    return projections_accum[:-(window_length - 1), :, :]


def extract_target_spatial_interference_artifacts(
        true_sources: np.ndarray,
        source_estimates: np.ndarray,
        filter_length: int,
        window_length: int,
        hop_size: int,
        flag_two_projections: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits multi-source signal mixtures into Target, Spatial Distortion,
    Interference, and Artifact components. Replaces extractTSIA.m [1].
    """
    total_samples, num_channels, num_sources = true_sources.shape
    num_estimates = source_estimates.shape[2] if len(source_estimates.shape) > 2 else 1
    if len(source_estimates.shape) == 2:
        source_estimates = source_estimates[:, :, np.newaxis]

    sources_reshaped = true_sources.reshape((total_samples, num_sources * num_channels), order='F')
    estimates_reshaped = source_estimates.reshape((total_samples, num_estimates * num_channels), order='F')

    projections_all = least_squares_decompose_time_varying(estimates_reshaped, sources_reshaped, filter_length,
                                                           window_length, hop_size)

    y_projected = np.zeros((total_samples, num_channels * num_estimates, num_sources), dtype=true_sources.dtype)
    for nSource in range(num_sources):
        start_idx = nSource * num_channels
        end_idx = (nSource + 1) * num_channels
        y_projected[:, :, nSource] = np.sum(projections_all[:total_samples, :, start_idx:end_idx], axis=2)

    spatial_distortion = np.zeros((total_samples, num_estimates * num_channels), dtype=source_estimates.dtype)
    if flag_two_projections:
        for nEst in range(num_estimates):
            start_est = nEst * num_channels
            end_est = (nEst + 1) * num_channels
            spatial_proj = least_squares_decompose_time_varying(
                estimates_reshaped[:, start_est:end_est],
                sources_reshaped[:, :num_channels],
                filter_length, window_length, hop_size
            )
            spatial_distortion[:, start_est:end_est] = np.sum(spatial_proj[:total_samples, :, :], axis=2)

    true_reference = np.zeros((total_samples, num_channels * num_estimates), dtype=true_sources.dtype)
    for nEst in range(num_estimates):
        start_est = nEst * num_channels
        end_est = (nEst + 1) * num_channels
        true_reference[:, start_est:end_est] = sources_reshaped[:, :num_channels]

    if flag_two_projections:
        spatial_distortion = spatial_distortion - true_reference
    else:
        spatial_distortion = y_projected[:, :, :num_estimates].reshape((total_samples, num_estimates * num_channels),
                                                                       order='F') - true_reference

    interference = np.sum(y_projected, axis=2) - spatial_distortion - true_reference
    artifacts = estimates_reshaped - true_reference - spatial_distortion - interference

    true_reference_3d = true_reference.reshape((total_samples, num_channels, num_estimates), order='F')
    spatial_distortion_3d = spatial_distortion.reshape((total_samples, num_channels, num_estimates), order='F')
    interference_3d = interference.reshape((total_samples, num_channels, num_estimates), order='F')
    artifacts_3d = artifacts.reshape((total_samples, num_channels, num_estimates), order='F')

    return true_reference_3d, spatial_distortion_3d, interference_3d, artifacts_3d


def extract_distortion_components(
        src_files: List[Union[str, np.ndarray]],
        est_file: Union[str, np.ndarray],
        options: dict = None,
        sampling_frequency: float = None
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Subband least-squares decomposes estimates into distinct physical components.
    Replaces extractDistortionComponents.m [1].
    """
    default_options = {
        'destDir':            './',
        'FLAG_2PROJ':         False,
        'frameLength':        0.5,
        'filterLength':       0.04,
        'shadeInMs':          10,
        'shadeOutMs':         10,
        'segmentationFactor': 1
    }

    if options is None:
        options = default_options
    else:
        for k, v in default_options.items():
            if k not in options or options[k] is None:
                options[k] = v

    is_file_mode = isinstance(est_file, (str, pathlib.Path))

    if is_file_mode:
        est_data, sampling_frequency = sf.read(est_file)
        if len(est_data.shape) == 1:
            est_data = est_data[:, np.newaxis]

        src_data_list = []
        for src_path in src_files:
            data, fs_s = sf.read(src_path)
            if fs_s != sampling_frequency:
                raise ValueError("Sampling rates of all files must match.")
            if len(data.shape) == 1:
                data = data[:, np.newaxis]
            src_data_list.append(data)
    else:
        est_data = np.atleast_2d(est_file)
        if est_data.shape[0] < est_data.shape[1]:
            est_data = est_data.T

        src_data_list = []
        for s_arr in src_files:
            s_arr = np.atleast_2d(s_arr)
            if s_arr.shape[0] < s_arr.shape[1]:
                s_arr = s_arr.T
            src_data_list.append(s_arr)

        if sampling_frequency is None:
            raise ValueError("In-memory mode requires explicit sampling rate 'fs'.")

    J = len(src_data_list)
    L_original = est_data.shape[0]
    NChan = est_data.shape[1]

    for j, s_data in enumerate(src_data_list):
        if s_data.shape != est_data.shape:
            raise ValueError("All source signals must be of matching dimensions.")

    def apply_shading(sig, fs, shade_in, shade_out):
        sig_shaded = sig.copy()
        if shade_in > 0:
            win_len = 2 * int(round(shade_in / 1000.0 * fs + 1))
            wShadeIn = signal.windows.hann(win_len, sym=False)[:win_len // 2]
            for c in range(sig_shaded.shape[1]):
                sig_shaded[:len(wShadeIn), c] *= wShadeIn
        if shade_out > 0:
            win_len = 2 * int(round(shade_out / 1000.0 * fs + 1))
            wShadeOut = signal.windows.hann(win_len, sym=False)[:win_len // 2]
            wShadeOut = np.flip(wShadeOut)
            for c in range(sig_shaded.shape[1]):
                sig_shaded[-len(wShadeOut):, c] *= wShadeOut
        return sig_shaded

    src_shaded = [apply_shading(s, sampling_frequency, options['shadeInMs'], options['shadeOutMs']) for s in
                  src_data_list]
    est_shaded = apply_shading(est_data, sampling_frequency, options['shadeInMs'], options['shadeOutMs'])

    # Analysis Gammatone Filterbank
    sj_gamma = [[None for _ in range(NChan)] for _ in range(J)]
    Mmod = None
    analyzer = None

    for j in range(J):
        for nChan in range(NChan):
            sj_gamma[j][nChan], analyzer, Mmod = my_analysis_filter_bank(src_shaded[j][:, nChan], sampling_frequency,
                                                                         Mmod)

    sj_est_gamma = [None for _ in range(NChan)]
    for nChan in range(NChan):
        sj_est_gamma[nChan], analyzer, _ = my_analysis_filter_bank(est_shaded[:, nChan], sampling_frequency, Mmod)

    # Convert to subband blocks
    Nb = len(sj_gamma[0][0])
    s = []
    sEst = []
    for b in range(Nb):
        L_band = len(sj_gamma[0][0][b])
        s_band = np.zeros((L_band, NChan, J), dtype=complex)
        sEst_band = np.zeros((L_band, NChan, 1), dtype=complex)
        for nChan in range(NChan):
            sEst_band[:, nChan, 0] = sj_est_gamma[nChan][b]
            for j in range(J):
                s_band[:, nChan, j] = sj_gamma[j][nChan][b]
        s.append(s_band)
        sEst.append(sEst_band)

    fRef = 1000.0
    TframeFRef = options['frameLength']
    ThopFRef = TframeFRef / 4.0
    idx_fref = np.argmin(np.abs(analyzer.center_frequencies - fRef))
    bwRef = analyzer.bandwidths[idx_fref]

    # Corrected object-subscripting glitch:
    fsb = analyzer.sampling_frequency / analyzer.Ndec

    TfilterFRef = min(options['filterLength'], TframeFRef / NChan / J / 3.0)
    flens = np.maximum(3, 2 * np.round((TfilterFRef * bwRef / analyzer.bw * fsb - 1) / 2.0) + 1).astype(int)
    Lws = np.maximum(3, np.round(TframeFRef * bwRef / analyzer.bw * fsb)).astype(int)
    hops = np.maximum(1, np.round(ThopFRef * bwRef / analyzer.bw * fsb)).astype(int)

    sgTrue, egTarget, egInterf, egArtif = [], [], [], []
    for b in range(Nb):
        sTrue_b, eSpat_b, eInterf_b, eArtif_b = extract_target_spatial_interference_artifacts(
            s[b], sEst[b], flens[b], Lws[b], hops[b], flag_two_projections=options['FLAG_2PROJ']
        )
        sgTrue.append(sTrue_b)
        egTarget.append(eSpat_b)
        egInterf.append(eInterf_b)
        egArtif.append(eArtif_b)

    s_gamma_true = [[None for _ in range(Nb)] for _ in range(NChan)]
    s_gamma_target = [[None for _ in range(Nb)] for _ in range(NChan)]
    s_gamma_interf = [[None for _ in range(Nb)] for _ in range(NChan)]
    s_gamma_artif = [[None for _ in range(Nb)] for _ in range(NChan)]

    for nChan in range(NChan):
        for b in range(Nb):
            s_gamma_true[nChan][b] = sgTrue[b][:, nChan, 0]
            s_gamma_target[nChan][b] = egTarget[b][:, nChan, 0]
            s_gamma_interf[nChan][b] = egInterf[b][:, nChan, 0]
            s_gamma_artif[nChan][b] = egArtif[b][:, nChan, 0]

    trueSynth = np.zeros((L_original, NChan))
    targetSynth = np.zeros((L_original, NChan))
    interfSynth = np.zeros((L_original, NChan))
    artifSynth = np.zeros((L_original, NChan))

    def fit_to_length(sig, target_len):
        if len(sig) >= target_len:
            return sig[:target_len]
        return np.pad(sig, (0, target_len - len(sig)), mode='constant')

    for nChan in range(NChan):
        synth_t, _ = my_synthesis_filter_bank(s_gamma_true[nChan], analyzer)
        synth_s, _ = my_synthesis_filter_bank(s_gamma_target[nChan], analyzer)
        synth_i, _ = my_synthesis_filter_bank(s_gamma_interf[nChan], analyzer)
        synth_a, _ = my_synthesis_filter_bank(s_gamma_artif[nChan], analyzer)

        trueSynth[:, nChan] = fit_to_length(synth_t, L_original)
        targetSynth[:, nChan] = fit_to_length(synth_s, L_original)
        interfSynth[:, nChan] = fit_to_length(synth_i, L_original)
        artifSynth[:, nChan] = fit_to_length(synth_a, L_original)

    if is_file_mode:
        dest_path = pathlib.Path(options['destDir'])
        filename = pathlib.Path(est_file).stem
        out_filenames = [
            str(dest_path / f"{filename}_true.wav"),
            str(dest_path / f"{filename}_eTarget.wav"),
            str(dest_path / f"{filename}_eInterf.wav"),
            str(dest_path / f"{filename}_eArtif.wav")
        ]
        sf.write(out_filenames[0], trueSynth, int(sampling_frequency))
        sf.write(out_filenames[1], targetSynth, int(sampling_frequency))
        sf.write(out_filenames[2], interfSynth, int(sampling_frequency))
        sf.write(out_filenames[3], artifSynth, int(sampling_frequency))
        return out_filenames, [trueSynth, targetSynth, interfSynth, artifSynth]
    else:
        return [], [trueSynth, targetSynth, interfSynth, artifSynth]


def my_analysis_filter_bank(x: np.ndarray, fs: float, Mmod: np.ndarray = None):
    """Temporary local alias for packaging isolation."""
    from .gammatone import calculate_erb_bandwidth
    MinCF = 20.0
    MaxCF = fs / 2.0
    base_freq = 1000.0
    filters_per_ERB = 1.0

    fsOrig = fs
    if fs / 2.0 < 1.5 * MaxCF:
        new_fs = int(round(1.5 * fs))
        x = signal.resample(x, int(round(len(x) * new_fs / fs)))
        fs = new_fs

    analyzer = GammatoneAnalyzer(fs, MinCF, base_freq, MaxCF, filters_per_ERB)
    analyzer.fsOrig = fsOrig

    gfb_out = analyzer.process(x)
    Nb = gfb_out.shape[0]

    if Mmod is None:
        time_steps = np.arange(gfb_out.shape[1])
        cfs = analyzer.center_frequencies[:, np.newaxis]
        Mmod = np.exp(-2j * np.pi / fs * cfs * time_steps)

    gfb_out = gfb_out * Mmod

    bw = calculate_erb_bandwidth(analyzer.center_frequencies)
    alpha_dec = 2.0
    Ndec = np.maximum(1, np.floor(fs / (bw * alpha_dec))).astype(int)

    analyzer.Ndec = Ndec
    analyzer.fs = fs
    analyzer.bw = bw

    gfb_out_dec = []
    for k in range(Nb):
        decimated = signal.resample_poly(gfb_out[k, :], 1, Ndec[k])
        gfb_out_dec.append(decimated)

    return gfb_out_dec, analyzer, Mmod


def my_synthesis_filter_bank(xFB: list, analyzer: GammatoneAnalyzer):
    """Temporary local alias for packaging isolation."""
    Nb = len(xFB)
    fs = analyzer.fs

    max_len = max(len(xFB[k]) * analyzer.Ndec[k] for k in range(Nb))
    gfb_out_proc = np.zeros((Nb, max_len), dtype=complex)
    for k in range(Nb):
        target_len = len(xFB[k]) * analyzer.Ndec[k]
        upsampled = signal.resample_poly(xFB[k], analyzer.Ndec[k], 1)
        if len(upsampled) > target_len:
            upsampled = upsampled[:target_len]
        elif len(upsampled) < target_len:
            upsampled = np.pad(upsampled, (0, target_len - len(upsampled)), mode='constant')
        gfb_out_proc[k, :target_len] = upsampled

    time_steps = np.arange(max_len)
    cfs = analyzer.center_frequencies[:, np.newaxis]
    Mmod_synth = np.exp(2j * np.pi / fs * cfs * time_steps)

    gfb_out_proc = gfb_out_proc * Mmod_synth

    desired_delay_in_seconds = 1000.0 / fs
    synthesizer = GammatoneSynthesizer(analyzer, desired_delay_in_seconds)

    # Corrected object-oriented process execution directly matching interface definitions:
    output = synthesizer.process(gfb_out_proc)

    fsOrig = analyzer.fsOrig
    output = signal.resample(output, int(round(len(output) * fsOrig / fs)))
    delay_samples = int(round(desired_delay_in_seconds * fsOrig))
    output = output[delay_samples:]

    return output, synthesizer