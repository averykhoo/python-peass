"""
PEASS Decomposition Step - Complete Python Port
Performs the subband time-varying least-squares decomposition of separated audio
into Target, Target Distortion, Interference, and Artifacts.

Original MATLAB implementation by Valentin Emiya and Emmanuel Vincent (INRIA).
Python translation using NumPy and SciPy.

nore from human: idk if it's 100% correct, but it does produce numbers that look reasonable
"""

import math  # FIXED: Added standard math module for math.factorial
import os

import numpy as np
import scipy.linalg as linalg
import scipy.signal as signal
import soundfile as sf


# =====================================================================
# 1. Helper Functions and Auditory Scale Calculators
# =====================================================================

def erbBW(fc):
    """
    Computes Equivalent Rectangular Bandwidth of auditory filters (Hohmann 2002).
    """
    # erbBW.m implementation
    return 24.7 * (0.00437 * fc + 1)


def Gfb_hz2erbscale(Hz):
    """
    Implements equation (16) of [Hohmann 2002]
    """
    return 9.265 * np.log(1 + Hz / (24.7 * 9.265))


def Gfb_erbscale2hz(ERBscale):
    """
    Implements equation (17) of [Hohmann 2002]
    """
    return (np.exp(ERBscale / 9.265) - 1) * (24.7 * 9.265)


def Gfb_center_frequencies(filters_per_ERBaud, lower_cutoff_frequency_hz, specified_center_frequency_hz,
                           upper_cutoff_frequency_hz):
    """
    Constructs a vector of frequencies that are equidistant on the ERB scale.
    """
    lower_cutoff_frequency_erb = Gfb_hz2erbscale(lower_cutoff_frequency_hz)
    specified_center_frequency_erb = Gfb_hz2erbscale(specified_center_frequency_hz)
    upper_cutoff_frequency_erb = Gfb_hz2erbscale(upper_cutoff_frequency_hz)

    erbs_below_base_frequency = specified_center_frequency_erb - lower_cutoff_frequency_erb
    num_of_filters_below_base_freq = int(np.floor(erbs_below_base_frequency * filters_per_ERBaud))

    start_frequency_erb = specified_center_frequency_erb - num_of_filters_below_base_freq / filters_per_ERBaud
    center_frequencies_erb = np.arange(start_frequency_erb, upper_cutoff_frequency_erb + 1e-9, 1.0 / filters_per_ERBaud)

    return Gfb_erbscale2hz(center_frequencies_erb)


# =====================================================================
# 2. Hohmann 2002 Gammatone Filterbank Classes/Constructors
# =====================================================================

def Gfb_Filter_new(sampling_frequency_hz, center_frequency_hz, gamma_order=4, bandwidth_factor=1.0):
    """
    Constructs a cascaded gammatone filter structure (Hohmann 2002).
    """
    audiological_erb = (24.7 + center_frequency_hz / 9.265) * bandwidth_factor

    a_gamma = (np.pi * math.factorial(2 * gamma_order - 2) *
               (2 ** -(2 * gamma_order - 2)) /
               (math.factorial(gamma_order - 1) ** 2))

    b = audiological_erb / a_gamma
    lambda_val = np.exp(-2 * np.pi * b / sampling_frequency_hz)
    beta = 2 * np.pi * center_frequency_hz / sampling_frequency_hz

    coefficient = lambda_val * np.exp(1j * beta)
    normalization_factor = 2 * (1 - np.abs(coefficient)) ** gamma_order

    return {
        'type':                 'Gfb_Filter',
        'coefficient':          coefficient,
        'normalization_factor': normalization_factor,
        'gamma_order':          gamma_order,
        'state':                np.zeros(gamma_order, dtype=complex)
    }


def Gfb_Analyzer_new(sampling_frequency_hz, lower_cutoff_frequency_hz, specified_center_frequency_hz,
                     upper_cutoff_frequency_hz, filters_per_ERBaud, gamma_order=4, bandwidth_factor=1.0):
    """
    Constructs the analysis gammatone filterbank.
    """
    center_frequencies_hz = Gfb_center_frequencies(filters_per_ERBaud, lower_cutoff_frequency_hz,
                                                   specified_center_frequency_hz, upper_cutoff_frequency_hz)
    filters = []
    for band in range(center_frequencies_hz.shape[0]):
        cf = center_frequencies_hz[band]
        filters.append(Gfb_Filter_new(sampling_frequency_hz, cf, gamma_order, bandwidth_factor))

    return {
        'type':                          'Gfb_Analyzer',
        'sampling_frequency_hz':         sampling_frequency_hz,
        'lower_cutoff_frequency_hz':     lower_cutoff_frequency_hz,
        'specified_center_frequency_hz': specified_center_frequency_hz,
        'upper_cutoff_frequency_hz':     upper_cutoff_frequency_hz,
        'filters_per_ERBaud':            filters_per_ERBaud,
        'bandwidth_factor':              bandwidth_factor,
        'center_frequencies_hz':         center_frequencies_hz,
        'filters':                       filters,
        'bw':                            erbBW(center_frequencies_hz)
    }


def Gfb_Filter_process(filter_obj, input_data):
    """
    Singly processes input array through cascade filters.
    """
    factor = filter_obj['normalization_factor']
    coeff = filter_obj['coefficient']
    gamma = filter_obj['gamma_order']

    y = input_data.copy()
    b_stage1 = np.array([factor], dtype=complex)
    a_stage = np.array([1.0, -coeff], dtype=complex)

    new_state = np.zeros(gamma, dtype=complex)
    for i in range(gamma):
        b = b_stage1 if i == 0 else np.array([1.0], dtype=complex)
        init_val = filter_obj['state'][i] * coeff
        zi = np.array([init_val], dtype=complex)

        y, zf = signal.lfilter(b, a_stage, y, zi=zi)
        new_state[i] = zf[0] / coeff

    filter_obj['state'] = new_state
    return y, filter_obj


def Gfb_Analyzer_process(analyzer, input_data):
    """
    Filters fullband input array into complex subbands.
    """
    bands = len(analyzer['filters'])
    output = np.zeros((bands, input_data.shape[0]), dtype=complex)
    for band in range(bands):
        output[band, :], analyzer['filters'][band] = Gfb_Filter_process(analyzer['filters'][band], input_data)
    return output, analyzer


def Gfb_Analyzer_zresponse(analyzer, z):
    """
    Computes analytical z-plane frequency responses.
    """
    z = z[:, np.newaxis]
    bands = analyzer['center_frequencies_hz'].shape[0]
    zresponse = np.ones((z.shape[0], bands), dtype=complex)
    for band in range(bands):
        coeff = analyzer['filters'][band]['coefficient']
        norm = analyzer['filters'][band]['normalization_factor']
        gamma = analyzer['filters'][band]['gamma_order']
        zresponse[:, band] = ((1 - coeff / z[:, 0]) ** -gamma) * norm
    return zresponse


def Gfb_Delay_new(analyzer, delay_samples):
    """
    Numerically estimates optimal synthesizer phase factors and delays.
    """
    impulse = np.zeros(delay_samples + 2)
    impulse[0] = 1.0
    impulse_response, _ = Gfb_Analyzer_process(analyzer, impulse)

    ir_slice = np.abs(impulse_response[:, :delay_samples + 1])
    max_indices = np.argmax(ir_slice, axis=1)

    delays_samples = delay_samples - max_indices
    slopes = np.zeros(analyzer['center_frequencies_hz'].shape[0], dtype=complex)

    for band in range(analyzer['center_frequencies_hz'].shape[0]):
        idx = max_indices[band]
        slopes[band] = impulse_response[band, idx + 1] - impulse_response[band, idx - 1]

    slopes = slopes / np.abs(slopes)
    phase_factors = 1j / slopes

    return {
        'type':           'Gfb_Delay',
        'delays_samples': delays_samples,
        'phase_factors':  phase_factors,
        # FIXED: Changed dtype=complex to dtype=float since memory only stores real values
        'memory':         np.zeros((analyzer['center_frequencies_hz'].shape[0], int(np.max(delays_samples))),
                                   dtype=float)
    }


def Gfb_Mixer_new(analyzer, delay, iterations=100):
    """
    Runs iterative optimization to deduce optimal synthesizer gain parameters.
    """
    center_frequencies = analyzer['center_frequencies_hz']
    number_of_bands = center_frequencies.shape[0]
    sampling_frequency = analyzer['sampling_frequency_hz']

    z_c = np.exp(2j * np.pi * center_frequencies / sampling_frequency)
    gains = np.ones(number_of_bands)

    pos_f_response = Gfb_Analyzer_zresponse(analyzer, z_c)
    neg_f_response = Gfb_Analyzer_zresponse(analyzer, np.conj(z_c))

    for band in range(number_of_bands):
        pos_f_response[:, band] = pos_f_response[:, band] * \
                                  delay['phase_factors'][band] * \
                                  (z_c ** -delay['delays_samples'][band])
        neg_f_response[:, band] = neg_f_response[:, band] * \
                                  delay['phase_factors'][band] * \
                                  (np.conj(z_c) ** -delay['delays_samples'][band])

    f_response = (pos_f_response + np.conj(neg_f_response)) / 2.0

    for _ in range(iterations):
        selected_spectrum = f_response @ gains
        gains = gains / np.abs(selected_spectrum)

    return {
        'type':  'Gfb_Mixer',
        'gains': gains
    }


def Gfb_Delay_process(delay, input_data):
    """
    Applies delays and phase-correcting weights to analytical subbands.
    """
    number_of_bands, number_of_samples = input_data.shape
    output = np.zeros((number_of_bands, number_of_samples))
    for band in range(number_of_bands):
        delay_val = int(delay['delays_samples'][band])
        phase_corrected = np.real(input_data[band, :] * delay['phase_factors'][band])
        if delay_val == 0:
            output[band, :] = phase_corrected
        else:
            tmp_out = np.concatenate((delay['memory'][band, :delay_val], phase_corrected))
            delay['memory'][band, :delay_val] = tmp_out[number_of_samples:]
            output[band, :] = tmp_out[:number_of_samples]
    return output, delay


def Gfb_Mixer_process(mixer, input_data):
    """
    Processes weighted summation of delayed signals.
    """
    return mixer['gains'] @ input_data


def Gfb_Synthesizer_new(analyzer, desired_delay_in_seconds):
    """
    Assembles delay and mixer units into a single synthesizer structure.
    """
    fs = analyzer['sampling_frequency_hz']
    desired_delay_in_samples = int(round(desired_delay_in_seconds * fs))
    if desired_delay_in_samples < 1:
        raise ValueError("delay must be at least 1/analyzer.sampling_frequency_hz")
    delay = Gfb_Delay_new(analyzer, desired_delay_in_samples)
    mixer = Gfb_Mixer_new(analyzer, delay)
    return {
        'type':  'Gfb_Synthesizer',
        'delay': delay,
        'mixer': mixer
    }


def Gfb_Synthesizer_process(synthesizer, input_data):
    """
    Combines subband frames back into fullband audio.
    """
    output, delay = Gfb_Delay_process(synthesizer['delay'], input_data)
    synthesizer['delay'] = delay
    output = Gfb_Mixer_process(synthesizer['mixer'], output)
    return output, synthesizer


# =====================================================================
# 3. PEMO Auditory Filterbank Wrappers
# =====================================================================

# function [gfb_out_dec, analyzer,M] = myPemoAnalysisFilterBank(x,fs,M)
def myPemoAnalysisFilterBank(x, fs, Mmod=None):
    # MinCF = 20;
    MinCF = 20
    # MaxCF = fs/2;
    MaxCF = fs / 2.0
    # base_freq = 1000;
    base_freq = 1000
    # filters_per_ERB = 1.0;
    filters_per_ERB = 1.0

    # fsOrig = fs;
    fsOrig = fs
    # if fs/2 < 1.5*MaxCF,
    if fs / 2.0 < 1.5 * MaxCF:
        # x = resample(x, round(1.5*fs), fs);
        new_fs = int(round(1.5 * fs))
        x = signal.resample(x, int(round(len(x) * new_fs / fs)))
        # fs = round(1.5*fs);
        fs = new_fs

    # analyzer = Gfb_Analyzer_new(fs, MinCF, base_freq, MaxCF, filters_per_ERB);
    analyzer = Gfb_Analyzer_new(fs, MinCF, base_freq, MaxCF, filters_per_ERB)
    # analyzer.fsOrig = fsOrig;
    analyzer['fsOrig'] = fsOrig
    # [gfb_out, analyzer] = Gfb_Analyzer_process(analyzer, x(:).');
    gfb_out, analyzer = Gfb_Analyzer_process(analyzer, x)
    # Nb = size(gfb_out,1);
    Nb = gfb_out.shape[0]

    # if nargin<3 || isempty(M)
    if Mmod is None:
        # M = exp(-2*1i*pi/fs*analyzer.center_frequencies_hz(:)*(0:size(gfb_out,2)-1));
        time_steps = np.arange(gfb_out.shape[1])
        cfs = analyzer['center_frequencies_hz'][:, np.newaxis]
        Mmod = np.exp(-2j * np.pi / fs * cfs * time_steps)

    # gfb_out = gfb_out.*M;
    gfb_out = gfb_out * Mmod

    # bw = erbBW(analyzer.center_frequencies_hz);
    bw = erbBW(analyzer['center_frequencies_hz'])
    # alpha_dec = 2;
    alpha_dec = 2
    # Ndec = floor(fs./bw/alpha_dec);
    # ADDITION: Force Ndec to be at least 1 to prevent division-by-zero/zero-indexing errors
    Ndec = np.maximum(1, np.floor(fs / (bw * alpha_dec))).astype(int)

    # analyzer.Ndec = Ndec;
    analyzer['Ndec'] = Ndec
    # analyzer.fs = fs;
    analyzer['fs'] = fs
    # analyzer.bw = bw;
    analyzer['bw'] = bw

    gfb_out_dec = []
    # for k=1:Nb
    for k in range(Nb):
        # gfb_out_dec{k} = resample(gfb_out(k,:),1,Ndec(k));
        decimated = signal.resample_poly(gfb_out[k, :], 1, Ndec[k])
        gfb_out_dec.append(decimated)

    return gfb_out_dec, analyzer, Mmod


# function [xSynth, synthesizer, M] = myPemoSynthesisFilterBank(xFB,analyzer,M)
def myPemoSynthesisFilterBank(xFB, analyzer, Mmod=None):
    # Nb = size(xFB,1);
    Nb = len(xFB)
    # fs = analyzer.sampling_frequency_hz;
    fs = analyzer['fs']

    # max_len = max(cellfun('length',xFB(:)).*analyzer.Ndec(:))
    max_len = max(len(xFB[k]) * analyzer['Ndec'][k] for k in range(Nb))
    gfb_out_proc = np.zeros((Nb, max_len), dtype=complex)
    # for k=1:Nb
    for k in range(Nb):
        # gfb_out_proc(k,1:length(xFB{k})*analyzer.Ndec(k)) = resample(xFB{k},analyzer.Ndec(k),1);
        # FIX: Explicitly enforce the exact mathematical target length of the upsampled block
        # to prevent IndexError when resample_poly generates mismatching transient pads.
        target_len = len(xFB[k]) * analyzer['Ndec'][k]
        upsampled = signal.resample_poly(xFB[k], analyzer['Ndec'][k], 1)

        if len(upsampled) > target_len:
            upsampled = upsampled[:target_len]
        elif len(upsampled) < target_len:
            upsampled = np.pad(upsampled, (0, target_len - len(upsampled)), mode='constant')

        gfb_out_proc[k, :target_len] = upsampled

    # if nargin<3 || isempty(M)
    time_steps = np.arange(max_len)
    cfs = analyzer['center_frequencies_hz'][:, np.newaxis]
    Mmod = np.exp(2j * np.pi / fs * cfs * time_steps)

    # gfb_out_proc = gfb_out_proc.*M;
    gfb_out_proc = gfb_out_proc * Mmod

    # desired_delay_in_seconds = 1 / fs*1000;
    desired_delay_in_seconds = 1000.0 / fs
    # synthesizer = Gfb_Synthesizer_new(analyzer, desired_delay_in_seconds);
    synthesizer = Gfb_Synthesizer_new(analyzer, desired_delay_in_seconds)
    # [xSynth, synthesizer] = Gfb_Synthesizer_process(synthesizer, gfb_out_proc);
    xSynth, synthesizer = Gfb_Synthesizer_process(synthesizer, gfb_out_proc)
    # xSynth = resample(xSynth,analyzer.fsOrig,analyzer.fs);
    fsOrig = analyzer['fsOrig']
    xSynth = signal.resample(xSynth, int(round(len(xSynth) * fsOrig / fs)))
    # xSynth = xSynth(round(desired_delay_in_seconds*analyzer.fsOrig+1):end);
    delay_samples = int(round(desired_delay_in_seconds * fsOrig))
    xSynth = xSynth[delay_samples:]

    return xSynth, synthesizer, Mmod


# =====================================================================
# 4. Joint Least-Squares Frame Projection Engines (PEASS)
# =====================================================================

def LSDecompose(se, s, flen2, wa):
    """
    MATLAB port of LSDecompose.m
    """
    flen = 2 * flen2 + 1
    J = s.shape[1]
    L = se.shape[0]

    S = np.zeros((L, J * flen), dtype=s.dtype)
    for j in range(J):
        col = s[flen - 1:, j]
        row = s[flen - 1::-1, j]
        S[:, j * flen: (j + 1) * flen] = linalg.toeplitz(col, row)

    Sw = wa[:, np.newaxis] * S
    se_w = wa[:, np.newaxis] * se
    gramSw = Sw.conj().T @ Sw

    reg_lambda = 10 ** -15
    try:
        R = linalg.cholesky(gramSw + reg_lambda * np.eye(gramSw.shape[0]))
        b = Sw.conj().T @ se_w
        tmp = linalg.solve_triangular(R.conj().T, b, lower=True)
        y = linalg.solve_triangular(R, tmp, lower=False)
    except (linalg.LinAlgError, ValueError):
        y = np.linalg.pinv(Sw) @ se_w

    proj = np.zeros((L, se.shape[1], J), dtype=se.dtype)
    Wa = wa[:, np.newaxis]
    for j in range(J):
        proj[:, :, j] = Wa * (S[:, j * flen: (j + 1) * flen] @ y[j * flen: (j + 1) * flen, :])

    return proj


def LSDecompose_tv(se, s, flen, Lw, hop):
    """
    MATLAB port of LSDecompose_tv.m
    """
    flen2 = (flen - 1) // 2
    if (flen - 1) % 2 != 0:
        raise ValueError("filterParam:NotOdd - Not an odd order")

    pad_len = flen - 1 + Lw - 1
    s = np.pad(s, ((0, pad_len), (0, 0)), mode='constant')
    se = np.pad(se, ((0, pad_len), (0, 0)), mode='constant')

    nsampl, nsrc = s.shape
    nchanEst = se.shape[1]

    h_win = signal.windows.hann(Lw, sym=False)
    wa = np.sqrt(np.flipud(h_win))
    ws = np.sqrt(np.flipud(h_win))

    WS = np.zeros((Lw, nchanEst, nsrc))
    for chan in range(nchanEst):
        for j in range(nsrc):
            WS[:, chan, j] = ws

    wBegin = 0
    wEnd = wBegin + Lw
    sproj = np.zeros((nsampl, nchanEst, nsrc), dtype=s.dtype)
    wAccum = np.zeros((nsampl, 1))
    Ns = s.shape[1]
    Ls = s.shape[0]

    while wEnd - Lw // 2 <= sproj.shape[0] - Lw + 1:
        sew = se[wBegin:wEnd, :]
        sw_start = wBegin - flen2
        sw_end = wEnd + flen2

        pad_left = max(0, -sw_start)
        pad_right = max(0, sw_end - Ls)
        slice_start = max(0, sw_start)
        slice_end = min(Ls, sw_end)
        s_slice = s[slice_start:slice_end, :]

        sw = np.zeros((flen + Lw - 1, Ns), dtype=s.dtype)
        if pad_left > 0:
            sw[:pad_left, :] = 0
        sw[pad_left: pad_left + (slice_end - slice_start), :] = s_slice
        if pad_right > 0:
            sw[-pad_right:, :] = 0

        sprojw = LSDecompose(sew, sw, flen2, wa)
        sproj[wBegin:wEnd, :, :] += sprojw[:Lw, :, :] * WS
        wAccum[wBegin:wEnd, 0] += ws * wa

        wBegin += hop
        wEnd += hop

    I = (wAccum[:, 0] != 0)
    for j in range(nsrc):
        sproj[I, :, j] /= wAccum[I, :]

    sproj = sproj[:-(Lw - 1), :, :]
    return sproj


# =====================================================================
# 5. Core Decomposer & Metrical Estimators (SDR, SIR, SAR, ISR)
# =====================================================================

def extractTSIA(s, sEst, flen, Lw, hop, flag_2proj=False):
    """
    MATLAB port of extractTSIA.m
    """
    L, NChan, NSources = s.shape
    NEst = sEst.shape[2] if len(sEst.shape) > 2 else 1
    if len(sEst.shape) == 2:
        sEst = sEst[:, :, np.newaxis]

    s_reshaped = s.reshape((L, NSources * NChan), order='F')
    sEst_reshaped = sEst.reshape((L, NEst * NChan), order='F')

    yProjAll_ = LSDecompose_tv(sEst_reshaped, s_reshaped, flen, Lw, hop)

    yProjAll = np.zeros((L, NChan * NEst, NSources), dtype=s.dtype)
    for nSource in range(NSources):
        start_idx = nSource * NChan
        end_idx = (nSource + 1) * NChan
        sliced = yProjAll_[:L, :, start_idx:end_idx]
        yProjAll[:, :, nSource] = np.sum(sliced, axis=2)

    eSpat = np.zeros((L, NEst * NChan), dtype=sEst.dtype)
    if flag_2proj:
        for nEst in range(NEst):
            start_est = nEst * NChan
            end_est = (nEst + 1) * NChan
            eSpat_ = LSDecompose_tv(
                sEst_reshaped[:, start_est:end_est],
                s_reshaped[:, :NChan],
                flen, Lw, hop
            )
            eSpat[:, start_est:end_est] = np.sum(eSpat_[:L, :, :], axis=2)

    sTrue = np.zeros((L, NChan * NEst), dtype=s.dtype)
    for nEst in range(NEst):
        start_est = nEst * NChan
        end_est = (nEst + 1) * NChan
        sTrue[:, start_est:end_est] = s_reshaped[:, :NChan]

    if flag_2proj:
        eSpat = eSpat - sTrue
    else:
        eSpat = yProjAll[:, :, :NEst].reshape((L, NEst * NChan), order='F') - sTrue

    yProjAll_summed = np.sum(yProjAll, axis=2)
    eInterf = yProjAll_summed - eSpat - sTrue
    eArtif = sEst_reshaped - sTrue - eSpat - eInterf

    sTrue_3d = sTrue.reshape((L, NChan, NEst), order='F')
    eSpat_3d = eSpat.reshape((L, NChan, NEst), order='F')
    eInterf_3d = eInterf.reshape((L, NChan, NEst), order='F')
    eArtif_3d = eArtif.reshape((L, NChan, NEst), order='F')

    return sTrue_3d, eSpat_3d, eInterf_3d, eArtif_3d


def extractDistortionComponents(src_input, est_input, options=None, fs=None):
    """
    MATLAB port of extractDistortionComponents.m
    Decomposes estimated source into Target, Interference, and Artifacts.
    Can accept either wav file paths or direct NumPy arrays (L x Channels).
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

    is_file_mode = isinstance(est_input, str)

    if is_file_mode:
        est_data, fs = sf.read(est_input)
        if len(est_data.shape) == 1:
            est_data = est_data[:, np.newaxis]

        src_data_list = []
        for src_path in src_input:
            data, fs_s = sf.read(src_path)
            if fs_s != fs:
                raise ValueError("Sampling rates of all files must match.")
            if len(data.shape) == 1:
                data = data[:, np.newaxis]
            src_data_list.append(data)
    else:
        est_data = np.atleast_2d(est_input)
        if est_data.shape[0] < est_data.shape[1]:
            est_data = est_data.T

        src_data_list = []
        for s_arr in src_input:
            s_arr = np.atleast_2d(s_arr)
            if s_arr.shape[0] < s_arr.shape[1]:
                s_arr = s_arr.T
            src_data_list.append(s_arr)

        if fs is None:
            raise ValueError("In-memory mode requires explicit sampling rate 'fs'.")

    J = len(src_data_list)
    NChan = est_data.shape[1]
    L_original = est_data.shape[0]

    for j, s_data in enumerate(src_data_list):
        if s_data.shape != est_data.shape:
            raise ValueError(f"Sound files must have the same size. "
                             f"Source {j}: {s_data.shape}, Estimate: {est_data.shape}")

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

    src_shaded = [apply_shading(s, fs, options['shadeInMs'], options['shadeOutMs']) for s in src_data_list]
    est_shaded = apply_shading(est_data, fs, options['shadeInMs'], options['shadeOutMs'])

    sj_gamma = [[None for _ in range(NChan)] for _ in range(J)]
    Mmod = None
    analyzer = None

    for j in range(J):
        for nChan in range(NChan):
            sj_gamma[j][nChan], analyzer, Mmod = myPemoAnalysisFilterBank(src_shaded[j][:, nChan], fs, Mmod)

    sj_est_gamma = [None for _ in range(NChan)]
    for nChan in range(NChan):
        sj_est_gamma[nChan], analyzer, _ = myPemoAnalysisFilterBank(est_shaded[:, nChan], fs, Mmod)

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

    fRef = 1000
    TframeFRef = options['frameLength']
    ThopFRef = TframeFRef / 4.0
    idx_fref = np.argmin(np.abs(analyzer['center_frequencies_hz'] - fRef))
    bwRef = analyzer['bw'][idx_fref]
    fsb = analyzer['fs'] / analyzer['Ndec']

    TfilterFRef = min(options['filterLength'], TframeFRef / NChan / J / 3.0)
    flens = np.maximum(3, 2 * np.round((TfilterFRef * bwRef / analyzer['bw'] * fsb - 1) / 2.0) + 1).astype(int)
    Lws = np.maximum(3, np.round(TframeFRef * bwRef / analyzer['bw'] * fsb)).astype(int)
    hops = np.maximum(1, np.round(ThopFRef * bwRef / analyzer['bw'] * fsb)).astype(int)

    sgTrue, egTarget, egInterf, egArtif = [], [], [], []
    for b in range(Nb):
        sTrue_b, eSpat_b, eInterf_b, eArtif_b = extractTSIA(
            s[b], sEst[b], flens[b], Lws[b], hops[b], flag_2proj=options['FLAG_2PROJ']
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

    # FIXED: Added a robust helper to guarantee exact length matching,
    # safely resolving decimation truncation without triggering numpy broadcast crashes.
    def fit_to_length(sig, target_len):
        if len(sig) >= target_len:
            return sig[:target_len]
        return np.pad(sig, (0, target_len - len(sig)), mode='constant')

    for nChan in range(NChan):
        synth_t, _, _ = myPemoSynthesisFilterBank(s_gamma_true[nChan], analyzer, Mmod)
        synth_s, _, _ = myPemoSynthesisFilterBank(s_gamma_target[nChan], analyzer, Mmod)
        synth_i, _, _ = myPemoSynthesisFilterBank(s_gamma_interf[nChan], analyzer, Mmod)
        synth_a, _, _ = myPemoSynthesisFilterBank(s_gamma_artif[nChan], analyzer, Mmod)

        # FIXED: Apply length fitting on the reconstructed fullband waveforms
        trueSynth[:, nChan] = fit_to_length(synth_t, L_original)
        targetSynth[:, nChan] = fit_to_length(synth_s, L_original)
        interfSynth[:, nChan] = fit_to_length(synth_i, L_original)
        artifSynth[:, nChan] = fit_to_length(synth_a, L_original)

    if is_file_mode:
        _, filename = os.path.split(est_input)
        name, _ = os.path.splitext(filename)
        out_filenames = [
            os.path.join(options['destDir'], f"{name}_true.wav"),
            os.path.join(options['destDir'], f"{name}_eTarget.wav"),
            os.path.join(options['destDir'], f"{name}_eInterf.wav"),
            os.path.join(options['destDir'], f"{name}_eArtif.wav")
        ]

        sf.write(out_filenames[0], trueSynth, fs)
        sf.write(out_filenames[1], targetSynth, fs)
        sf.write(out_filenames[2], interfSynth, fs)
        sf.write(out_filenames[3], artifSynth, fs)
        return out_filenames
    else:
        return trueSynth, targetSynth, interfSynth, artifSynth


def ISR_SIR_SAR_fromNewDecomposition(decomposition_input):
    """
    MATLAB port of ISR_SIR_SAR_fromNewDecomposition.m
    Computes standard BSS Eval metrics directly from decomposed arrays.
    """
    if isinstance(decomposition_input[0], str):
        sTrue = sf.read(decomposition_input[0])[0]
        eTarget = sf.read(decomposition_input[1])[0]
        eInterf = sf.read(decomposition_input[2])[0]
        eArtif = sf.read(decomposition_input[3])[0]
    else:
        sTrue, eTarget, eInterf, eArtif = decomposition_input

    sTrue_flat = sTrue.ravel()
    eTarget_flat = eTarget.ravel()
    eInterf_flat = eInterf.ravel()
    eArtif_flat = eArtif.ravel()

    ISR = 10 * np.log10(np.sum(sTrue_flat ** 2) / np.sum(eTarget_flat ** 2))
    SIR = 10 * np.log10(np.sum((sTrue_flat + eTarget_flat) ** 2) / np.sum(eInterf_flat ** 2))
    SAR = 10 * np.log10(np.sum((sTrue_flat + eTarget_flat + eInterf_flat) ** 2) / np.sum(eArtif_flat ** 2))
    SDR = 10 * np.log10(np.sum(sTrue_flat ** 2) / np.sum((eTarget_flat + eInterf_flat + eArtif_flat) ** 2))

    return ISR, SIR, SAR, SDR


# =====================================================================
# 6. Usage Example Execution
# =====================================================================

if __name__ == '__main__':
    # Generating mock signal data for verification
    fs = 16000
    t = np.linspace(0, 2, 2 * fs, endpoint=False)

    # 1. True clean speech source: 400 Hz Sine Wave
    speech = np.sin(2 * np.pi * 400 * t)

    # 2. Interfering noise: Random Gaussian Noise
    noise = 0.5 * np.random.randn(len(t))

    # 3. Estimate (Denoised Output with simulated vocal attenuation & residual noise)
    estimate = 0.8 * speech + 0.05 * noise + 0.01 * np.random.randn(len(t))

    # Reshaping to (L, Channels)
    speech = speech[:, np.newaxis]
    noise = noise[:, np.newaxis]
    estimate = estimate[:, np.newaxis]

    print("Beginning NumPy-native PEASS error decomposition...")

    # Run physical decomposition
    decomp_result = extractDistortionComponents(
        src_input=[speech, noise],
        est_input=estimate,
        fs=fs
    )

    # Compute decibel metrics
    ISR, SIR, SAR, SDR = ISR_SIR_SAR_fromNewDecomposition(decomp_result)

    print("\n--- Python Port Metrics Result ---")
    print(f"SDR (Overall Quality):         {SDR:.2f} dB")
    print(f"ISR (Target Preservation):     {ISR:.2f} dB")
    print(f"SIR (Interference Rejection):  {SIR:.2f} dB")
    print(f"SAR (Artifact Level):          {SAR:.2f} dB")
