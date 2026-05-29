# %%% extractDistortionComponents.m %%%
import os
import numpy as np
import scipy.signal as signal
import soundfile as sf
from myPemoAnalysisFilterBank import myPemoAnalysisFilterBank
from myPemoSynthesisFilterBank import myPemoSynthesisFilterBank
from extractTSIA import extractTSIA
from erbBW import erbBW


# function outputFilenames = extractDistortionComponents(srcFiles,estFile,options)
def extractDistortionComponents(srcFiles, estFile, options=None, fs=None):
    defaultOptions = {
        'destDir':            './',
        'FLAG_2PROJ':         False,
        'frameLength':        0.5,
        'filterLength':       0.04,
        'shadeInMs':          10,
        'shadeOutMs':         10,
        'segmentationFactor': 1
    }

    if options is None:
        options = defaultOptions
    else:
        for k, v in defaultOptions.items():
            if k not in options or options[k] is None:
                options[k] = v

    is_file_mode = isinstance(estFile, str)

    if is_file_mode:
        # Load from disk
        est_data, fs = sf.read(estFile)
        if len(est_data.shape) == 1:
            est_data = est_data[:, np.newaxis]

        src_data_list = []
        for src_path in srcFiles:
            data, fs_s = sf.read(src_path)
            if fs_s != fs:
                raise ValueError("Sampling rates of all files must match.")
            if len(data.shape) == 1:
                data = data[:, np.newaxis]
            src_data_list.append(data)
    else:
        # In-memory arrays
        est_data = np.atleast_2d(estFile)
        if est_data.shape[0] < est_data.shape[1]:
            est_data = est_data.T

        src_data_list = []
        for s_arr in srcFiles:
            s_arr = np.atleast_2d(s_arr)
            if s_arr.shape[0] < s_arr.shape[1]:
                s_arr = s_arr.T
            src_data_list.append(s_arr)

        if fs is None:
            raise ValueError("In-memory mode requires explicit sampling rate 'fs'.")

    # J = length(srcFiles);
    J = len(src_data_list)
    L_original = est_data.shape[0]

    # Check that all sounds have the same size
    for j, s_data in enumerate(src_data_list):
        if s_data.shape != est_data.shape:
            raise ValueError(f"Sound files must have the same size. "
                             f"Source {j}: {s_data.shape}, Estimate: {est_data.shape}")

    # NChan = infos_src.NumChannels;
    NChan = est_data.shape[1]

    def apply_shading(sig, fs, shade_in, shade_out):
        sig_shaded = sig.copy()
        # if options.shadeInMs>0
        if shade_in > 0:
            # wShadeIn = hann(2*round(options.shadeInMs/1000*fs+1),'periodic');
            # wShadeIn = wShadeIn(2:end/2);
            win_len = 2 * int(round(shade_in / 1000.0 * fs + 1))
            wShadeIn = signal.windows.hann(win_len, sym=False)[:win_len // 2]
            for c in range(sig_shaded.shape[1]):
                sig_shaded[:len(wShadeIn), c] *= wShadeIn
        # if options.shadeOutMs>0
        if shade_out > 0:
            # wShadeOut = hann(2*round(options.shadeOutMs/1000*fs+1),'periodic');
            # wShadeOut = wShadeOut(2:end/2);
            # wShadeOut = flipud(wShadeOut);
            win_len = 2 * int(round(shade_out / 1000.0 * fs + 1))
            wShadeOut = signal.windows.hann(win_len, sym=False)[:win_len // 2]
            wShadeOut = np.flip(wShadeOut)
            for c in range(sig_shaded.shape[1]):
                sig_shaded[-len(wShadeOut):, c] *= wShadeOut
        return sig_shaded

    src_shaded = [apply_shading(s, fs, options['shadeInMs'], options['shadeOutMs']) for s in src_data_list]
    est_shaded = apply_shading(est_data, fs, options['shadeInMs'], options['shadeOutMs'])

    # %% Auditory filter bank
    # sj_gamma = cell(J,NChan);
    sj_gamma = [[None for _ in range(NChan)] for _ in range(J)]
    # Mmod = [];
    Mmod = None
    analyzer = None

    # for j=1:J
    for j in range(J):
        for nChan in range(NChan):
            # [sj_gamma{j,nChan}, analyzer, Mmod] = myPemoAnalysisFilterBank(sj(:,nChan),fs,Mmod);
            sj_gamma[j][nChan], analyzer, Mmod = myPemoAnalysisFilterBank(src_shaded[j][:, nChan], fs, Mmod)

    # sj_est_gamma = cell(1,NChan);
    sj_est_gamma = [None for _ in range(NChan)]
    # for nChan = 1:NChan
    for nChan in range(NChan):
        # [sj_est_gamma{1,nChan}, analyzer] = myPemoAnalysisFilterBank(sj(:,nChan),fs,Mmod);
        sj_est_gamma[nChan], analyzer, _ = myPemoAnalysisFilterBank(est_shaded[:, nChan], fs, Mmod)

    Mmod = None

    # %% Bandwise component extraction
    # s = cellA12_cellB_vecC_2_cellB_vecCA21(sj_gamma);
    # sEst = cellA12_cellB_vecC_2_cellB_vecCA21(sj_est_gamma);
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

    # fRef = 1000;
    fRef = 1000
    # TframeFRef = options.frameLength;
    TframeFRef = options['frameLength']
    # ThopFRef = TframeFRef/4;
    ThopFRef = TframeFRef / 4.0
    # [dum I] = min(abs(analyzer.center_frequencies_hz-fRef));
    idx_fref = np.argmin(np.abs(analyzer['center_frequencies_hz'] - fRef))
    # bwRef = analyzer.bw(I);
    bwRef = analyzer['bw'][idx_fref]
    # fsb = analyzer.fs./analyzer.Ndec;
    fsb = analyzer['fs'] / analyzer['Ndec']
    # TfilterFRef = min(options.filterLength,TframeFRef/NChan/J/3);
    TfilterFRef = min(options['filterLength'], TframeFRef / NChan / J / 3.0)
    # flens = max(3, 2*round((TfilterFRef*bwRef./analyzer.bw.*fsb-1)/2)'+1);
    flens = np.maximum(3, 2 * np.round((TfilterFRef * bwRef / analyzer['bw'] * fsb - 1) / 2.0) + 1).astype(int)
    # Lws = max(3,round(TframeFRef*bwRef./analyzer.bw.*fsb))';
    Lws = np.maximum(3, np.round(TframeFRef * bwRef / analyzer['bw'] * fsb)).astype(int)
    # hops = max(1,round(ThopFRef*bwRef./analyzer.bw.*fsb))';
    hops = np.maximum(1, np.round(ThopFRef * bwRef / analyzer['bw'] * fsb)).astype(int)

    sgTrue, egTarget, egInterf, egArtif = [], [], [], []
    # for k=1:length(s)
    for b in range(Nb):
        # [sgTrue{k},egTarget{k},egInterf{k},egArtif{k}] = extractTSIA(s{k},sEst{k},flens(k),Lws(k),hops(k),options);
        sTrue_b, eSpat_b, eInterf_b, eArtif_b = extractTSIA(
            s[b], sEst[b], flens[b], Lws[b], hops[b], options
        )
        sgTrue.append(sTrue_b)
        egTarget.append(eSpat_b)
        egInterf.append(eInterf_b)
        egArtif.append(eArtif_b)

    # s_gamma_true = cellB_vecCA21_2_cellA12_cellB_vecC(sgTrue);
    # s_gamma_target = cellB_vecCA21_2_cellA12_cellB_vecC(egTarget);
    # s_gamma_interf = cellB_vecCA21_2_cellA12_cellB_vecC(egInterf);
    # s_gamma_artif = cellB_vecCA21_2_cellA12_cellB_vecC(egArtif);
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

    # %% Component synthesis
    trueSynth = np.zeros((L_original, NChan))
    targetSynth = np.zeros((L_original, NChan))
    interfSynth = np.zeros((L_original, NChan))
    artifSynth = np.zeros((L_original, NChan))

    # for nChan = 1:NChan
    for nChan in range(NChan):
        # [trueSynth(:,nChan), synthesizer, Mmod] = myPemoSynthesisFilterBank(s_gamma_true{1,nChan},analyzer,Mmod);
        synth_t, _, _ = myPemoSynthesisFilterBank(s_gamma_true[nChan], analyzer, Mmod)
        # [targetSynth(:,nChan), synthesizer, Mmod] = myPemoSynthesisFilterBank(s_gamma_target{1,nChan},analyzer,Mmod);
        synth_s, _, _ = myPemoSynthesisFilterBank(s_gamma_target[nChan], analyzer, Mmod)
        # [interfSynth(:,nChan), synthesizer, Mmod] = myPemoSynthesisFilterBank(s_gamma_interf{1,nChan},analyzer,Mmod);
        synth_i, _, _ = myPemoSynthesisFilterBank(s_gamma_interf[nChan], analyzer, Mmod)
        # [artifSynth(:,nChan), synthesizer, Mmod] = myPemoSynthesisFilterBank(s_gamma_artif{1,nChan},analyzer,Mmod);
        synth_a, _, _ = myPemoSynthesisFilterBank(s_gamma_artif[nChan], analyzer, Mmod)

        trueSynth[:, nChan] = synth_t[:L_original]
        targetSynth[:, nChan] = synth_s[:L_original]
        interfSynth[:, nChan] = synth_i[:L_original]
        artifSynth[:, nChan] = synth_a[:L_original]

    if is_file_mode:
        _, filename = os.path.split(estFile)
        name, _ = os.path.splitext(filename)
        out_filenames = [
            os.path.join(options['destDir'], f"{name}_true.wav"),
            os.path.join(options['destDir'], f"{name}_eTarget.wav"),
            os.path.join(options['destDir'], f"{name}_eInterf.wav"),
            os.path.join(options['destDir'], f"{name}_eArtif.wav")
        ]

        # Save output signals
        sf.write(out_filenames[0], trueSynth, fs)
        sf.write(out_filenames[1], targetSynth, fs)
        sf.write(out_filenames[2], interfSynth, fs)
        sf.write(out_filenames[3], artifSynth, fs)
        return out_filenames
    else:
        return trueSynth, targetSynth, interfSynth, artifSynth