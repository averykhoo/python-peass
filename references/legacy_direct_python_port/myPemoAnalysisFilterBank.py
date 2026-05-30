"""
PEASS Toolkit - Python Port
Equivalent of myPemoAnalysisFilterBank.m
"""
import numpy as np
import scipy.signal as signal

from erbBW import erbBW
from gammatone_helper import Gfb_Analyzer_new
from gammatone_helper import Gfb_Analyzer_process


def myPemoAnalysisFilterBank(x: np.ndarray, fs: float, Mmod: np.ndarray = None):
    # %%% MATLAB Code %%%
    # MinCF = 20; MaxCF = fs/2; base_freq = 1000; filters_per_ERB = 1.0;
    MinCF = 20.0
    MaxCF = fs / 2.0
    base_freq = 1000.0
    filters_per_ERB = 1.0

    # fsOrig = fs;
    fsOrig = fs
    # if fs/2 < 1.5*MaxCF, x = resample(x, round(1.5*fs), fs); fs = round(1.5*fs); end
    if fs / 2.0 < 1.5 * MaxCF:
        new_fs = int(round(1.5 * fs))
        x = signal.resample(x, int(round(len(x) * new_fs / fs)))
        fs = new_fs

    # analyzer = Gfb_Analyzer_new(fs, MinCF, base_freq, MaxCF, filters_per_ERB);
    analyzer = Gfb_Analyzer_new(fs, MinCF, base_freq, MaxCF, filters_per_ERB)
    analyzer['fsOrig'] = fsOrig

    # [gfb_out, analyzer] = Gfb_Analyzer_process(analyzer, x(:).');
    gfb_out, analyzer = Gfb_Analyzer_process(analyzer, x)
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
    # alpha_dec = 2; Ndec = floor(fs./bw/alpha_dec);
    alpha_dec = 2.0
    Ndec = np.maximum(1, np.floor(fs / (bw * alpha_dec))).astype(int)

    analyzer['Ndec'] = Ndec
    analyzer['fs'] = fs
    analyzer['bw'] = bw

    # gfb_out_dec = cell(Nb,1);
    gfb_out_dec = []
    for k in range(Nb):
        # gfb_out_dec{k} = resample(gfb_out(k,:),1,Ndec(k));
        decimated = signal.resample_poly(gfb_out[k, :], 1, Ndec[k])
        gfb_out_dec.append(decimated)

    return gfb_out_dec, analyzer, Mmod
