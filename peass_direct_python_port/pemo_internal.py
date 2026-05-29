"""
PEASS Toolkit - Python Port
Equivalent of pemo_internal.m
"""
import numpy as np
import scipy.signal as signal

from gammatone_helper import Gfb_Analyzer_new
from gammatone_helper import Gfb_Analyzer_process


def pemo_internal(x: np.ndarray, fs: float, modproc: str = 'lp'):
    # %%% MATLAB Code %%%
    if len(x.shape) > 1:
        if x.shape[0] < x.shape[1]:
            x = x.T
        x = x.ravel()
    nsampl = x.shape[0]

    # %%% Scaling %%%
    x = 10.0 * x

    # %%% Basilar membrane filtering %%%
    fmin = 235
    fmax = min(0.5 * fs, 14500)
    if fs < 3 * fmax:
        new_fs = int(round(1.5 * fs))
        x = signal.resample(x, int(round(len(x) * new_fs / fs)))
        fs = new_fs

    nsampl = len(x)
    analyzer = Gfb_Analyzer_new(fs, fmin, 1000, fmax, 1.0)
    nband = analyzer['center_frequencies_hz'].shape[0]

    # rx=real(Gfb_Analyzer_process(analyzer,x));
    gfb_out, _ = Gfb_Analyzer_process(analyzer, x)
    rx = np.real(gfb_out)

    # %%% Envelope extraction %%%
    # gain=exp(-pi*2000/fs); rx=filter(1-gain,[1 -gain],max(rx,0),[],2);
    gain_hc = np.exp(-np.pi * 2000.0 / fs)
    b_hc = np.array([1.0 - gain_hc])
    a_hc = np.array([1.0, -gain_hc])
    rx = signal.lfilter(b_hc, a_hc, np.maximum(rx, 0.0), axis=1)

    # % Frequency-independent absolute hearing threshold and adaptation loops
    dbrange = 100.0
    thresh = 10.0 ** (-dbrange / 20.0)
    bw_loop = 1.0 / (np.pi * np.array([0.005, 0.05, 0.129, 0.253, 0.5]))
    rx = np.maximum(rx, thresh)

    # Vectorized loop processing natively recreating adapt.c execution inside python
    sthresh = thresh
    for s in range(5):
        gain_val = np.exp(-np.pi * bw_loop[s] / fs)
        sthresh = np.sqrt(sthresh)
        factor = np.full(nband, sthresh)
        for t in range(nsampl):
            val = rx[:, t] / factor
            rx[:, t] = val
            factor = np.maximum((1.0 - gain_val) * val + gain_val * factor, sthresh)

    # rx=double(dbrange/(1-sthresh))*(double(rx)-double(sthresh));
    rx = (dbrange / (1.0 - sthresh)) * (rx - sthresh)

    # %%% Modulation filtering %%%
    if modproc == 'fb':
        rx = signal.resample(rx, int(round(rx.shape[1] * 800.0 / fs)), axis=1)
        fs = 800
        fc = np.concatenate(([0.0, 5.0], 10.0 * (5.0 / 3.0) ** np.arange(6)))
        bw_mod = np.concatenate(([5.0, 5.0], 5.0 * (5.0 / 3.0) ** np.arange(6)))
    else:
        rx = signal.resample(rx, int(round(rx.shape[1] * 100.0 / fs)), axis=1)
        fs = 100
        fc = np.array([0.0])
        bw_mod = np.array([15.92])

    nmod = len(fc)
    nsampl = rx.shape[1]
    mx = np.zeros((nband, nsampl, nmod), dtype=complex)

    for m in range(nmod):
        gain_val = np.exp(-np.pi * bw_mod[m] / fs)
        b_mod = np.array([1.0 - gain_val])
        a_mod = np.array([1.0, -gain_val * np.exp(2j * np.pi * fc[m] / fs)])
        mx[:, :, m] = signal.lfilter(b_mod, a_mod, rx, axis=1)

    above = (fc > 10.0)
    mx[:, :, ~above] = np.real(mx[:, :, ~above])
    mx[:, :, above] = np.abs(mx[:, :, above])

    return mx, fs
