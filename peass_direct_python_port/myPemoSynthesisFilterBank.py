"""
PEASS Toolkit - Python Port
Equivalent of myPemoSynthesisFilterBank.m
"""
import numpy as np
import scipy.signal as signal

from gammatone_helper import Gfb_Synthesizer_new
from gammatone_helper import Gfb_Synthesizer_process


def myPemoSynthesisFilterBank(xFB: list, analyzer: dict, Mmod: np.ndarray = None):
    # %%% MATLAB Code %%%
    # Nb = size(xFB,1); fs = analyzer.sampling_frequency_hz;
    Nb = len(xFB)
    fs = analyzer['fs']

    # max_len = max(cellfun('length',xFB(:)).*analyzer.Ndec(:))
    max_len = max(len(xFB[k]) * analyzer['Ndec'][k] for k in range(Nb))
    gfb_out_proc = np.zeros((Nb, max_len), dtype=complex)

    # for k=1:Nb
    for k in range(Nb):
        # gfb_out_proc(k,1:length(xFB{k})*analyzer.Ndec(k)) = resample(xFB{k},analyzer.Ndec(k),1);
        target_len = len(xFB[k]) * analyzer['Ndec'][k]
        upsampled = signal.resample_poly(xFB[k], analyzer['Ndec'][k], 1)

        # Enforce exact dimension limits to avoid broadcast errors from transient lengths
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
