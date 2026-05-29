"""
PEASS Toolkit - Python Port
Equivalent of audioQualityFeatures.m
"""
import numpy as np
import soundfile as sf

from pemo_internal import pemo_internal
from pemo_metric import pemo_metric


def audioQualityFeatures(decompositionFilenames: list):
    # %%% MATLAB Code %%%
    if isinstance(decompositionFilenames[0], str):
        sTrue, fs = sf.read(decompositionFilenames[0])
        eTarget, _ = sf.read(decompositionFilenames[1])
        eInterf, _ = sf.read(decompositionFilenames[2])
        eArtif, _ = sf.read(decompositionFilenames[3])
    else:
        sTrue, eTarget, eInterf, eArtif = decompositionFilenames
        fs = 16000

    if len(sTrue.shape) == 1:
        sTrue = sTrue[:, np.newaxis]
        eTarget = eTarget[:, np.newaxis]
        eInterf = eInterf[:, np.newaxis]
        eArtif = eArtif[:, np.newaxis]

    testAll = sTrue + eTarget + eInterf + eArtif
    NChan = sTrue.shape[1]

    qTarget = np.zeros(NChan)
    qInterf = np.zeros(NChan)
    qArtif = np.zeros(NChan)
    qGlobal = np.zeros(NChan)

    for kChan in range(NChan):
        mtest, fr = pemo_internal(testAll[:, kChan], fs)

        mref_t, _ = pemo_internal(sTrue[:, kChan] + eInterf[:, kChan] + eArtif[:, kChan], fs)
        qTarget[kChan] = pemo_metric(mref_t, mtest, fr)

        mref_i, _ = pemo_internal(sTrue[:, kChan] + eTarget[:, kChan] + eArtif[:, kChan], fs)
        qInterf[kChan] = pemo_metric(mref_i, mtest, fr)

        mref_a, _ = pemo_internal(sTrue[:, kChan] + eTarget[:, kChan] + eInterf[:, kChan], fs)
        qArtif[kChan] = pemo_metric(mref_a, mtest, fr)

        mref_g, _ = pemo_internal(sTrue[:, kChan], fs)
        qGlobal[kChan] = pemo_metric(mref_g, mtest, fr)

    # Select the minimum value across all channels
    return np.min(qTarget), np.min(qInterf), np.min(qArtif), np.min(qGlobal)
