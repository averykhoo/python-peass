# %%% audioQualityFeatures.m %%%
import numpy as np
import soundfile as sf
from pemo_internal import pemo_internal
from pemo_metric import pemo_metric


# function [qTarget, qInterf, qArtif, qGlobal] = audioQualityFeatures(decompositionFilenames)
def audioQualityFeatures(decompositionFilenames):
    # sTrue = audioread(decompositionFilenames{1});
    # eTarget = audioread(decompositionFilenames{2});
    # eInterf = audioread(decompositionFilenames{3});
    # eArtif = audioread(decompositionFilenames{4});
    if isinstance(decompositionFilenames[0], str):
        sTrue, fs = sf.read(decompositionFilenames[0])
        eTarget, _ = sf.read(decompositionFilenames[1])
        eInterf, _ = sf.read(decompositionFilenames[2])
        eArtif, _ = sf.read(decompositionFilenames[3])
    else:
        sTrue, eTarget, eInterf, eArtif = decompositionFilenames[0], decompositionFilenames[1], decompositionFilenames[
            2], decompositionFilenames[3]
        # In-memory evaluation assumes sampling rate is mapped or default to 16000
        fs = 16000

    if len(sTrue.shape) == 1:
        sTrue = sTrue[:, np.newaxis]
        eTarget = eTarget[:, np.newaxis]
        eInterf = eInterf[:, np.newaxis]
        eArtif = eArtif[:, np.newaxis]

    # testAll=sTrue+eTarget+eInterf+eArtif;
    testAll = sTrue + eTarget + eInterf + eArtif

    # NChan = size(sTrue,2);
    NChan = sTrue.shape[1]
    # qTarget = NaN(NChan,1);
    # qInterf = NaN(NChan,1);
    # qArtif = NaN(NChan,1);
    # qGlobal = NaN(NChan,1);
    qTarget = np.zeros(NChan)
    qInterf = np.zeros(NChan)
    qArtif = np.zeros(NChan)
    qGlobal = np.zeros(NChan)

    # for kChan = 1:NChan
    for kChan in range(NChan):
        # [mtest,fr] = pemo_internal(testAll(:,kChan),fs);
        mtest, fr = pemo_internal(testAll[:, kChan], fs)

        # mref = pemo_internal(sTrue(:,kChan)+eInterf(:,kChan)+eArtif(:,kChan),fs);
        mref_t, _ = pemo_internal(sTrue[:, kChan] + eInterf[:, kChan] + eArtif[:, kChan], fs)
        # qTarget(kChan) = pemo_metric(mref,mtest,fr);
        qTarget[kChan] = pemo_metric(mref_t, mtest, fr)

        # mref=pemo_internal(sTrue(:,kChan)+eTarget(:,kChan)+eArtif(:,kChan),fs);
        mref_i, _ = pemo_internal(sTrue[:, kChan] + eTarget[:, kChan] + eArtif[:, kChan], fs)
        # qInterf(kChan) = pemo_metric(mref,mtest,fr);
        qInterf[kChan] = pemo_metric(mref_i, mtest, fr)

        # mref=pemo_internal(sTrue(:,kChan)+eTarget(:,kChan)+eInterf(:,kChan),fs);
        mref_a, _ = pemo_internal(sTrue[:, kChan] + eTarget[:, kChan] + eInterf[:, kChan], fs)
        # qArtif(kChan) = pemo_metric(mref,mtest,fr);
        qArtif[kChan] = pemo_metric(mref_a, mtest, fr)

        # mref=pemo_internal(sTrue(:,kChan),fs);
        mref_g, _ = pemo_internal(sTrue[:, kChan], fs)
        # qGlobal(kChan) = pemo_metric(mref,mtest,fr);
        qGlobal[kChan] = pemo_metric(mref_g, mtest, fr)

    # qTarget = min(qTarget);
    qTarget = np.min(qTarget)
    # qInterf = min(qInterf);
    qInterf = np.min(qInterf)
    # qArtif = min(qArtif);
    qArtif = np.min(qArtif)
    # qGlobal = min(qGlobal);
    qGlobal = np.min(qGlobal)

    # return
    return qTarget, qInterf, qArtif, qGlobal