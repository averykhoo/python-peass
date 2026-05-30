"""
PEASS Metrics Package - Auditory Features & Similarity Metrics [1, 2]

This module computes the perceptual features and linear/energy ratio calculations
such as SDR, ISR, SIR, and SAR [1]. It houses the core PEMO-Q time-frequency
cross-correlation engine.
"""

from typing import Tuple

import numpy as np

from .auditory_model import generate_internal_representation


def calculate_energy_ratios(
        s_true: np.ndarray,
        e_target: np.ndarray,
        e_interf: np.ndarray,
        e_artif: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Computes standard BSS Eval energy ratio metrics from physically decomposed components.
    Replaces ISR_SIR_SAR_fromNewDecomposition.m [1].
    """
    sTrue_flat = s_true.ravel()
    eTarget_flat = e_target.ravel()
    eInterf_flat = e_interf.ravel()
    eArtif_flat = e_artif.ravel()

    # Eq. (11), (12), (13) of Emiya 2011 [4]:
    ISR = 10.0 * np.log10(np.sum(sTrue_flat ** 2) / np.sum(eTarget_flat ** 2))
    SIR = 10.0 * np.log10(np.sum((sTrue_flat + eTarget_flat) ** 2) / np.sum(eInterf_flat ** 2))
    SAR = 10.0 * np.log10(np.sum((sTrue_flat + eTarget_flat + eInterf_flat) ** 2) / np.sum(eArtif_flat ** 2))
    SDR = 10.0 * np.log10(np.sum(sTrue_flat ** 2) / np.sum((eTarget_flat + eInterf_flat + eArtif_flat) ** 2))

    return ISR, SIR, SAR, SDR


def pemo_similarity_metric(internal_reference: np.ndarray, internal_test: np.ndarray,
                           sampling_frequency: float) -> float:
    """
    Compares two internal representations to produce an auditory similarity metric.
    Replaces pemo_metric.m [1].

    Performs assimilation of masked content, local framing, cross-correlation,
    moving RMS weighting, and percentile assessment [2].
    """
    nband, nsampl, nmod = internal_reference.shape

    # Assimilation (Eq. of PEMO-Q [2]):
    assim = (np.abs(internal_test) < np.abs(internal_reference))
    internal_test[assim] = 0.25 * internal_reference[assim] + 0.75 * internal_test[assim]

    # Convert frame sizes
    flen = int(min(nsampl, 0.1 * sampling_frequency))
    nfram = int(np.floor(nsampl / flen))
    nsampl = nfram * flen

    internal_reference = internal_reference[:, :nsampl, :]
    internal_test = internal_test[:, :nsampl, :]

    PSMt = np.zeros(nfram)
    lPSM = np.zeros(nmod)
    lNMS = np.zeros(nmod)

    for t in range(nfram):
        for m in range(nmod):
            lref = internal_reference[:, t * flen: (t + 1) * flen, m]
            lref_flat = lref.ravel()
            lref_flat = lref_flat - np.mean(lref_flat)

            ltest = internal_test[:, t * flen: (t + 1) * flen, m]
            ltest_flat_orig = ltest.ravel()
            lNMS[m] = np.sum(ltest_flat_orig ** 2)

            ltest_flat = ltest_flat_orig - np.mean(ltest_flat_orig)
            denom = np.sqrt(np.sum(lref_flat ** 2) * np.sum(ltest_flat ** 2))
            lPSM[m] = np.sum(lref_flat * ltest_flat) / denom if denom != 0 else 0.0

        sum_lnms = np.sum(lNMS)
        PSMt[t] = np.sum(lPSM * lNMS) / sum_lnms if sum_lnms != 0 else 0.0

    # From local to global similarity
    ilen = int(1 * sampling_frequency)
    mtest_sq = np.sum(internal_test ** 2, axis=(0, 2))

    RMS = np.zeros(nfram)
    for t in range(nfram):
        start_idx = int(max(0, (t + 0.5) * flen - 0.5 * ilen))
        end_idx = int(min(nsampl, (t + 0.5) * flen + 0.5 * ilen))
        ltest = mtest_sq[start_idx:end_idx]
        RMS[t] = np.mean(ltest) if len(ltest) > 0 else 0.0

    # Sorted weighted percentile extraction
    ind = np.argsort(PSMt)
    PSMt_sorted = PSMt[ind]
    RMS_sorted = RMS[ind]
    RMS_cum = np.cumsum(RMS_sorted)

    cutoff = 0.5 * RMS_cum[-1]
    match_indices = np.where(RMS_cum >= cutoff)[0]

    return PSMt_sorted[match_indices[0]] if len(match_indices) > 0 else 0.0


def audio_quality_features(decomposition_signals: list[np.ndarray], sampling_frequency: float = 16000.0) -> Tuple[
    float, float, float, float]:
    """
    Computes quality features by sending decomposed signals through the internal auditory model.
    Replaces audioQualityFeatures.m [1].
    """
    sTrue, eTarget, eInterf, eArtif = decomposition_signals

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
        mtest, fr = generate_internal_representation(testAll[:, kChan], sampling_frequency)

        mref_t, _ = generate_internal_representation(sTrue[:, kChan] + eInterf[:, kChan] + eArtif[:, kChan],
                                                     sampling_frequency)
        qTarget[kChan] = pemo_similarity_metric(mref_t, mtest, fr)

        mref_i, _ = generate_internal_representation(sTrue[:, kChan] + eTarget[:, kChan] + eArtif[:, kChan],
                                                     sampling_frequency)
        qInterf[kChan] = pemo_similarity_metric(mref_i, mtest, fr)

        mref_a, _ = generate_internal_representation(sTrue[:, kChan] + eTarget[:, kChan] + eInterf[:, kChan],
                                                     sampling_frequency)
        qArtif[kChan] = pemo_similarity_metric(mref_a, mtest, fr)

        mref_g, _ = generate_internal_representation(sTrue[:, kChan], sampling_frequency)
        qGlobal[kChan] = pemo_similarity_metric(mref_g, mtest, fr)

    return np.min(qTarget), np.min(qInterf), np.min(qArtif), np.min(qGlobal)
