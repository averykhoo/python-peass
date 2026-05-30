"""
PEASS Toolkit - Python Port
Equivalent of ISR_SIR_SAR_fromNewDecomposition.m
"""
import numpy as np
import soundfile as sf


def ISR_SIR_SAR_fromNewDecomposition(decompositionFilenames):
    # %%% MATLAB Code %%%
    # sTrue = audioread(decompositionFilenames{1});
    # eTarget = audioread(decompositionFilenames{2});
    # eInterf = audioread(decompositionFilenames{3});
    # eArtif = audioread(decompositionFilenames{4});
    if isinstance(decompositionFilenames[0], str):
        sTrue = sf.read(decompositionFilenames[0])[0]
        eTarget = sf.read(decompositionFilenames[1])[0]
        eInterf = sf.read(decompositionFilenames[2])[0]
        eArtif = sf.read(decompositionFilenames[3])[0]
    else:
        sTrue, eTarget, eInterf, eArtif = decompositionFilenames

    sTrue_flat = sTrue.ravel()
    eTarget_flat = eTarget.ravel()
    eInterf_flat = eInterf.ravel()
    eArtif_flat = eArtif.ravel()

    # ISR = 10*log10(sum(sTrue(:).^2)/sum(eTarget(:).^2));
    ISR = 10.0 * np.log10(np.sum(sTrue_flat ** 2) / np.sum(eTarget_flat ** 2))
    # SIR = 10*log10(sum((sTrue(:)+eTarget(:)).^2)/sum(eInterf(:).^2));
    SIR = 10.0 * np.log10(np.sum((sTrue_flat + eTarget_flat) ** 2) / np.sum(eInterf_flat ** 2))
    # SAR = 10*log10(sum((sTrue(:)+eTarget(:)+eInterf(:)).^2)/sum(eArtif(:).^2));
    SAR = 10.0 * np.log10(np.sum((sTrue_flat + eTarget_flat + eInterf_flat) ** 2) / np.sum(eArtif_flat ** 2))

    # SDR = 10*log10(sum(sTrue(:).^2)/sum((eTarget(:)+eInterf(:)+eArtif(:)).^2));
    SDR = 10.0 * np.log10(np.sum(sTrue_flat ** 2) / np.sum((eTarget_flat + eInterf_flat + eArtif_flat) ** 2))

    return ISR, SIR, SAR, SDR
