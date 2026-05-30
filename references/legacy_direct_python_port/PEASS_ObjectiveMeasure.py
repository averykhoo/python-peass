"""
PEASS Toolkit - Python Port
Equivalent of PEASS_ObjectiveMeasure.m
Main wrapper entry point for predicting Perceptual Audio Quality
"""
from ISR_SIR_SAR_fromNewDecomposition import ISR_SIR_SAR_fromNewDecomposition
from audioQualityFeatures import audioQualityFeatures
from extractDistortionComponents import extractDistortionComponents
from map2SubjScale import map2SubjScale


def PEASS_ObjectiveMeasure(originalFiles: list, estimateFile, options: dict = None, fs: float = None) -> dict:
    # %%% MATLAB Code %%%
    res = {}

    # % Decompose the distortion into specific components
    res['decompositionFilenames'] = extractDistortionComponents(originalFiles, estimateFile, options, fs)

    # % Compute ISR, SIR, SAR, SDR from the estimated components
    ISR, SIR, SAR, SDR = ISR_SIR_SAR_fromNewDecomposition(res['decompositionFilenames'])
    res['ISR'] = ISR
    res['SIR'] = SIR
    res['SAR'] = SAR
    res['SDR'] = SDR

    # % Compute quality features using PEMO-Q
    qTarget, qInterf, qArtif, qGlobal = audioQualityFeatures(res['decompositionFilenames'])
    res['qTarget'] = qTarget
    res['qInterf'] = qInterf
    res['qArtif'] = qArtif
    res['qGlobal'] = qGlobal

    # % Non-linear mapping to subjective scale
    OPS, TPS, IPS, APS = map2SubjScale(qTarget, qInterf, qArtif, qGlobal)
    res['OPS'] = OPS
    res['TPS'] = TPS
    res['IPS'] = IPS
    res['APS'] = APS

    return res
