# %%% PEASS_ObjectiveMeasure.m (Complete 100% Version) %%%
from extractDistortionComponents import extractDistortionComponents
from ISR_SIR_SAR_fromNewDecomposition import ISR_SIR_SAR_fromNewDecomposition
from audioQualityFeatures import audioQualityFeatures
from map2SubjScale import map2SubjScale


# function res = PEASS_ObjectiveMeasure(originalFiles,estimateFile,options)
def PEASS_ObjectiveMeasure(originalFiles, estimateFile, options=None, fs=None):
    res = {}

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Decompose the distortion into specific components
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # res.decompositionFilenames = extractDistortionComponents(originalFiles,estimateFile,options);
    res['decompositionFilenames'] = extractDistortionComponents(originalFiles, estimateFile, options, fs)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Compute ISR, SIR, SAR, SDR from the estimated components (optional)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # [res.ISR, res.SIR, res.SAR, res.SDR] = ISR_SIR_SAR_fromNewDecomposition(res.decompositionFilenames);
    ISR, SIR, SAR, SDR = ISR_SIR_SAR_fromNewDecomposition(res['decompositionFilenames'])
    res['ISR'] = ISR
    res['SIR'] = SIR
    res['SAR'] = SAR
    res['SDR'] = SDR

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Compute quality features using PEMO-Q
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # [res.qTarget, res.qInterf, res.qArtif, res.qGlobal] = audioQualityFeatures(res.decompositionFilenames);
    qTarget, qInterf, qArtif, qGlobal = audioQualityFeatures(res['decompositionFilenames'])
    res['qTarget'] = qTarget
    res['qInterf'] = qInterf
    res['qArtif'] = qArtif
    res['qGlobal'] = qGlobal

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Non-linear mapping to subjective scale
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # [res.OPS, res.TPS,res.IPS,res.APS] = map2SubjScale(res.qTarget, res.qInterf, res.qArtif, res.qGlobal);
    OPS, TPS, IPS, APS = map2SubjScale(qTarget, qInterf, qArtif, qGlobal)
    res['OPS'] = OPS
    res['TPS'] = TPS
    res['IPS'] = IPS
    res['APS'] = APS

    # return
    return res