# %%% extractTSIA.m %%%
import numpy as np
from LSDecompose_tv import LSDecompose_tv


# function [sTrue,eSpat,eInterf,eArtif] = extractTSIA(s,sEst,flen,Lw,hop,options)
def extractTSIA(s, sEst, flen, Lw, hop, options):
    # [L,NChan,NSources] = size(s);
    L, NChan, NSources = s.shape
    # NEst = size(sEst,3);
    NEst = sEst.shape[2] if len(sEst.shape) > 2 else 1
    if len(sEst.shape) == 2:
        sEst = sEst[:, :, np.newaxis]

    # s = reshape(s,[L,NSources*NChan]);
    s_reshaped = s.reshape((L, NSources * NChan), order='F')
    # sEst = reshape(sEst,[L,NEst*NChan]);
    sEst_reshaped = sEst.reshape((L, NEst * NChan), order='F')

    # %   Projection on all sources/channels
    # yProjAll_ = LSDecompose_tv(sEst,s,flen,Lw,hop);
    yProjAll_ = LSDecompose_tv(sEst_reshaped, s_reshaped, flen, Lw, hop)

    # % merge projections on channels and remove last samples (filter length)
    # yProjAll = zeros([L,NChan,NSources]);
    yProjAll = np.zeros((L, NChan * NEst, NSources), dtype=s.dtype)
    # for nSource = 1:NSources
    for nSource in range(NSources):
        # yProjAll(:,:,nSource) = sum(yProjAll_(1:end-flen+1,:,(nSource-1)*NChan+(1:NChan)),3);
        start_idx = nSource * NChan
        end_idx = (nSource + 1) * NChan
        sliced = yProjAll_[:L, :, start_idx:end_idx]
        yProjAll[:, :, nSource] = np.sum(sliced, axis=2)

    # if options.FLAG_2PROJ
    flag_2proj = options.get('FLAG_2PROJ', False) if isinstance(options, dict) else options.FLAG_2PROJ
    # eSpat = zeros(size(sEst));
    eSpat = np.zeros((L, NEst * NChan), dtype=sEst.dtype)
    if flag_2proj:
        # for nEst=1:NEst
        for nEst in range(NEst):
            # eSpat_ = LSDecompose_tv(sEst(:,(nEst-1)*NChan+(1:NChan)), s(:,1:NChan),flen,Lw,hop);
            start_est = nEst * NChan
            end_est = (nEst + 1) * NChan
            eSpat_ = LSDecompose_tv(
                sEst_reshaped[:, start_est:end_est],
                s_reshaped[:, :NChan],
                flen, Lw, hop
            )
            # eSpat(:,:) = sum(eSpat_(1:end-flen+1,:,:),3);
            eSpat[:, start_est:end_est] = np.sum(eSpat_[:L, :, :], axis=2)

    # % Build distortion components
    # sTrue = zeros([L,NChan,NEst]);
    sTrue = np.zeros((L, NChan * NEst), dtype=s.dtype)
    # for nEst=1:NEst
    for nEst in range(NEst):
        # sTrue(:,:,nEst) = s(:,1:NChan);
        start_est = nEst * NChan
        end_est = (nEst + 1) * NChan
        sTrue[:, start_est:end_est] = s_reshaped[:, :NChan]

    # if options.FLAG_2PROJ
    if flag_2proj:
        # eSpat = eSpat-sTrue;
        eSpat = eSpat - sTrue
    # else
    else:
        # eSpat = yProjAll(:,:,1:NEst)-sTrue;
        eSpat = yProjAll[:, :, :NEst].reshape((L, NEst * NChan), order='F') - sTrue

    # eInterf = sum(yProjAll,3) - eSpat - sTrue;
    yProjAll_summed = np.sum(yProjAll, axis=2)
    eInterf = yProjAll_summed - eSpat - sTrue
    # eArtif = reshape(sEst,[L,NChan,NEst]) - sTrue - eSpat - eInterf;
    eArtif = sEst_reshaped - sTrue - eSpat - eInterf

    # Reconvert back to 3D arrays to match MATLAB's return format
    sTrue_3d = sTrue.reshape((L, NChan, NEst), order='F')
    eSpat_3d = eSpat.reshape((L, NChan, NEst), order='F')
    eInterf_3d = eInterf.reshape((L, NChan, NEst), order='F')
    eArtif_3d = eArtif.reshape((L, NChan, NEst), order='F')

    return sTrue_3d, eSpat_3d, eInterf_3d, eArtif_3d