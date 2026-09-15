"""Per-band decomposition of an estimate into target / spatial / interference / artifact.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "extractTSIA.m"

import numpy as np

from .LSDecompose_tv import LSDecompose_tv

# >>> MATLAB
# function [sTrue,eSpat,eInterf,eArtif] =...
#     extractTSIA(s,sEst,flen,Lw,hop,options)
# % decompose each multichannel estimate sEst(:,:,nEst) into TSIA
# % using original sources s
# % where s(:,:,1) is the target source and
# % s(:,:,j), j>1, are the interfering sources
#
# <<< MATLAB


def extractTSIA(s, sEst, flen, Lw, hop, options):
# >>> MATLAB
# [L,NChan,NSources] = size(s);
# NEst = size(sEst,3);
#
# s = reshape(s,[L,NSources*NChan]);
# sEst = reshape(sEst,[L,NEst*NChan]);
#
# <<< MATLAB
    L, NChan, NSources = s.shape
    NEst = sEst.shape[2]  # MATLAB size(sEst,3)

    # MATLAB's reshape is COLUMN-MAJOR, so column (nSource-1)*NChan + nChan of
    # the flattened array holds channel nChan of source nSource -- i.e. channel
    # varies fastest. `order='F'` is what preserves that; the default C order
    # would silently transpose the source/channel grouping and every
    # `(nSource-1)*NChan+(1:NChan)` slice below would then address the wrong
    # columns.
    s = np.reshape(s, (L, NSources * NChan), order='F')
    sEst = np.reshape(sEst, (L, NEst * NChan), order='F')

# >>> MATLAB
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %   Projection on all sources/channels      %
# yProjAll_ = LSDecompose_tv(...
#     sEst,s,flen,Lw,hop);
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# <<< MATLAB
    yProjAll_ = LSDecompose_tv(
        sEst, s, flen, Lw, hop)

# >>> MATLAB
# % merge projections on channels and remove last samples (filter length)
# yProjAll = zeros([L,NChan,NSources]);
# for nSource = 1:NSources
#     yProjAll(:,:,nSource) = ...
#         sum(yProjAll_(1:end-flen+1,:,(nSource-1)*NChan+(1:NChan)),3);
# end
#
# <<< MATLAB
    yProjAll = np.zeros((L, NChan, NSources), dtype=yProjAll_.dtype)
    for nSource in range(1, NSources + 1):  # MATLAB nSource=1:NSources
        # MATLAB yProjAll_(1:end-flen+1,:,(nSource-1)*NChan+(1:NChan)):
        # keep the first L rows (drop the filter tail) and sum the projection
        # over that source's NChan channels.
        rows = slice(0, yProjAll_.shape[0] - flen + 1)
        cols = slice((nSource - 1) * NChan, (nSource - 1) * NChan + NChan)
        yProjAll[:, :, nSource - 1] = np.sum(yProjAll_[rows, :, cols], axis=2)

# >>> MATLAB
# if options.FLAG_2PROJ
#     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#     %   Projection on target sources/channels   %
#     eSpat = zeros(size(sEst));
#     for nEst=1:NEst
#         % project each estimate onto the target (first NChan channels in s)
#         eSpat_ = LSDecompose_tv(...
#             sEst(:,(nEst-1)*NChan+(1:NChan)),...
#             s(:,1:NChan),flen,Lw,hop);
#         % merge projections on channels and remove last samples (filter length)
#         eSpat(:,:) = ...
#             sum(eSpat_(1:end-flen+1,:,:),3);
#     end
#     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# end
#
# <<< MATLAB
    if options['FLAG_2PROJ']:
        eSpat = np.zeros(sEst.shape, dtype=yProjAll_.dtype)
        for nEst in range(1, NEst + 1):  # MATLAB nEst=1:NEst
            eSpat_ = LSDecompose_tv(
                sEst[:, (nEst - 1) * NChan:(nEst - 1) * NChan + NChan],
                s[:, 0:NChan], flen, Lw, hop)
            # MATLAB writes `eSpat(:,:)`, i.e. the WHOLE array, on every
            # iteration -- so this loop only produces a correct result for
            # NEst==1 (which is all the PEASS pipeline ever uses). Transcribed
            # as written rather than repaired.
            eSpat[:, :] = np.sum(eSpat_[0:eSpat_.shape[0] - flen + 1, :, :], axis=2)
        eSpat = np.reshape(eSpat, (L, NChan, NEst), order='F')

# >>> MATLAB
# % Build distortion components
# sTrue = zeros([L,NChan,NEst]);
# for nEst=1:NEst
#     sTrue(:,:,nEst) = s(:,1:NChan);
# end
# if options.FLAG_2PROJ
#     eSpat = eSpat-sTrue;
# else
#     eSpat = yProjAll(:,:,1:NEst)-sTrue;
# end
#
# <<< MATLAB
    sTrue = np.zeros((L, NChan, NEst), dtype=s.dtype)
    for nEst in range(1, NEst + 1):  # MATLAB nEst=1:NEst
        sTrue[:, :, nEst - 1] = s[:, 0:NChan]  # MATLAB s(:,1:NChan)
    if options['FLAG_2PROJ']:
        eSpat = eSpat - sTrue
    else:
        eSpat = yProjAll[:, :, 0:NEst] - sTrue

# >>> MATLAB
# eInterf = sum(yProjAll,3) - eSpat - sTrue;
#
# eArtif = reshape(sEst,[L,NChan,NEst]) - sTrue - eSpat - eInterf;
#
# return
# <<< MATLAB
    # MATLAB's sum(yProjAll,3) drops to (L,NChan) and relies on NEst==1 for the
    # subtraction to line up. keepdims makes the same thing explicit and
    # broadcast-safe.
    eInterf = np.sum(yProjAll, axis=2, keepdims=True) - eSpat - sTrue

    eArtif = np.reshape(sEst, (L, NChan, NEst), order='F') - sTrue - eSpat - eInterf

    return sTrue, eSpat, eInterf, eArtif
