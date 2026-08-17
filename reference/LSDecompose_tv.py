"""Time-varying (framed, overlap-added) least-squares projection.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "LSDecompose_tv.m"

import numpy as np

from ._matlab_runtime import hann_periodic
from ._matlab_runtime import matlab_round
from .LSDecompose import LSDecompose

# >>> MATLAB
# function sproj = LSDecompose_tv(se,s,flen,Lw,hop)
# % SPROJ Least-squares projection of each channel of se on the subspace
# % spanned by delayed versions of the channels of s, with delays between
# % -flen2 and +flen2 where flen = 2*flen2+1
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Version 1.0
# % Copyright 2010 Valentin Emiya (INRIA).
# % This software is distributed under the terms of the GNU Public License
# % version 3 (http://www.gnu.org/licenses/gpl.txt).
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# <<< MATLAB


def LSDecompose_tv(se, s, flen, Lw, hop):
# >>> MATLAB
# flen2 = (flen-1)/2;
# if flen2~=round(flen2)
#     error('filterParam:NotOdd','Not an odd order');
# end
#
# <<< MATLAB
    flen2 = (flen - 1) / 2
    if flen2 != matlab_round(flen2):
        raise ValueError("filterParam:NotOdd Not an odd order")
    flen2 = int(flen2)

# >>> MATLAB
# s (end+1:end+flen-1+Lw-1,:) = 0;
# se(end+1:end+flen-1+Lw-1,:) = 0;
# [nsampl,nsrc]=size(s);
# nchanEst = size(se,2);
# <<< MATLAB
    # MATLAB grows the array in place; it is copy-on-write, so the caller's
    # array is untouched. `np.vstack` on a fresh pad reproduces that.
    npad = flen - 1 + Lw - 1
    s = np.vstack([s, np.zeros((npad, s.shape[1]), dtype=s.dtype)])
    se = np.vstack([se, np.zeros((npad, se.shape[1]), dtype=se.dtype)])
    nsampl, nsrc = s.shape
    nchanEst = se.shape[1]  # MATLAB size(se,2)

    dtype = np.result_type(se.dtype, s.dtype)

# >>> MATLAB
#
# % sine analysis window / sine synthesis window
# fahandle = @hann;
# fshandle = @hann;
# wa = sqrt(flipud(window(fahandle,Lw,'periodic')));
# ws = sqrt(flipud(window(fshandle,Lw,'periodic')));
# <<< MATLAB
    # `window(@hann,Lw,'periodic')` == `hann(Lw,'periodic')`. The window is
    # flipped before the square root, so wa[0] is the LAST hann sample and
    # wa[-1] is hann's leading zero.
    wa = np.sqrt(np.flipud(hann_periodic(Lw)))
    ws = np.sqrt(np.flipud(hann_periodic(Lw)))

# >>> MATLAB
#
# WS = zeros(Lw,nchanEst,nsrc);
# for chan = 1:nchanEst
#     for j=1:nsrc
#         WS(:,chan,j) = ws;
#     end
# end
# wBegin = 1;
# wEnd = wBegin+Lw-1;
# sproj = zeros(nsampl,nchanEst,nsrc);
# wAccum = zeros(nsampl,1);
# Ns = size(s,2);
# Ls = size(s,1);
# <<< MATLAB
    WS = np.zeros((Lw, nchanEst, nsrc))
    for chan in range(1, nchanEst + 1):  # MATLAB chan=1:nchanEst
        for j in range(1, nsrc + 1):  # MATLAB j=1:nsrc
            WS[:, chan - 1, j - 1] = ws
    # wBegin/wEnd are kept 1-BASED, exactly as MATLAB writes them, and every
    # slice below subtracts 1 from the start only. Renumbering them from 0
    # would silently change the `flen2-wBegin+1` zero-pad count on the first
    # frame, which is the whole point of the boundary handling.
    wBegin = 1
    wEnd = wBegin + Lw - 1
    sproj = np.zeros((nsampl, nchanEst, nsrc), dtype=dtype)
    wAccum = np.zeros(nsampl)  # MATLAB zeros(nsampl,1)
    Ns = s.shape[1]  # MATLAB size(s,2)
    Ls = s.shape[0]  # MATLAB size(s,1)

# >>> MATLAB
#
# while wEnd-Lw/2<=size(sproj,1)-Lw+1
#     % get frames
#     sew = se(wBegin:wEnd,:);
#     sw = [zeros(max(0,flen2-wBegin+1),Ns);...
#         s(max(1,wBegin-flen2):min(end,wEnd+flen2),:);...
#         zeros(max(0,wEnd+flen2-Ls),Ns)];
#     
#     % projection
#     sprojw=LSDecompose(sew,sw,flen2,wa);
#     
#     % overlap add
#     sproj(wBegin:wEnd,:,:) = sproj(wBegin:wEnd,:,:)...
#         + sprojw(1:Lw,:,:).*WS;
#     
#     wAccum(wBegin:wEnd,1) = wAccum(wBegin:wEnd,1)+ws.*wa;
#     
#     % update for next iteration
#     wBegin = wBegin+hop;
#     wEnd = wEnd+hop;
# end
# <<< MATLAB
    while wEnd - Lw / 2 <= sproj.shape[0] - Lw + 1:
        # MATLAB se(wBegin:wEnd,:)
        sew = se[wBegin - 1:wEnd, :]
        # MATLAB s(max(1,wBegin-flen2):min(end,wEnd+flen2),:), where `end` is Ls.
        # The head/tail zero blocks make `sw` exactly Lw+flen-1 long whatever the
        # frame position, which is what LSDecompose's Toeplitz build assumes.
        lo = max(1, wBegin - flen2)
        hi = min(Ls, wEnd + flen2)
        sw = np.vstack([
            np.zeros((max(0, flen2 - wBegin + 1), Ns), dtype=s.dtype),
            s[lo - 1:hi, :],
            np.zeros((max(0, wEnd + flen2 - Ls), Ns), dtype=s.dtype),
        ])

        sprojw = LSDecompose(sew, sw, flen2, wa)

        # MATLAB sproj(wBegin:wEnd,:,:) += sprojw(1:Lw,:,:).*WS
        sproj[wBegin - 1:wEnd, :, :] = (sproj[wBegin - 1:wEnd, :, :]
                                        + sprojw[:Lw, :, :] * WS)

        wAccum[wBegin - 1:wEnd] = wAccum[wBegin - 1:wEnd] + ws * wa

        wBegin = wBegin + hop
        wEnd = wEnd + hop

# >>> MATLAB
# I = wAccum~=0;
# for j=1:nsrc
#     sproj(I,:,j) = sproj(I,:,j) ./(wAccum(I)*ones(1,nchanEst));
# end
#
# sproj = sproj(1:end-Lw+1,:,:);
# return
#
#
# <<< MATLAB
    I = wAccum != 0
    for j in range(1, nsrc + 1):  # MATLAB j=1:nsrc
        # MATLAB wAccum(I)*ones(1,nchanEst) is an outer product broadcasting the
        # accumulated window across channels.
        sproj[I, :, j - 1] = sproj[I, :, j - 1] / wAccum[I][:, np.newaxis]

    # MATLAB sproj(1:end-Lw+1,:,:)
    sproj = sproj[:sproj.shape[0] - Lw + 1, :, :]
    return sproj
