# %%% LSDecompose_tv.m %%%
import numpy as np
import scipy.signal as signal
from LSDecompose import LSDecompose


# function sproj = LSDecompose_tv(se,s,flen,Lw,hop)
def LSDecompose_tv(se, s, flen, Lw, hop):
    # flen2 = (flen-1)/2;
    flen2 = (flen - 1) // 2
    # if flen2~=round(flen2)
    if (flen - 1) % 2 != 0:
        # error('filterParam:NotOdd','Not an odd order');
        raise ValueError("filterParam:NotOdd - Not an odd order")

    # s (end+1:end+flen-1+Lw-1,:) = 0;
    pad_len = flen - 1 + Lw - 1
    s = np.pad(s, ((0, pad_len), (0, 0)), mode='constant')
    # se(end+1:end+flen-1+Lw-1,:) = 0;
    se = np.pad(se, ((0, pad_len), (0, 0)), mode='constant')

    # [nsampl,nsrc]=size(s);
    nsampl, nsrc = s.shape
    # nchanEst = size(se,2);
    nchanEst = se.shape[1]

    # % sine analysis window / sine synthesis window
    # fahandle = @hann;
    # fshandle = @hann;
    # wa = sqrt(flipud(window(fahandle,Lw,'periodic')));
    # ws = sqrt(flipud(window(fshandle,Lw,'periodic')));
    h_win = signal.windows.hann(Lw, sym=False)
    wa = np.sqrt(np.flipud(h_win))
    ws = np.sqrt(np.flipud(h_win))

    # WS = zeros(Lw,nchanEst,nsrc);
    WS = np.zeros((Lw, nchanEst, nsrc))
    # for chan = 1:nchanEst
    for chan in range(nchanEst):
        # for j=1:nsrc
        for j in range(nsrc):
            # WS(:,chan,j) = ws;
            WS[:, chan, j] = ws

    # wBegin = 1;
    # wEnd = wBegin+Lw-1;
    # Convert boundary pointers to 0-based indexing
    wBegin = 0
    wEnd = wBegin + Lw

    # sproj = zeros(nsampl,nchanEst,nsrc);
    sproj = np.zeros((nsampl, nchanEst, nsrc), dtype=s.dtype)
    # wAccum = zeros(nsampl,1);
    wAccum = np.zeros((nsampl, 1))
    # Ns = size(s,2);
    Ns = s.shape[1]
    # Ls = size(s,1);
    Ls = s.shape[0]

    # while wEnd-Lw/2<=size(sproj,1)-Lw+1
    while wEnd - Lw // 2 <= sproj.shape[0] - Lw + 1:
        # sew = se(wBegin:wEnd,:);
        sew = se[wBegin:wEnd, :]
        # sw = [zeros(max(0,flen2-wBegin+1),Ns); s(max(1,wBegin-flen2):min(end,wEnd+flen2),:); zeros(max(0,wEnd+flen2-Ls),Ns)];
        sw_start = wBegin - flen2
        sw_end = wEnd + flen2

        pad_left = max(0, -sw_start)
        pad_right = max(0, sw_end - Ls)
        slice_start = max(0, sw_start)
        slice_end = min(Ls, sw_end)
        s_slice = s[slice_start:slice_end, :]

        sw = np.zeros((flen + Lw - 1, Ns), dtype=s.dtype)
        if pad_left > 0:
            sw[:pad_left, :] = 0
        sw[pad_left: pad_left + (slice_end - slice_start), :] = s_slice
        if pad_right > 0:
            sw[-pad_right:, :] = 0

        # sprojw=LSDecompose(sew,sw,flen2,wa);
        sprojw = LSDecompose(sew, sw, flen2, wa)

        # sproj(wBegin:wEnd,:,:) = sproj(wBegin:wEnd,:,:) + sprojw(1:Lw,:,:).*WS;
        sproj[wBegin:wEnd, :, :] += sprojw[:Lw, :, :] * WS

        # wAccum(wBegin:wEnd,1) = wAccum(wBegin:wEnd,1)+ws.*wa;
        wAccum[wBegin:wEnd, 0] += ws * wa

        # wBegin = wBegin+hop;
        wBegin += hop
        # wEnd = wEnd+hop;
        wEnd += hop

    # I = wAccum~=0;
    I = (wAccum[:, 0] != 0)
    # for j=1:nsrc
    for j in range(nsrc):
        # sproj(I,:,j) = sproj(I,:,j) ./(wAccum(I)*ones(1,nchanEst));
        sproj[I, :, j] /= wAccum[I, :]

    # sproj = sproj(1:end-Lw+1,:,:);
    sproj = sproj[:-(Lw - 1), :, :]
    # return
    return sproj