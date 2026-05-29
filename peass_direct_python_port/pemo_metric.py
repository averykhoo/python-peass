"""
PEASS Toolkit - Python Port
Equivalent of pemo_metric.m
"""
import numpy as np


def pemo_metric(mref: np.ndarray, mtest: np.ndarray, fs: float) -> float:
    # %%% MATLAB Code %%%
    nband, nsampl, nmod = mref.shape

    # %%% Assimilation and cross-correlation %%%
    # assim=(abs(mtest)<abs(mref));
    # mtest(assim)=.25*mref(assim)+.75*mtest(assim);
    assim = (np.abs(mtest) < np.abs(mref))
    mtest[assim] = 0.25 * mref[assim] + 0.75 * mtest[assim]

    # flen=min(nsampl,.1*fs); nfram=floor(nsampl/flen); nsampl=nfram*flen;
    flen = int(min(nsampl, 0.1 * fs))
    nfram = int(np.floor(nsampl / flen))
    nsampl = nfram * flen

    mref = mref[:, :nsampl, :]
    mtest = mtest[:, :nsampl, :]

    PSMt = np.zeros(nfram)
    lPSM = np.zeros(nmod)
    lNMS = np.zeros(nmod)

    for t in range(nfram):
        for m in range(nmod):
            # lref=mref(:,(t-1)*flen+1:t*flen,m); lref=lref(:)-mean(lref(:));
            lref = mref[:, t * flen: (t + 1) * flen, m]
            lref_flat = lref.ravel()
            lref_flat = lref_flat - np.mean(lref_flat)

            # ltest=mtest(:,(t-1)*flen+1:t*flen,m); lNMS(m)=sum(ltest(:).*ltest(:));
            ltest = mtest[:, t * flen: (t + 1) * flen, m]
            ltest_flat_orig = ltest.ravel()
            lNMS[m] = np.sum(ltest_flat_orig ** 2)

            # ltest=ltest(:)-mean(ltest(:));
            ltest_flat = ltest_flat_orig - np.mean(ltest_flat_orig)

            # lPSM(m)=sum(lref.*ltest)./sqrt(sum(lref.*lref)*sum(ltest.*ltest));
            denom = np.sqrt(np.sum(lref_flat ** 2) * np.sum(ltest_flat ** 2))
            lPSM[m] = np.sum(lref_flat * ltest_flat) / denom if denom != 0 else 0.0

        # PSMt(t)=sum(lPSM.*lNMS)/sum(lNMS);
        sum_lnms = np.sum(lNMS)
        PSMt[t] = np.sum(lPSM * lNMS) / sum_lnms if sum_lnms != 0 else 0.0

    # %%% From local to global similarity %%%
    # ilen=1*fs; mtest=sum(sum(mtest.^2,1),3);
    ilen = int(1 * fs)
    mtest_sq = np.sum(mtest ** 2, axis=(0, 2))

    RMS = np.zeros(nfram)
    for t in range(nfram):
        # ltest=mtest(max(1,(t-.5)*flen-.5*ilen+1):min(nsampl,(t-.5)*flen+.5*ilen));
        start_idx = int(max(0, (t + 0.5) * flen - 0.5 * ilen))
        end_idx = int(min(nsampl, (t + 0.5) * flen + 0.5 * ilen))
        ltest = mtest_sq[start_idx:end_idx]
        # RMS(t)=mean(ltest);
        RMS[t] = np.mean(ltest) if len(ltest) > 0 else 0.0

    # [PSMt,ind]=sort(PSMt); RMS=RMS(ind); RMS=cumsum(RMS);
    ind = np.argsort(PSMt)
    PSMt_sorted = PSMt[ind]
    RMS_sorted = RMS[ind]
    RMS_cum = np.cumsum(RMS_sorted)

    # ind=find(RMS>=.5*RMS(end)); PSMt=PSMt(ind(1));
    cutoff = 0.5 * RMS_cum[-1]
    match_indices = np.where(RMS_cum >= cutoff)[0]

    return PSMt_sorted[match_indices[0]] if len(match_indices) > 0 else 0.0
