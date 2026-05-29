# %%% pemo_internal.m %%%
import numpy as np
import scipy.signal as signal
from gammatone_helper import Gfb_Analyzer_new, Gfb_Analyzer_process


# function [mx,fs]=pemo_internal(x,fs,modproc)
def pemo_internal(x, fs, modproc='lp'):
    # %%% Errors and warnings %%%
    # [nchan,nsampl]=size(x);
    # if nsampl < nchan, x=x'; nchan=nsampl; end
    if len(x.shape) > 1:
        if x.shape[0] < x.shape[1]:
            x = x.T
        x = x.ravel()
    nsampl = x.shape[0]

    # %%% Scaling %%%
    # x=10*x;
    x = 10.0 * x

    # %%% Basilar membrane filtering %%%
    # fmin=235;
    fmin = 235
    # fmax=min(.5*fs,14500);
    fmax = min(0.5 * fs, 14500)
    # if fs < 3*fmax,
    if fs < 3 * fmax:
        # x=resample(x,round(1.5*fs),fs);
        new_fs = int(round(1.5 * fs))
        x = signal.resample(x, int(round(len(x) * new_fs / fs)))
        # fs=round(1.5*fs);
        fs = new_fs

    nsampl = length = len(x)
    # analyzer=Gfb_Analyzer_new(fs,fmin,1000,fmax,1);
    analyzer = Gfb_Analyzer_new(fs, fmin, 1000, fmax, 1.0)
    # nband=length(analyzer.center_frequencies_hz);
    nband = analyzer['center_frequencies_hz'].shape[0]

    # rx=real(Gfb_Analyzer_process(analyzer,x));
    gfb_out, _ = Gfb_Analyzer_process(analyzer, x)
    rx = np.real(gfb_out)

    # %%% Envelope extraction %%%
    # % Haircell model (halfwave rectification, 1 kHz lowpass filter)
    # gain=exp(-pi*2000/fs);
    # rx=filter(1-gain,[1 -gain],max(rx,0),[],2);
    gain_hc = np.exp(-np.pi * 2000.0 / fs)
    b_hc = np.array([1.0 - gain_hc])
    a_hc = np.array([1.0, -gain_hc])
    rx_rect = np.maximum(rx, 0.0)
    for b in range(nband):
        rx[b, :], _ = signal.lfilter(b_hc, a_hc, rx_rect[b, :], zi=[0.0])

    # % Frequency-independent absolute hearing threshold and adaptation loops
    # Note: Porting the native MATLAB fallback slow-loop to bypass mex files
    # dbrange=single(100);
    dbrange = np.float32(100.0)
    # thresh=single(10^(-dbrange/20));
    thresh = np.float32(10.0 ** (-dbrange / 20.0))
    # bw=1./(pi*[0.005 0.05 0.129 0.253 0.5]);
    bw_loop = 1.0 / (np.pi * np.array([0.005, 0.05, 0.129, 0.253, 0.5], dtype=np.float32))
    # rx=max(single(rx),thresh);
    rx = np.maximum(rx.astype(np.float32), thresh)

    # for b=1:nband,
    for b in range(nband):
        # sthresh=thresh;
        sthresh = thresh
        # for s=1:5,
        for s in range(5):
            # gain=single(exp(-pi*bw(s)/fs));
            gain_val = np.float32(np.exp(-np.pi * bw_loop[s] / fs))
            # sthresh=sqrt(sthresh);
            sthresh = np.sqrt(sthresh)
            # factor=sthresh;
            factor = sthresh
            # for t=1:nsampl,
            for t in range(nsampl):
                # rx(b,t)=rx(b,t)/factor;
                val = rx[b, t] / factor
                rx[b, t] = val
                # factor=max((1-gain)*rx(b,t)+gain*factor,sthresh);
                factor = np.maximum((1.0 - gain_val) * val + gain_val * factor, sthresh)

    # rx=double(dbrange/(1-sthresh))*(double(rx)-double(sthresh));
    rx = float(dbrange / (1.0 - sthresh)) * (rx.astype(float) - float(sthresh))

    # %%% Modulation filtering %%%
    # if strcmp(modproc,'fb'),
    if modproc == 'fb':
        # rx=resample(rx.',800,fs).';
        rx = signal.resample(rx.T, int(round(rx.shape[1] * 800.0 / fs))).T
        # fs=800;
        fs = 800
        # fc=[0 5 10*(5/3).^(0:5)];
        fc = np.concatenate(([0.0, 5.0], 10.0 * (5.0 / 3.0) ** np.arange(6)))
        # bw=[5 5 5*(5/3).^(0:5)];
        bw_mod = np.concatenate(([5.0, 5.0], 5.0 * (5.0 / 3.0) ** np.arange(6)))
    # else
    else:
        # rx=resample(rx.',100,fs).';
        rx = signal.resample(rx.T, int(round(rx.shape[1] * 100.0 / fs))).T
        # fs=100;
        fs = 100
        # fc=0;
        fc = np.array([0.0])
        # bw=15.92;
        bw_mod = np.array([15.92])

    nmod = len(fc)
    nsampl = rx.shape[1]
    # mx=zeros(nband,nsampl,nmod);
    mx = np.zeros((nband, nsampl, nmod), dtype=complex)

    # for m=1:nmod,
    for m in range(nmod):
        # gain=exp(-pi*bw(m)/fs);
        gain_val = np.exp(-np.pi * bw_mod[m] / fs)
        # mx(:,:,m)=filter(1-gain,[1 -gain*exp(2i*pi*fc(m)/fs)],rx,[],2);
        b_mod = np.array([1.0 - gain_val])
        a_mod = np.array([1.0, -gain_val * np.exp(2j * np.pi * fc[m] / fs)])
        for b in range(nband):
            mx[b, :, m], _ = signal.lfilter(b_mod, a_mod, rx[b, :], zi=[0.0])

    # % Hilbert envelope above 10 Hz
    # above=(fc>10);
    above = (fc > 10.0)
    # mx(:,:,~above)=real(mx(:,:,~above));
    mx[:, :, ~above] = np.real(mx[:, :, ~above])
    # mx(:,:,above)=abs(mx(:,:,above));
    mx[:, :, above] = np.abs(mx[:, :, above])

    return mx, fs