"""PEMO-Q auditory analysis filterbank: upsample, gammatone, demodulate, decimate.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "myPemoAnalysisFilterBank.m"

import numpy as np
import scipy.signal

from ._matlab_runtime import matlab_round
from ._matlab_runtime import mstruct_get
from ._matlab_runtime import mstruct_set
from .erbBW import erbBW
from .gammatone import Gfb_Analyzer_new
from .gammatone import Gfb_Analyzer_process

# >>> MATLAB
# function [gfb_out_dec, analyzer,M] = myPemoAnalysisFilterBank(x,fs,M)
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Version history
# % - Version 2.0, October 2011: reduced the number of frequency bands per
# % ERB
# % - Version 1.0, June 2010: first release
# % Copyright 2010-2011 Valentin Emiya (INRIA), Simon Maller and Pierre
# % Leveau (Audionamix)
# % This software is distributed under the terms of the GNU Public License
# % version 3 (http://www.gnu.org/licenses/gpl.txt).
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# <<< MATLAB


def myPemoAnalysisFilterBank(x, fs, M=None):
# >>> MATLAB
# if ~isdeployed
#     addpath(genpath('gammatone/'));
# end
#
# if nargin < 2,
#     error('Not enough input arguments.');
# end
#
# <<< MATLAB
    # `addpath(genpath('gammatone/'))` is MATLAB's path setup; the Python
    # equivalent is the module-level import of `reference.gammatone` above.
    #
    # `nargin < 2` cannot happen: `fs` is a required positional parameter.

# >>> MATLAB
# % ----------------------------------------------------------------------
# % settings
# % ----------------------------------------------------------------------
# MinCF = 20;				% (desired) minimum center frequency of the Gammatone filterbank in Hz
# MaxCF = fs/2; % (desired) maximum center frequency of the Gammatone filterbank in Hz
# base_freq = 1000;			% one of the gammatone filters will have this center frequency
# filters_per_ERB = 1.0;		% density of gammatone filterbank
#
# <<< MATLAB
    MinCF = 20  # (desired) minimum center frequency of the Gammatone filterbank in Hz
    MaxCF = fs / 2  # (desired) maximum center frequency of the Gammatone filterbank in Hz
    base_freq = 1000  # one of the gammatone filters will have this center frequency
    filters_per_ERB = 1.0  # density of gammatone filterbank

# >>> MATLAB
# % upsampling of input signal to avoid aliasing in upper gammatone filters
# fsOrig = fs;
# if fs/2 < 1.5*MaxCF,
#     x = resample(x, round(1.5*fs), fs);
#     fs = round(1.5*fs);
# end
# <<< MATLAB
    fsOrig = fs
    # MaxCF was fixed at the ORIGINAL fs/2 two statements ago and is NOT
    # recomputed after the upsample, so the filterbank still stops at the
    # original Nyquist. The upsample exists only to give the top filters
    # headroom. Reading `MaxCF` as `fs/2` of the new rate would add bands.
    if fs / 2 < 1.5 * MaxCF:  # always true, since MaxCF == fsOrig/2
        # !!! DEVIATION: MATLAB's `resample` is not reproducible from Octave or
        # scipy primitives at this project's precision -- measured: Octave's own
        # `resample` differs by ~9e-2, and Octave's `fir1` differs from scipy's
        # `firwin` by ~0.9% on the taps, so even rebuilding the filter by hand
        # does not converge. `scipy.signal.resample_poly` is used instead. Its
        # Kaiser(beta=5) design with 10*max(up,down) half-length is the same
        # family, and it reproduces the gold WAVs to a correlation of ~1.0 with
        # a known flat gain offset (SciPy's `firwin` normalizes the filter to
        # unit DC gain; MATLAB's does not). This is the ONLY unavoidable
        # numerical deviation in the decomposition layer.
        x = scipy.signal.resample_poly(x, matlab_round(1.5 * fs), int(fs))
        fs = matlab_round(1.5 * fs)

# >>> MATLAB
#
# % ----------------------------------------------------------------------
# % actual signal processing
# % ----------------------------------------------------------------------
#
# % gammatone filterbank
# analyzer = Gfb_Analyzer_new(fs, MinCF, base_freq, MaxCF, filters_per_ERB);
# analyzer.fsOrig = fsOrig;
# analyzer.fast = true;
# [gfb_out, analyzer] = Gfb_Analyzer_process(analyzer, x(:).');
# Nb = size(gfb_out,1);
# <<< MATLAB
    analyzer = Gfb_Analyzer_new(fs, MinCF, base_freq, MaxCF, filters_per_ERB)
    mstruct_set(analyzer, 'fsOrig', fsOrig)
    mstruct_set(analyzer, 'fast', True)
    # MATLAB x(:).' is a column-major flatten transposed back to a row vector.
    gfb_out, analyzer = Gfb_Analyzer_process(analyzer, np.asarray(x).ravel(order='F'))
    gfb_out = np.asarray(gfb_out)
    Nb = gfb_out.shape[0]  # MATLAB size(gfb_out,1)

# >>> MATLAB
# if nargin<3 || isempty(M)
#     M = exp(-2*1i*pi/fs*analyzer.center_frequencies_hz(:)*(0:size(gfb_out,2)-1));
# end
# gfb_out = gfb_out.*M;
# <<< MATLAB
    if M is None or np.size(M) == 0:  # MATLAB nargin<3 || isempty(M)
        # Outer product of the Nb center frequencies with sample indices
        # 0..N-1: the per-band complex demodulation to baseband. Cached by the
        # caller across signals of equal length, hence the third argument.
        center_frequencies_hz = np.asarray(mstruct_get(analyzer, 'center_frequencies_hz'))
        M = np.exp(-2 * 1j * np.pi / fs
                   * center_frequencies_hz.reshape(-1, 1)  # MATLAB (...)(:)
                   * np.arange(gfb_out.shape[1]))  # MATLAB (0:size(gfb_out,2)-1)
    gfb_out = gfb_out * M

# >>> MATLAB
#
# % decimate
# bw = erbBW(analyzer.center_frequencies_hz);
# alpha_dec = 2;
# gfb_out_dec = cell(Nb,1);
# Ndec = floor(fs./bw/alpha_dec);
# for k=1:Nb
#     gfb_out_dec{k} = resample(gfb_out(k,:),1,Ndec(k));
# end
# <<< MATLAB
    bw = erbBW(np.asarray(mstruct_get(analyzer, 'center_frequencies_hz')))
    alpha_dec = 2
    gfb_out_dec = [None] * Nb  # MATLAB cell(Nb,1)
    Ndec = np.floor(fs / bw / alpha_dec).astype(np.int64)
    for k in range(1, Nb + 1):  # MATLAB k=1:Nb
        # !!! DEVIATION: `resample` again -- see the note above. Per-band
        # decimation of the complex baseband subband by an integer factor.
        gfb_out_dec[k - 1] = scipy.signal.resample_poly(gfb_out[k - 1, :], 1, int(Ndec[k - 1]))

# >>> MATLAB
#
# if nargout>1
#     analyzer.Ndec = Ndec;
#     analyzer.fs = fs;
#     analyzer.bw = bw;
# end
# return;
#
#
#
# <<< MATLAB
    # `nargout>1` is not observable in Python, so the analyzer is always
    # decorated. Callers that ignore it are unaffected.
    mstruct_set(analyzer, 'Ndec', Ndec)
    mstruct_set(analyzer, 'fs', fs)
    mstruct_set(analyzer, 'bw', bw)
    return gfb_out_dec, analyzer, M
