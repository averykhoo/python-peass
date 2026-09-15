"""PEMO-Q auditory synthesis filterbank: upsample subbands, remodulate, resynthesize.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "myPemoSynthesisFilterBank.m"

import numpy as np
import scipy.signal

from ._matlab_runtime import matlab_round
from ._matlab_runtime import mstruct_get
from .gammatone import Gfb_Synthesizer_new
from .gammatone import Gfb_Synthesizer_process

# >>> MATLAB
# function [xSynth, synthesizer, M] = ...
#     myPemoSynthesisFilterBank(xFB,analyzer,M)
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Version 1.0
# % Copyright 2010 Valentin Emiya (INRIA).
# % This software is distributed under the terms of the GNU Public License
# % version 3 (http://www.gnu.org/licenses/gpl.txt).
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# <<< MATLAB


def myPemoSynthesisFilterBank(xFB, analyzer, M=None):
# >>> MATLAB
# if nargin < 2,
#     error('Not enough input arguments.');
# end
#
# <<< MATLAB
    # `nargin < 2` cannot happen: `analyzer` is a required positional parameter.

# >>> MATLAB
# % upsample
# Nb = size(xFB,1);
# fs = analyzer.sampling_frequency_hz;
# gfb_out_proc = zeros(Nb,max(cellfun('length',xFB(:)).*analyzer.Ndec(:)));
# for k=1:Nb
#     gfb_out_proc(k,1:length(xFB{k})*analyzer.Ndec(k)) = resample(xFB{k},analyzer.Ndec(k),1);
# end
# <<< MATLAB
    Nb = len(xFB)  # MATLAB size(xFB,1); xFB is a Nb-by-1 cell of row vectors
    fs = mstruct_get(analyzer, 'sampling_frequency_hz')
    Ndec = np.asarray(mstruct_get(analyzer, 'Ndec'))
    width = int(max(len(xFB[k]) * int(Ndec[k]) for k in range(Nb)))
    # Complex from the start: MATLAB's `zeros` is real but the assignment on the
    # next line promotes it, and numpy would instead discard the imaginary part.
    gfb_out_proc = np.zeros((Nb, width), dtype=complex)
    for k in range(1, Nb + 1):  # MATLAB k=1:Nb
        # !!! DEVIATION: MATLAB's `resample` is not reproducible from Octave or
        # scipy primitives at this project's precision (see
        # myPemoAnalysisFilterBank for the measurements).
        # `scipy.signal.resample_poly` is used instead. Interpolation by the
        # per-band integer decimation factor, undoing the analysis decimation.
        upsampled = scipy.signal.resample_poly(xFB[k - 1], int(Ndec[k - 1]), 1)
        # MATLAB gfb_out_proc(k,1:length(xFB{k})*analyzer.Ndec(k))
        gfb_out_proc[k - 1, 0:len(xFB[k - 1]) * int(Ndec[k - 1])] = upsampled

# >>> MATLAB
#
# if nargin<3 || isempty(M)
#     M = exp(2*1i*pi/fs*analyzer.center_frequencies_hz(:)*(0:size(gfb_out_proc,2)-1));
# end
# gfb_out_proc = gfb_out_proc.*M;
# <<< MATLAB
    if M is None or np.size(M) == 0:  # MATLAB nargin<3 || isempty(M)
        # Conjugate of the analysis modulation: shifts each baseband subband
        # back up to its center frequency.
        center_frequencies_hz = np.asarray(mstruct_get(analyzer, 'center_frequencies_hz'))
        M = np.exp(2 * 1j * np.pi / fs
                   * center_frequencies_hz.reshape(-1, 1)  # MATLAB (...)(:)
                   * np.arange(gfb_out_proc.shape[1]))  # MATLAB (0:size(...,2)-1)
    gfb_out_proc = gfb_out_proc * M

# >>> MATLAB
#
# % synthesis gammatone filterbank
# desired_delay_in_seconds = 1 / fs*1000;
# synthesizer = Gfb_Synthesizer_new(analyzer, desired_delay_in_seconds);
# [xSynth, synthesizer] = Gfb_Synthesizer_process(synthesizer, gfb_out_proc);
# xSynth = resample(xSynth,analyzer.fsOrig,analyzer.fs);
# xSynth = xSynth(round(desired_delay_in_seconds*analyzer.fsOrig+1):end);
#
# return
#
# <<< MATLAB
    # `1 / fs*1000` parses as `(1/fs)*1000`, i.e. 1000 SAMPLES expressed in
    # seconds -- not one millisecond. At fs=24000 that is 41.67 ms.
    desired_delay_in_seconds = 1 / fs * 1000
    synthesizer = Gfb_Synthesizer_new(analyzer, desired_delay_in_seconds)
    xSynth, synthesizer = Gfb_Synthesizer_process(synthesizer, gfb_out_proc)
    xSynth = np.asarray(xSynth).ravel()
    # !!! DEVIATION: MATLAB's `resample` -- see above. Back down to the input
    # sampling rate.
    xSynth = scipy.signal.resample_poly(xSynth,
                                        int(mstruct_get(analyzer, 'fsOrig')),
                                        int(mstruct_get(analyzer, 'fs')))
    # MATLAB xSynth(round(...+1):end). MATLAB `round` is half-away-from-zero,
    # and the +1 is inside it, so this is `matlab_round(delay*fsOrig + 1)` as a
    # 1-based start -> subtract 1 for the 0-based slice. Doing the +1 outside
    # the rounding would be off by one whenever the product lands near .5.
    start = matlab_round(desired_delay_in_seconds * mstruct_get(analyzer, 'fsOrig') + 1)
    xSynth = xSynth[start - 1:]

    return xSynth, synthesizer, M
