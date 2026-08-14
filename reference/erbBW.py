"""Glasberg & Moore equivalent rectangular bandwidth of an auditory filter.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "erbBW.m"

import numpy as np

# >>> MATLAB
# function bw = erbBW(fc)
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Version 1.0
# % Copyright 2010 Valentin Emiya (INRIA).
# % This software is distributed under the terms of the GNU Public License
# % version 3 (http://www.gnu.org/licenses/gpl.txt).
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# <<< MATLAB


def erbBW(fc):
# >>> MATLAB
# bw = 24.7*(.00437*fc+1);
#
# return
# <<< MATLAB
    bw = 24.7 * (.00437 * np.asarray(fc, dtype=float) + 1)

    return bw
