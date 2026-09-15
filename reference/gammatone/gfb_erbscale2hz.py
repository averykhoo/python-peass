"""Frequency in Hz from a value on the ERB scale, equation (17) [Hohmann 2002].

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "gammatone/Gfb_erbscale2hz.m"

import numpy as np

from . import gfb_set_constants
from .gfb_set_constants import Gfb_set_constants


# >>> MATLAB
# function Hz = Gfb_erbscale2hz(ERBscale)
# <<< MATLAB
def Gfb_erbscale2hz(ERBscale):
# >>> MATLAB
# % Hz = Gfb_erbscale2hz(ERBscale)
# % 
# % implements equation (17) of [Hohmann 2002]: computes a frequency
# % in Hz from its value on the ERBscale
# %
# % copyright: Universitaet Oldenburg
# % author   : tp
# % date     : Jan 2002, Nov 2006
#
# % filename : Gfb_set_erbscale2hz.m
#
#
# global GFB_L;
# global GFB_Q;
# Gfb_set_constants;
# <<< MATLAB
    Gfb_set_constants()
    GFB_L = gfb_set_constants.GFB_L
    GFB_Q = gfb_set_constants.GFB_Q
# >>> MATLAB
#
# Hz = (exp(ERBscale / GFB_Q) - 1) * (GFB_L * GFB_Q);
# <<< MATLAB
    Hz = (np.exp(ERBscale / GFB_Q) - 1) * (GFB_L * GFB_Q)

    return Hz

# >>> MATLAB
#
#
# %%-----------------------------------------------------------------------------
# %%
# %%   Copyright (C) 2002 2006  AG Medizinische Physik,
# %%                        Universitaet Oldenburg, Germany
# %%                        http://www.physik.uni-oldenburg.de/docs/medi
# %%
# %%   Permission to use, copy, and distribute this software/file and its
# %%   documentation for any purpose without permission by UNIVERSITAET OLDENBURG
# %%   is not granted.
# %%   
# %%   Permission to use this software for academic purposes is generally
# %%   granted.
# %%
# %%   Permission to modify the software is granted, but not the right to
# %%   distribute the modified code.
# %%
# %%   This software is provided "as is" without expressed or implied warranty.
# %%
# %%   Author: Tobias Herzke
# %%
# %%-----------------------------------------------------------------------------
#
# <<< MATLAB
