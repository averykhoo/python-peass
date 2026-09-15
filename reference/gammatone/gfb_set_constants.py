"""Global constants for the MATLAB gammatone filterbank implementation.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "gammatone/Gfb_set_constants.m"

# >>> MATLAB
# % This file defines global constants for the matlab gammatone filterbank
# % implementation.
# %
# % copyright: Universitaet Oldenburg
# % author   : tp
# % date     : Jan 2002, Nov 2006
#
# % filename : Gfb_set_constants
#
# global GFB_L GFB_Q GFB_PREFERED_GAMMA_ORDER GFB_GAINCALC_ITERATIONS;
# <<< MATLAB
# Gfb_set_constants.m is a *script*, not a function: a caller declares the
# globals it wants with `global GFB_L` and then runs the script, which assigns
# them.  Here the globals are module-level names, and calling
# Gfb_set_constants() re-assigns them -- exactly what running the script does.
# Callers read them back as `gfb_set_constants.GFB_L` etc. so that they always
# see the current value, the way a MATLAB `global` declaration does.
GFB_L = None
GFB_Q = None
GFB_PREFERED_GAMMA_ORDER = None
GFB_GAINCALC_ITERATIONS = None


def Gfb_set_constants():
    global GFB_L, GFB_Q, GFB_PREFERED_GAMMA_ORDER, GFB_GAINCALC_ITERATIONS
# >>> MATLAB
#
# GFB_L = 24.7;  % see equation (17) in [Hohmann 2002]
# GFB_Q = 9.265; % see equation (17) in [Hohmann 2002]
# <<< MATLAB
    GFB_L = 24.7
    GFB_Q = 9.265
# >>> MATLAB
#
# % We will use 4th order gammatone filters:
# GFB_PREFERED_GAMMA_ORDER = 4;
# <<< MATLAB
    GFB_PREFERED_GAMMA_ORDER = 4
# >>> MATLAB
#
# % The gain factors are approximated in iterations. This is the default
# % number of iterations:
# GFB_GAINCALC_ITERATIONS  = 100;
# <<< MATLAB
    GFB_GAINCALC_ITERATIONS = 100


# >>> MATLAB
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


# Run the script once at import time so the module-level names are populated
# even for a reader who never calls Gfb_set_constants() explicitly.
Gfb_set_constants()
