"""Return a copy of a gammatone filter with its state cleared.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "gammatone/Gfb_Filter_clear_state.m"

import numpy as np


# >>> MATLAB
# function filter = Gfb_Filter_clear_state(filter)
# <<< MATLAB
def Gfb_Filter_clear_state(filter):
    # MATLAB structs are values: the caller's filter is not modified, the
    # cleared copy is the return value.
    filter = filter.copy()
# >>> MATLAB
# % filter = Gfb_Filter_clear_state(filter)
# % 
# % returns a copy of the filter, with the filter state cleared.
# %
# % PARAMETER:
# % filter  a Gfb_Filter structure as returned by Gfb_Filter_new.  A copy
# %         of the filter is returned, with the filter state cleared
# %
# % copyright: Universitaet Oldenburg
# % author   : tp
# % date     : Jan 2002, Nov 2006
#
# % filename : Gfb_Filter_clear_state.m
#
#
# filter.state = zeros(1, filter.gamma_order);
# <<< MATLAB
    # MATLAB zeros(1, N) is a 1xN row vector; a 1-D array of length N here.
    filter.state = np.zeros(int(filter.gamma_order))

    return filter

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
