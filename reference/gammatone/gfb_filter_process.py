"""Run a signal through one cascaded gammatone filter.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "gammatone/Gfb_Filter_process.m"

import numpy as np
from scipy.signal import lfilter


# >>> MATLAB
# function [output, filter_obj] = Gfb_Filter_process(filter_obj, input)
# <<< MATLAB
def Gfb_Filter_process(filter_obj, input):
    # MATLAB structs are values, so the filter the caller holds is untouched;
    # only the copy returned as the second output carries the new state.
    filter_obj = filter_obj.copy()
    input = np.asarray(input)
# >>> MATLAB
# % [output, filter] = Gfb_Filter_process(filter, input)
# %
# % The filter processes the input data.
# %
# % PARAMETERS
# % filter  A Gfb_Filter struct created from Gfb_Filter_new.  The filter
# %         will be returned with an updated filter state as the second
# %         return parameter
# % input   A vector containing the input signal to process
# % output  A vector containing the filter's output signal
# %
# % copyright: Universitaet Oldenburg
# % author   : tp
# % date     : Jan 2002, Nov 2006, Jan 2007
#
# % filename : Gfb_filter_process.m
#
#
# factor = filter_obj.normalization_factor;
# <<< MATLAB
    factor = filter_obj.normalization_factor
# >>> MATLAB
#
# % for compatibility of the filter state with the MEX extension, we
# % have to multiply the filter state with the filter coefficient before the
# % call to filter:
# filter_state = filter_obj.state * filter_obj.coefficient;
# <<< MATLAB
    filter_state = filter_obj.state * filter_obj.coefficient
# >>> MATLAB
#
# for i = [1:filter_obj.gamma_order]
#   [input, filter_state(i)] = ...
#       filter(factor, [1, -filter_obj.coefficient], ...
#              input, filter_state(i));
#   factor = 1;
# end
# <<< MATLAB
    for i in range(1, int(filter_obj.gamma_order) + 1):  # MATLAB: i = 1:gamma_order
        # scipy.signal.lfilter accepts complex b, a and zi and uses the same
        # transposed-direct-form-II state as MATLAB's filter(), so the returned
        # zf is MATLAB's final-condition output element for element.
        # A one-element slice keeps zi a length-1 array; MATLAB's filter_state(i)
        # is the scalar state of this first-order section.
        input, zf = lfilter([factor], [1, -filter_obj.coefficient],
                            input, zi=filter_state[i - 1:i])  # MATLAB filter_state(i)
        filter_state[i - 1] = zf[0]  # MATLAB filter_state(i)
        factor = 1
# >>> MATLAB
#
# output = input;
# <<< MATLAB
    output = input
# >>> MATLAB
#
# % for compatibility of the filter state with the MEX extension, we
# % have to divide the filter state by the filter coefficient after the
# % call to filter:
# filter_obj.state = filter_state / filter_obj.coefficient;
# <<< MATLAB
    filter_obj.state = filter_state / filter_obj.coefficient

    return output, filter_obj

# >>> MATLAB
#
#
# %%-----------------------------------------------------------------------------
# %%
# %%   Copyright (C) 2002 2006 2007 AG Medizinische Physik,
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
