"""Run a signal through every band of the gammatone filterbank.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "gammatone/Gfb_Analyzer_process.m"

import numpy as np

from .gfb_filter_process import Gfb_Filter_process


# >>> MATLAB
# function [output, analyzer] = Gfb_Analyzer_process(analyzer, input)
# <<< MATLAB
def Gfb_Analyzer_process(analyzer, input):
    # MATLAB structs are values, so the analyzer the caller holds is untouched;
    # only the copy returned as the second output carries the new filter states.
    analyzer = analyzer.copy()
    input = np.asarray(input)
# >>> MATLAB
# % [output, analyzer] = Gfb_Analyzer_process(analyzer, input)
# %
# % The analyzer processes the input data.
# %
# % PARAMETERS
# % analyzer A Gfb_Analyzer struct created from Gfb_Analyzer_new. The
# %          analyzer will be returned (with updated filter states) as
# %          the second return parameter
# % input   Either a row vector containing the input signal to process, or
# %         a matrix containing different input signals for the different
# %         bands.  Different rows correspond to different filter bands,
# %         while different colums correspond to different times. 
# % output  A matrix containing the analyzer's output signals.  The
# %         rows correspond to the filter bands
# %
# % copyright: Universitaet Oldenburg
# % author   : tp
# % date     : Jan 2002, Sep 2003, Nov 2006, Jan 2007
#
# % filename : Gfb_Analyzer_process.m
#
#
# if (analyzer.fast)
#   % use matlab extension for fast computation.
#   [output, analyzer] = Gfb_Analyzer_fprocess(analyzer, input);
# else
# <<< MATLAB
    # !!! DEVIATION: analyzer.fast selects a compiled MEX implementation of
    # this function (Gfb_Analyzer_fprocess / Gfb_analyze.c).  There is no MEX
    # in this port, so `fast` is hard-wired to false and the pure-MATLAB `else`
    # branch is always taken.  The two branches are meant to compute the same
    # thing -- that is why Gfb_Filter_process pre-multiplies its filter state
    # by the filter coefficient, "for compatibility of the filter state with
    # the MEX extension".  Note PEASS's myPemoAnalysisFilterBank.m does set
    # analyzer.fast = true, so this branch is the one it would take in MATLAB.
    fast = False
    if fast:
        raise NotImplementedError(
            "Gfb_Analyzer_fprocess is a MEX extension and is not part of this port")
    else:
# >>> MATLAB
#   number_of_bands = length(analyzer.center_frequencies_hz);
#   output = zeros(number_of_bands, length(input));
#   for band = [1:number_of_bands]
#     [output(band,:), analyzer.filters(band)] = ...
#         Gfb_Filter_process(analyzer.filters(band), ...
#                            input );
#   end
# end
# <<< MATLAB
        number_of_bands = len(analyzer.center_frequencies_hz)
        # MATLAB length(input) is max(size(input)); the pure-MATLAB branch only
        # really supports a row vector, because it hands the whole of `input`
        # to every band.  MATLAB's zeros() is real and would be promoted by the
        # complex assignment below; numpy will not promote, so allocate complex.
        output = np.zeros((number_of_bands, max(np.shape(input))), dtype=complex)
        for band in range(1, number_of_bands + 1):  # MATLAB: 1:number_of_bands
            output[band - 1, :], analyzer.filters[band - 1] = \
                Gfb_Filter_process(analyzer.filters[band - 1],  # MATLAB (band)
                                   input)

    return output, analyzer

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
