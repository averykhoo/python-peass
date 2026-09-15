"""Frequency response of every filter in the filterbank at given z values.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "gammatone/Gfb_Analyzer_zresponse.m"

import numpy as np

from .gfb_filter_zresponse import Gfb_Filter_zresponse


# >>> MATLAB
# function zresponse = Gfb_Analyzer_zresponse(analyzer, z)
# <<< MATLAB
def Gfb_Analyzer_zresponse(analyzer, z):
# >>> MATLAB
# % zresponse = Gfb_Analyzer_zresponse(analyzer, z)
# %
# % Computes the frequency response of the gammatone filters in the filterbank
# % at the frequencies z.
# % 
# % PARAMETERS
# % analyzer  A Gfb_Analyzer struct as created by Gfb_Analyzer_new.
# % z         A vector of z-plane frequencies where the frequency response
# %           should be computed. z = exp(2i*pi*f[Hz]/fs[Hz])
# % zresponse The complex frequency response of the filter (col) at z (row).
# %
# % copyright: Universitaet Oldenburg
# % author   : tp
# % date     : Jan & Nov 2006, Jan Feb 2007
#
# number_of_bands = length(analyzer.center_frequencies_hz);
# z = z(:);
# zresponse = ones(length(z), number_of_bands);
# <<< MATLAB
    number_of_bands = len(analyzer.center_frequencies_hz)
    # MATLAB z(:) reshapes to a column vector; a MATLAB column vector is a 1-D
    # array here (see NOTES.md).
    z = np.asarray(z).reshape(-1)
    # MATLAB's ones() is real and would be promoted by the complex assignment
    # in the loop below; numpy will not promote, so allocate complex.
    zresponse = np.ones((len(z), number_of_bands), dtype=complex)
# >>> MATLAB
#
# for band = [1:number_of_bands]
#   filter = analyzer.filters(band);
#   zresponse(:,band) = Gfb_Filter_zresponse(filter, z);
# end
# <<< MATLAB
    for band in range(1, number_of_bands + 1):  # MATLAB: 1:number_of_bands
        filter = analyzer.filters[band - 1]  # MATLAB (band)
        zresponse[:, band - 1] = Gfb_Filter_zresponse(filter, z)  # MATLAB (:,band)

    return zresponse

# >>> MATLAB
#
# %%-----------------------------------------------------------------------------
# %%
# %%   Copyright (C) 2006 2007  AG Medizinische Physik,
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
