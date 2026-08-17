"""Constructor of the per-band delay and phase correction of a synthesizer.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "gammatone/Gfb_Delay_new.m"

import copy

import numpy as np

from .gfb_analyzer_clear_state import Gfb_Analyzer_clear_state
from .gfb_analyzer_process import Gfb_Analyzer_process


class Gfb_Delay(object):
    """The MATLAB struct `delay` built by Gfb_Delay_new.

    The attribute names are the MATLAB field names, verbatim.  MATLAB structs
    are values: assigning one, or passing one to a function, copies it.  Python
    objects are references, so every function here that returns a modified
    struct starts by taking a copy.
    """

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return "Gfb_Delay(%s)" % ", ".join(sorted(self.__dict__))


# >>> MATLAB
# function delay = Gfb_Delay_new(analyzer, delay_samples)
# <<< MATLAB
def Gfb_Delay_new(analyzer, delay_samples):
# >>> MATLAB
# % delay = Gfb_Delay_new(analyzer, delay_samples)
# %
# % Gfb_Delay_new creates a new Gfb_Delay object that can act as the first stage
# % of a synthsizer that resynthesizes the output of the gammatone filterbank
# % analyzer.  The purpose of the delay object is to delay the output of each
# % band by a band-dependent ammount of samples, so that the envelope of
# % the impulse response of the analyzer is as large as possible at the desired
# % delay.
# % Additionally, the delay object will multiply this delayed output with a
# % band-dependent complex phase factor, so that the real part of the impulse
# % response has a local maximum at the desired delay.  Finally, the delay ob-
# % ject will output only the real part of each band.
# %
# % The phase factors are approximated numerically in this constructor,
# % using a method described in [Herzke & Hohmann 2007].  The
# % approximation assumes parabolic behaviour of the real part of the
# % impulse response in the region of the desired local maximum: The phase
# % factors are chosen so that the real parts of the impulse response in
# % the samples directly preceeding and following the desired local
# % maximum will be equal after multiplication with the pase factor.
# %
# % PARAMETERS:
# % analyzer      The Gfb_Analyzer structure as returned by Gfb_Analyzer_new.
# % delay_samples The desired group delay in samples. must be at least 1,
# %               because of the way the phase factors are computed.  Larger
# %               delays lead to better signal quality
# % delay         The new Gfb_Delay object
# %
# % copyright: Universitaet Oldenburg
# % author   : tp
# % date     : Jan 2002; Nov 2003; Mar Jun Nov 2006; Jan Feb 2007
#
# % filename : Gfb_Delay_new.m
#
#
# delay.type           = 'Gfb_Delay';
# <<< MATLAB
    delay = Gfb_Delay()
    delay.type = 'Gfb_Delay'
# >>> MATLAB
#
#   analyzer             = Gfb_Analyzer_clear_state(analyzer);
#   impulse              = zeros(1, delay_samples + 2);
#   impulse(1)           = 1;
# <<< MATLAB
    # Gfb_Analyzer_clear_state returns a copy, so the caller's analyzer keeps
    # whatever state it had -- as in MATLAB.
    analyzer = Gfb_Analyzer_clear_state(analyzer)
    impulse = np.zeros(int(delay_samples) + 2)  # MATLAB zeros(1, delay_samples+2)
    impulse[0] = 1  # MATLAB impulse(1)
# >>> MATLAB
#
#     impulse_response = ...
#       Gfb_Analyzer_process(analyzer, impulse);
# <<< MATLAB
    # MATLAB asks for one output here and drops the updated analyzer.
    impulse_response, _ = \
        Gfb_Analyzer_process(analyzer, impulse)
# >>> MATLAB
#
# number_of_bands      = size(impulse_response, 1);
# <<< MATLAB
    number_of_bands = np.shape(impulse_response)[0]  # MATLAB size(..., 1)
# >>> MATLAB
#
# [dummy, max_indices] = max(abs(impulse_response(:,1:(delay_samples+1))).');
# <<< MATLAB
    # MATLAB max() over the first dimension of the transposed matrix, i.e.
    # over time within each band, returning the maximum and its 1-based index.
    # Both MATLAB's max and numpy's argmax report the *first* maximum.
    _abs_response = np.abs(impulse_response[:, 0:(int(delay_samples) + 1)]).T  # MATLAB (:,1:(delay_samples+1)).'
    dummy = np.max(_abs_response, axis=0)
    max_indices = np.argmax(_abs_response, axis=0) + 1  # +1: MATLAB indices are 1-based
# >>> MATLAB
#
# delay.delays_samples = delay_samples + 1 - max_indices;
# <<< MATLAB
    delay.delays_samples = delay_samples + 1 - max_indices
# >>> MATLAB
#
# delay.memory         = zeros(number_of_bands, max(delay.delays_samples));
# <<< MATLAB
    # MATLAB zeros(N, M) with M = max(delays_samples); the memory holds real
    # samples only (Gfb_Delay_process stores real(...) into it).
    delay.memory = np.zeros((number_of_bands, int(np.max(delay.delays_samples))))
# >>> MATLAB
#
# slopes = zeros(1, number_of_bands);
# for band = [1:number_of_bands]
#   band_max_index = max_indices(band);
#   slopes(band) = (impulse_response(band, band_max_index+1) - ...
#                   impulse_response(band, band_max_index-1));
# end
# <<< MATLAB
    # MATLAB's zeros() is real and would be promoted by the complex assignment
    # in the loop; numpy will not promote, so allocate complex.
    slopes = np.zeros(number_of_bands, dtype=complex)  # MATLAB zeros(1, number_of_bands)
    for band in range(1, number_of_bands + 1):  # MATLAB: 1:number_of_bands
        band_max_index = max_indices[band - 1]  # MATLAB (band)
        if band_max_index < 2:
            # MATLAB would index impulse_response(band, 0) below and raise
            # "Subscript indices must be either real positive integers or
            # logicals".  Python would silently wrap to the last sample, so
            # raise instead of quietly computing something different.
            raise IndexError(
                "impulse response of band %d peaks at its first sample; "
                "Gfb_Delay_new needs a preceding sample to estimate the slope"
                % band)
        slopes[band - 1] = (impulse_response[band - 1, band_max_index + 1 - 1] -
                            impulse_response[band - 1, band_max_index - 1 - 1])
        # (MATLAB: impulse_response(band, band_max_index+1)
        #        - impulse_response(band, band_max_index-1); the trailing -1 on
        #        each subscript is the 1-based to 0-based conversion.)
# >>> MATLAB
# slopes = slopes ./ abs(slopes);
# delay.phase_factors = 1i ./ slopes;
# <<< MATLAB
    slopes = slopes / np.abs(slopes)
    delay.phase_factors = 1j / slopes

    return delay

# >>> MATLAB
#
#
# %%-----------------------------------------------------------------------------
# %%
# %%   Copyright (C) 2002 2003 2006 2007 AG Medizinische Physik,
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
