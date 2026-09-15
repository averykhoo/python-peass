"""Constructor of the band gain factors of a synthesizer.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "gammatone/Gfb_Mixer_new.m"

import copy

import numpy as np

from . import gfb_set_constants
from .gfb_analyzer_zresponse import Gfb_Analyzer_zresponse
from .gfb_set_constants import Gfb_set_constants


class Gfb_Mixer(object):
    """The MATLAB struct `mixer` built by Gfb_Mixer_new.

    The attribute names are the MATLAB field names, verbatim.  MATLAB structs
    are values: assigning one, or passing one to a function, copies it.  Python
    objects are references, so every function here that returns a modified
    struct starts by taking a copy.
    """

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return "Gfb_Mixer(%s)" % ", ".join(sorted(self.__dict__))


# >>> MATLAB
# function mixer = Gfb_Mixer_new(analyzer, delay, iterations)
# <<< MATLAB
def Gfb_Mixer_new(analyzer, delay, iterations=None):
    # MATLAB `nargin` counts the arguments actually passed by the caller.
    nargin = 2
    if iterations is not None:
        nargin = 3
# >>> MATLAB
# % mixer = Gfb_Mixer_new(analyzer, delay, iterations)
# % 
# % Gfb_Mixer_new creates a Gfb_Mixer object with gain factors suitable
# % to calculate a weighted sum of the bands present in the output of the
# % given delay.  The gain factors are computed using a numerical optimization
# % method described in [Herzke & Hohmann 2007].
# % The <iterations> argument may be omitted.
# %
# % PARAMETERS
# % analyzer   A Gfb_Analyzer structure as created by Gfb_Analyzer_new. The
# %            mixer created by this function can act as part of a synthesizer
# %            that resynthesizes the output of this analyzer
# % delay      A Gfb_Delay structure as created by Gfb_Delay_new, Together with
# %            the mixer created by this function, this delay can form a
# %            synthesizer that resynthesizes the output of the analyzer
# % iterations The gain factors are approximated numerically in iterations.
# %            If this parameter is omitted, then the number of iterations will
# %            be  GFB_GAINCALC_ITERATIONS (see Gfb_set_constants.m, usually
# %            =100)
# %
# % copyright: Universitaet Oldenburg
# % author   : tp
# % date     : Jan 2002, Nov 2003, Mar & Nov 2006, Jan Feb 2007
#
# % filename : Gfb_Mixer_new.m
#
#
# global GFB_GAINCALC_ITERATIONS;
# Gfb_set_constants;
# <<< MATLAB
    Gfb_set_constants()
    GFB_GAINCALC_ITERATIONS = gfb_set_constants.GFB_GAINCALC_ITERATIONS
# >>> MATLAB
#
# mixer.type           = 'Gfb_Mixer';
# center_frequencies   = analyzer.center_frequencies_hz;
# number_of_bands   = length(center_frequencies);
# sampling_frequency   = analyzer.sampling_frequency_hz;
# <<< MATLAB
    mixer = Gfb_Mixer()
    mixer.type = 'Gfb_Mixer'
    center_frequencies = analyzer.center_frequencies_hz
    number_of_bands = len(center_frequencies)
    sampling_frequency = analyzer.sampling_frequency_hz
# >>> MATLAB
#
#
# % The center frequencies in the z plain
# z_c = exp(2i * pi * center_frequencies(:) / sampling_frequency);
# <<< MATLAB
    # MATLAB center_frequencies(:) is a column vector; a MATLAB column vector
    # is a 1-D array here (see NOTES.md).
    z_c = np.exp(2j * np.pi * np.asarray(center_frequencies).reshape(-1) / sampling_frequency)
# >>> MATLAB
#
# mixer.gains          = ones(number_of_bands, 1);
# <<< MATLAB
    mixer.gains = np.ones(number_of_bands)  # MATLAB ones(number_of_bands, 1)
# >>> MATLAB
#
# % compute the frequency response of each filter (col) at the center
# % frequencies of all filters (row)
#   pos_f_response = ...
#     Gfb_Analyzer_zresponse(analyzer, z_c);
#   neg_f_response = ...
#     Gfb_Analyzer_zresponse(analyzer, conj(z_c));
# <<< MATLAB
    pos_f_response = \
        Gfb_Analyzer_zresponse(analyzer, z_c)
    neg_f_response = \
        Gfb_Analyzer_zresponse(analyzer, np.conj(z_c))
# >>> MATLAB
#
# % apply delay and phase correction
# for band = [1:number_of_bands]
#   pos_f_response(:,band) = pos_f_response(:,band) * ...
#     delay.phase_factors(band) .* ...
#     z_c .^ -delay.delays_samples(band);
#   neg_f_response(:,band) = neg_f_response(:,band) * ...
#     delay.phase_factors(band) .* ...
#     conj(z_c) .^ -delay.delays_samples(band);
# end
# <<< MATLAB
    for band in range(1, number_of_bands + 1):  # MATLAB: 1:number_of_bands
        pos_f_response[:, band - 1] = pos_f_response[:, band - 1] * \
            delay.phase_factors[band - 1] * \
            z_c ** -delay.delays_samples[band - 1]
        neg_f_response[:, band - 1] = neg_f_response[:, band - 1] * \
            delay.phase_factors[band - 1] * \
            np.conj(z_c) ** -delay.delays_samples[band - 1]
# >>> MATLAB
#
# % combine responses at positive and negative responses to yield
# % responses for real part.
# f_response = (pos_f_response + conj(neg_f_response)) / 2;
# <<< MATLAB
    f_response = (pos_f_response + np.conj(neg_f_response)) / 2
# >>> MATLAB
#
# if (nargin < 4)
#   iterations = GFB_GAINCALC_ITERATIONS;
# end
# <<< MATLAB
    # Note the MATLAB condition is nargin < 4 although Gfb_Mixer_new only
    # takes three arguments, so `iterations` is always overwritten by
    # GFB_GAINCALC_ITERATIONS, even when the caller passes it.  Transcribed as
    # written; see NOTES.md.
    if nargin < 4:
        iterations = GFB_GAINCALC_ITERATIONS
# >>> MATLAB
# for i = [1:iterations]
#   % add selected spectrum of all bands with gain factors
#   selected_spectrum = f_response * mixer.gains;
#
#   % calculate better gain factors from result
#   mixer.gains = mixer.gains ./ abs(selected_spectrum);
# end
# <<< MATLAB
    for i in range(1, iterations + 1):  # MATLAB: 1:iterations
        # add selected spectrum of all bands with gain factors
        selected_spectrum = f_response @ mixer.gains  # MATLAB * is a matrix product here

        # calculate better gain factors from result
        mixer.gains = mixer.gains / np.abs(selected_spectrum)
# >>> MATLAB
# mixer.gains = mixer.gains.';
# <<< MATLAB
    # MATLAB .' turns the Nx1 column of gains into a 1xN row.  Both are 1-D
    # arrays here (see NOTES.md), so there is nothing to do.

    return mixer

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
