"""Filter centre frequencies, equidistant on the ERB scale.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "gammatone/Gfb_center_frequencies.m"

import numpy as np

from .gfb_erbscale2hz import Gfb_erbscale2hz
from .gfb_hz2erbscale import Gfb_hz2erbscale


# >>> MATLAB
# function center_frequencies_hz =                            ...
#       Gfb_center_frequencies(filters_per_ERBaud,            ...
# 			     lower_cutoff_frequency_hz,     ...
# 			     specified_center_frequency_hz, ...
# 			     upper_cutoff_frequency_hz)
# <<< MATLAB
def Gfb_center_frequencies(filters_per_ERBaud,
                           lower_cutoff_frequency_hz,
                           specified_center_frequency_hz,
                           upper_cutoff_frequency_hz=None):
    # MATLAB `nargin` counts the arguments actually passed by the caller.
    nargin = 3
    if upper_cutoff_frequency_hz is not None:
        nargin = 4
# >>> MATLAB
# % function frequencies_hz =                                   ...
# %       Gfb_center_frequencies(frequencies_per_ERBaud,        ...
# %                              lower_cutoff_frequency_hz,     ...
# %                              specified_center_frequency_hz, ...
# %                              upper_cutoff_frequency_hz)
# % 
# % constructs a vector of frequencies that are equidistand on the ERB
# % scale.
# % PARAMETERS:
# % frequencies_per_ERBaud     The density of frequencies on the ERB scale.
# % lower_cutoff_frequency_hz  The lowest possible frequency.
# % specified_center_frequency_hz       ( == "base frequency")
# %                            The result vector will contain this exact
# %                            frequency. Must be >= lower_cutoff_frequency_hz
# % upper_cutoff_frequency_hz  The highest possible frequency. Must be >=
# %                            specified_center_frequency_hz
# % OUTPUT:
# % frequencies_hz             A vector containing frequencies between
# %                            lower_cutoff_frequency_hz and
# %                            upper_cutoff_frequency_hz, equally
# %                            distributed on the ERB scale with a distance
# %                            of (1 / frequencies_per_ERBaud) ERB, with
# %                            one of the frequencies being
# %                            specified_center_frequency_hz.
# % copyright: Universitaet Oldenburg
# % author   : tp
# % date     : Jan, Sep 2003, Nov 2006, Feb 2007
#
# % filename : Gfb_center_frequencies.m
#
# if (nargin < 4)
#   upper_cutoff_frequency_hz = specified_center_frequency_hz;
# end
# <<< MATLAB
    if nargin < 4:
        upper_cutoff_frequency_hz = specified_center_frequency_hz
# >>> MATLAB
#
# % Calculate the values of the parameter frequencies on the ERBscale:
# lower_cutoff_frequency_erb     = ...
#     Gfb_hz2erbscale(lower_cutoff_frequency_hz);
# specified_center_frequency_erb = ...
#     Gfb_hz2erbscale(specified_center_frequency_hz);
# upper_cutoff_frequency_erb     = ...
#     Gfb_hz2erbscale(upper_cutoff_frequency_hz);
# <<< MATLAB
    lower_cutoff_frequency_erb = \
        Gfb_hz2erbscale(lower_cutoff_frequency_hz)
    specified_center_frequency_erb = \
        Gfb_hz2erbscale(specified_center_frequency_hz)
    upper_cutoff_frequency_erb = \
        Gfb_hz2erbscale(upper_cutoff_frequency_hz)
# >>> MATLAB
#
#
# % The center frequencies of the individual filters are equally
# % distributed on the ERBscale.  Distance between adjacent filters'
# % center frequencies is 1/filters_per_ERBaud.
# % First, we compute how many filters are to be placed at center
# % frequencies below the base frequency:
# erbs_below_base_frequency = ...
#     specified_center_frequency_erb - lower_cutoff_frequency_erb;
# num_of_filters_below_base_freq = ...
#     floor(erbs_below_base_frequency * filters_per_ERBaud);
# <<< MATLAB
    erbs_below_base_frequency = \
        specified_center_frequency_erb - lower_cutoff_frequency_erb
    num_of_filters_below_base_freq = \
        np.floor(erbs_below_base_frequency * filters_per_ERBaud)
# >>> MATLAB
#
# % Knowing this number of filters with center frequencies below the
# % base frequency, we can easily compute the center frequency of the
# % gammatone filter with the lowest center frequency:
# start_frequency_erb = ...
#     specified_center_frequency_erb - ...
#     num_of_filters_below_base_freq / filters_per_ERBaud;
# <<< MATLAB
    start_frequency_erb = \
        specified_center_frequency_erb - \
        num_of_filters_below_base_freq / filters_per_ERBaud
# >>> MATLAB
#
# % Now we create a vector of the equally distributed ERBscale center
# % frequency values:
# center_frequencies_erb = ...
#     [start_frequency_erb:(1/filters_per_ERBaud):upper_cutoff_frequency_erb];
# center_frequencies_hz = Gfb_erbscale2hz(center_frequencies_erb);
# <<< MATLAB
    # !!! DEVIATION: MATLAB's colon operator `base:step:limit` is not
    # np.arange.  MATLAB picks the element count by rounding (limit-base)/step
    # to the nearest integer and stepping back by one only if that overshoots
    # the limit, which makes it robust to the roundoff that would make
    # np.arange drop or gain a final element.  That rule is spelled out below;
    # the elements themselves are base + k*step either way.
    # (MATLAB's builtin additionally refines the values from both ends of the
    # range, which can move the result by an ulp.  With the PEASS parameters
    # -- filters_per_ERBaud = 1, so step = 1 -- that refinement is exact and
    # this port is bit-identical to it.)
    step = 1 / filters_per_ERBaud
    quotient = (upper_cutoff_frequency_erb - start_frequency_erb) / step
    # MATLAB round(): half away from zero, unlike Python's banker's rounding.
    n = int(np.sign(quotient) * np.floor(np.abs(quotient) + 0.5))
    if step * (start_frequency_erb + n * step - upper_cutoff_frequency_erb) > 0:
        n = n - 1
    center_frequencies_erb = \
        start_frequency_erb + np.arange(n + 1) * step
    center_frequencies_hz = Gfb_erbscale2hz(center_frequencies_erb)

    return center_frequencies_hz

# >>> MATLAB
#
# %%-----------------------------------------------------------------------------
# %%
# %%   Copyright (C) 2003 2006 2007 AG Medizinische Physik,
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
