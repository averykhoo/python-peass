"""Constructor of the analysis half of a gammatone filterbank.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "gammatone/Gfb_Analyzer_new.m"

import copy

from . import gfb_set_constants
from .gfb_center_frequencies import Gfb_center_frequencies
from .gfb_filter_new import Gfb_Filter_new
from .gfb_set_constants import Gfb_set_constants


class Gfb_Analyzer(object):
    """The MATLAB struct `analyzer` built by Gfb_Analyzer_new.

    The attribute names are the MATLAB field names, verbatim.  MATLAB structs
    are values: assigning one, or passing one to a function, copies it.  Python
    objects are references, so every function here that returns a modified
    struct starts by taking a copy.
    """

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return "Gfb_Analyzer(%s)" % ", ".join(sorted(self.__dict__))


# >>> MATLAB
# function analyzer = Gfb_Analyzer_new(sampling_frequency_hz,         ...
#                                      lower_cutoff_frequency_hz,     ...
#                                      specified_center_frequency_hz, ...
#                                      upper_cutoff_frequency_hz,     ...
#                                      filters_per_ERBaud,            ...
#                                      gamma_order,                   ...
#                                      bandwidth_factor)
# <<< MATLAB
def Gfb_Analyzer_new(sampling_frequency_hz,
                     lower_cutoff_frequency_hz,
                     specified_center_frequency_hz,
                     upper_cutoff_frequency_hz,
                     filters_per_ERBaud,
                     gamma_order=None,
                     bandwidth_factor=None):
    # MATLAB `nargin` counts the arguments actually passed by the caller.
    nargin = 5
    if gamma_order is not None:
        nargin = 6
    if bandwidth_factor is not None:
        nargin = 7
# >>> MATLAB
# % analyzer = Gfb_Analyzer_new(sampling_frequency_hz,         ...
# %                             lower_cutoff_frequency_hz,     ...
# %                             specified_center_frequency_hz, ...
# %                             upper_cutoff_frequency_hz,     ...
# %                             filters_per_ERBaud             ...
# %                             gamma_order,                   ...
# %                             bandwidth_factor)
# %
# % Gfb_Analyzer_new constructs a new Gfb_Analyzer object.  The analyzer
# % implements the analysis part of a gammatone filterbank as described
# % in [Hohmann 2002].
# % It consists of several all-pole gammatone filters; each
# % one with a bandwidth of 1 ERBaud (times bandwidth_factor),
# % and an order of gamma_order.
# % The center frequencies of the individual filters are computed as
# % described in section 3 of [Hohmann 2002].
# %
# % PARAMETERS: (all frequencies in Hz)
# % sampling_frequency_hz      The sampling frequency of the signals on which
# %                            the analyzer will operate
# % lower_cutoff_frequency_hz  The lowest possible center frequency of a
# %                            contained gammatone filter
# % specified_center_frequency_hz       ( == "base frequency")
# %                            One of the gammatone filters of the analyzer
# %                            will have this center frequency.  Must be >=
# %                            lower_cutoff_frequency_hz
# % upper_cutoff_frequency_hz  The highest possible center frequency of a
# %                            contained gammatone filter.  Must be >=
# %                            specified_center_frequency_hz
# % filters_per_ERBaud         The density of gammatone filters on the ERB
# %                            scale.
# % gamma_order                optional:
# %                            The order of the gammatone filters in this
# %                            filterbank.
# %                            If unspecified, the default value from
# %                            Gfb_set_constants.m is used.
# % bandwidth_factor           optional:
# %                            The bandwidth parameter of the individual filters
# %                            is calculated from the Equivalent Rectangular
# %                            Bandwidth (ERB) according to equation 14 in
# %                            [Hohmann 2002]. ERB is taken from the Glasberg &
# %                            Moore formula for a specific center frequency
# %                            (equation 13 in [Hohmann 2002]).
# %                            Using this parameter, it is possible to widen or
# %                            narrow all filters of the filterbank with a
# %                            constant bandwidth factor.
# %                            Default value is 1.0
# %
# % OUTPUT:
# % analyzer                   The constructed Gfb_Analyzer object.
# %
# % copyright: Universitaet Oldenburg
# % author   : tp
# % date     : Jan 2002, Jan, Sep 2003, Nov 2006, Jan 2007
#
# % filename : Gfb_Analyzer_new.m
#
# if (nargin < 6)
#   % The order of the gammatone filter is derived from the global constant
#   % GFB_PREFERED_GAMMA_ORDER defined in "Gfb_set_constants.m".  Usually,
#   % this is equal to 4.
#   global GFB_PREFERED_GAMMA_ORDER;
#   Gfb_set_constants;
#   gamma_order = GFB_PREFERED_GAMMA_ORDER;
# end
# <<< MATLAB
    if nargin < 6:
        # The order of the gammatone filter is derived from the global constant
        # GFB_PREFERED_GAMMA_ORDER defined in "Gfb_set_constants.m".  Usually,
        # this is equal to 4.
        Gfb_set_constants()
        GFB_PREFERED_GAMMA_ORDER = gfb_set_constants.GFB_PREFERED_GAMMA_ORDER
        gamma_order = GFB_PREFERED_GAMMA_ORDER
# >>> MATLAB
# if (nargin < 7)
#   bandwidth_factor = 1.0;
# end
# <<< MATLAB
    if nargin < 7:
        bandwidth_factor = 1.0
# >>> MATLAB
#
# % To avoid storing information in global variables, we use Matlab
# % structures:
# analyzer.type                          = 'Gfb_Analyzer';
# analyzer.sampling_frequency_hz         = sampling_frequency_hz;
# analyzer.lower_cutoff_frequency_hz     = lower_cutoff_frequency_hz;
# analyzer.specified_center_frequency_hz = specified_center_frequency_hz;
# analyzer.upper_cutoff_frequency_hz     = upper_cutoff_frequency_hz;
# analyzer.filters_per_ERBaud            = filters_per_ERBaud;
# analyzer.bandwidth_factor              = bandwidth_factor;
# analyzer.fast                          = 0;
# <<< MATLAB
    analyzer = Gfb_Analyzer()
    analyzer.type = 'Gfb_Analyzer'
    analyzer.sampling_frequency_hz = sampling_frequency_hz
    analyzer.lower_cutoff_frequency_hz = lower_cutoff_frequency_hz
    analyzer.specified_center_frequency_hz = specified_center_frequency_hz
    analyzer.upper_cutoff_frequency_hz = upper_cutoff_frequency_hz
    analyzer.filters_per_ERBaud = filters_per_ERBaud
    analyzer.bandwidth_factor = bandwidth_factor
    analyzer.fast = 0
# >>> MATLAB
#
#
# %
# analyzer.center_frequencies_hz = ...
#     Gfb_center_frequencies(filters_per_ERBaud, ...
# 			   lower_cutoff_frequency_hz,     ...
# 			   specified_center_frequency_hz, ...
# 			   upper_cutoff_frequency_hz);
# <<< MATLAB
    analyzer.center_frequencies_hz = \
        Gfb_center_frequencies(filters_per_ERBaud,
                               lower_cutoff_frequency_hz,
                               specified_center_frequency_hz,
                               upper_cutoff_frequency_hz)
# >>> MATLAB
#
# % This loop actually creates the gammatone filters:
# for band = [1:length(analyzer.center_frequencies_hz)]
#   center_frequency_hz = analyzer.center_frequencies_hz(band);
#
#   % Construct gammatone filter with one ERBaud bandwidth:
#   analyzer.filters(1,band) = ...
#       Gfb_Filter_new(sampling_frequency_hz, center_frequency_hz, ...
#                      gamma_order, bandwidth_factor);
# end
# <<< MATLAB
    # MATLAB grows the 1xN struct array `analyzer.filters` by assigning to
    # analyzer.filters(1,band); a Python list appended in band order is the
    # same thing.  analyzer.filters(band) and analyzer.filters(1,band) address
    # the same element (linear vs subscript indexing of a 1xN array), so the
    # two spellings both become analyzer.filters[band - 1] here.
    analyzer.filters = []
    for band in range(1, len(analyzer.center_frequencies_hz) + 1):  # MATLAB: 1:length(...)
        center_frequency_hz = analyzer.center_frequencies_hz[band - 1]  # MATLAB (band)

        # Construct gammatone filter with one ERBaud bandwidth:
        analyzer.filters.append(  # MATLAB analyzer.filters(1,band) = ...
            Gfb_Filter_new(sampling_frequency_hz, center_frequency_hz,
                           gamma_order, bandwidth_factor))

    return analyzer

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
