"""Constructor of a cascaded gammatone filter.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "gammatone/Gfb_Filter_new.m"

import copy
import math

import numpy as np

from . import gfb_set_constants
from .gfb_set_constants import Gfb_set_constants


class Gfb_Filter(object):
    """The MATLAB struct `filter` built by Gfb_Filter_new.

    The attribute names are the MATLAB field names, verbatim.  MATLAB structs
    are values: assigning one, or passing one to a function, copies it.  Python
    objects are references, so every function here that returns a modified
    struct starts by taking a copy.
    """

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return "Gfb_Filter(%s)" % ", ".join(sorted(self.__dict__))


# >>> MATLAB
# function filter = Gfb_Filter_new(arg1,arg2,arg3,arg4,arg5)
# <<< MATLAB
def Gfb_Filter_new(arg1, arg2, arg3=None, arg4=None, arg5=None):
    # MATLAB `nargin` counts the arguments actually passed by the caller.
    nargin = 2
    for _arg in (arg3, arg4, arg5):
        if _arg is None:
            break
        nargin = nargin + 1
# >>> MATLAB
# % Gfb_Filter_new is the constructor of a cascaded gammatonefilter. 
# % it may be called with 2, 3, 4 or 5 arguments:
# %
# % 2 arguments:
# % Specify complex filter coefficient directly:
# % Gfb_Filter_new(a_tilde,           % complex filter constant
# %                gamma_filter_order)% positive integer
# %
# % 3 or 4 arguments:
# % Compute filter coefficient from sampling rate, center frequency, and
# % order of the gammatone filter.  Filter will have 1 ERBaud equivalent
# % rectangular bandwidth, times bandwidth_factor.
# % Filter coefficient is computed from equations (13),(14)[Hohmann 2002].
# % Gfb_Filter_new(sampling_rate_hz,    % positive real number
# %                center_frequency_hz, % positive real number
# %                                     %      < sampling_rate_hz/2
# %                gamma_filter_order,  % positive integer
# %                bandwidth_factor)    % real number > 0, default is 1.0
# % 
# % 5 arguments:
# % Compute filter coefficient from sampling rate, center frequency, the
# % desired bandwidth with respect to the given attenuation, and the
# % order of the gammatone filter.
# % Filter coefficient is computed as in equations (11),(12)[Hohmann 2002]
# % (section 2.3).
# % Gfb_Filter_new(sampling_rate_hz,    % positive real number
# %                center_frequency_hz, % positive real number
# %                                     %  < sampling_rate_hz/2
# %                bandwidth_hz,        % posivive real number
# %                attenuation_db,      % positive real number, the
# %                                     % damping of this filter at
# %                                     % (center_frequency_hz +/-
# %                                     %            bandwidth_hz/2)
# %                gamma_filter_order)  % positive integer
# %
# % copyright: Universitaet Oldenburg
# % author   : tp
# % date     : Jan 2002, Nov 2006, Jan 2007
#
# % filename : Gfb_Filter_new.m
#
#
# filter.type = 'Gfb_Filter';
# if (nargin == 2)
#   filter.coefficient = arg1;
#   filter.gamma_order = arg2;
# <<< MATLAB
    filter = Gfb_Filter()
    filter.type = 'Gfb_Filter'
    if nargin == 2:
        filter.coefficient = arg1
        filter.gamma_order = arg2
# >>> MATLAB
# elseif (nargin == 3) | (nargin == 4)
#   sampling_rate_hz    = arg1;
#   center_frequency_hz = arg2;
#   filter.gamma_order  = arg3;
#   bandwidth_factor    = 1.0;
#   if (nargin == 4)
#     bandwidth_factor  = arg4;
#   end
#   global GFB_L;
#   global GFB_Q;
#   Gfb_set_constants;
# <<< MATLAB
    elif (nargin == 3) or (nargin == 4):
        sampling_rate_hz = arg1
        center_frequency_hz = arg2
        filter.gamma_order = arg3
        bandwidth_factor = 1.0
        if nargin == 4:
            bandwidth_factor = arg4
        Gfb_set_constants()
        GFB_L = gfb_set_constants.GFB_L
        GFB_Q = gfb_set_constants.GFB_Q
# >>> MATLAB
#
#   % equation (13) [Hohmann 2002]:
#   audiological_erb = (GFB_L + center_frequency_hz / GFB_Q) * bandwidth_factor;
#   % equation (14), line 3 [Hohmann 2002]:
#   a_gamma          = (pi * factorial(2*filter.gamma_order - 2) * ...
#                       2 ^ -(2*filter.gamma_order - 2) /              ...
#                       factorial(filter.gamma_order - 1) ^ 2);
#   % equation (14), line 2 [Hohmann 2002]:
#   b                = audiological_erb / a_gamma;
#   % equation (14), line 1 [Hohmann 2002]:
#   lambda           = exp(-2 * pi * b / sampling_rate_hz);
#   % equation (10) [Hohmann 2002]:
#   beta             = 2 * pi * center_frequency_hz / sampling_rate_hz;
#   % equation (1), line 2 [Hohmann 2002]:
#   filter.coefficient   = lambda * exp(1i * beta);
# <<< MATLAB
        # equation (13) [Hohmann 2002]:
        audiological_erb = (GFB_L + center_frequency_hz / GFB_Q) * bandwidth_factor
        # equation (14), line 3 [Hohmann 2002]:
        # MATLAB's factorial() takes a double; math.factorial needs an int.
        a_gamma = (np.pi * math.factorial(int(2 * filter.gamma_order - 2)) *
                   2.0 ** -(2 * filter.gamma_order - 2) /
                   math.factorial(int(filter.gamma_order - 1)) ** 2)
        # equation (14), line 2 [Hohmann 2002]:
        b = audiological_erb / a_gamma
        # equation (14), line 1 [Hohmann 2002]:
        # `lambda` is a Python keyword, so the MATLAB variable is spelled
        # lambda_ here and below.  Nothing else about it changes.
        lambda_ = np.exp(-2 * np.pi * b / sampling_rate_hz)
        # equation (10) [Hohmann 2002]:
        beta = 2 * np.pi * center_frequency_hz / sampling_rate_hz
        # equation (1), line 2 [Hohmann 2002]:
        filter.coefficient = lambda_ * np.exp(1j * beta)
# >>> MATLAB
# elseif (nargin == 5)
#   sampling_rate_hz    = arg1;
#   center_frequency_hz = arg2;
#   bandwidth_hz        = arg3;
#   attenuation_db      = arg4;
#   filter.gamma_order  = arg5;
# <<< MATLAB
    elif nargin == 5:
        sampling_rate_hz = arg1
        center_frequency_hz = arg2
        bandwidth_hz = arg3
        attenuation_db = arg4
        filter.gamma_order = arg5
# >>> MATLAB
#
#   % equation (12), line 4 [Hohmann 2002]:
#   phi    =  pi * bandwidth_hz / sampling_rate_hz;
#   % equation (12), line 3 [Hohmann 2002]:
#   u      = -attenuation_db/filter.gamma_order;
#   % equation (12), line 2 [Hohmann 2002]:
#   p      =  (-2 + 2 * 10^(u/10) * cos(phi)) / (1 - 10^(u/10));
#   % equation (12), line 1 [Hohmann 2002]:
#   lambda = -p/2 - sqrt(p*p/4 - 1);
#   % equation (10) [Hohmann 2002]:
#   beta   =  2 * pi * center_frequency_hz / sampling_rate_hz;
#   % equation (1), line 2 [Hohmann 2002]:
#   filter.coefficient   = lambda * exp(1i*beta);
# <<< MATLAB
        # equation (12), line 4 [Hohmann 2002]:
        phi = np.pi * bandwidth_hz / sampling_rate_hz
        # equation (12), line 3 [Hohmann 2002]:
        u = -attenuation_db / filter.gamma_order
        # equation (12), line 2 [Hohmann 2002]:
        p = (-2 + 2 * 10 ** (u / 10) * np.cos(phi)) / (1 - 10 ** (u / 10))
        # equation (12), line 1 [Hohmann 2002]:
        lambda_ = -p / 2 - np.sqrt(p * p / 4 - 1)
        # equation (10) [Hohmann 2002]:
        beta = 2 * np.pi * center_frequency_hz / sampling_rate_hz
        # equation (1), line 2 [Hohmann 2002]:
        filter.coefficient = lambda_ * np.exp(1j * beta)
# >>> MATLAB
# else
#   error ('Gfb_Filter_new needs either 2, 3, 4 or 5 arguments');
# end
# <<< MATLAB
    else:
        raise ValueError('Gfb_Filter_new needs either 2, 3, 4 or 5 arguments')
# >>> MATLAB
#
# % normalization factor from section 2.2 (text) [Hohmann 2002]:
# filter.normalization_factor = ...
#     2 * (1 - abs(filter.coefficient)) ^ filter.gamma_order;
# <<< MATLAB
    filter.normalization_factor = \
        2 * (1 - abs(filter.coefficient)) ** filter.gamma_order
# >>> MATLAB
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
