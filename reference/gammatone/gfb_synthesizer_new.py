"""Constructor of a synthesizer matching a given analyzer.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "gammatone/Gfb_Synthesizer_new.m"

import copy

import numpy as np

from .gfb_delay_new import Gfb_Delay_new
from .gfb_mixer_new import Gfb_Mixer_new


class Gfb_Synthesizer(object):
    """The MATLAB struct `synthesizer` built by Gfb_Synthesizer_new.

    The attribute names are the MATLAB field names, verbatim.  MATLAB structs
    are values: assigning one, or passing one to a function, copies it.  Python
    objects are references, so every function here that returns a modified
    struct starts by taking a copy.
    """

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return "Gfb_Synthesizer(%s)" % ", ".join(sorted(self.__dict__))


# >>> MATLAB
# function synthesizer = Gfb_Synthesizer_new(analyzer, desired_delay_in_seconds)
# <<< MATLAB
def Gfb_Synthesizer_new(analyzer, desired_delay_in_seconds):
# >>> MATLAB
# % synthesizer = Gfb_Synthesizer_new(analyzer, desired_delay_in_seconds)
# %
# % Gfb_Synthesizer_new creates a new synthesizer object that fits to the
# % given analyzer.
# %
# % PARAMETERS:
# % analyzer                  an analyzer struct as returned by Gfb_Analyzer_new
# % desired_delay_in_seconds  the desired group delay of the analysis-synthesis
# %                           system in seconds.  Greater delays result in better
# %                           output signal quality.  Minimum delay is
# %                           (1 / analyzer.sampling_frequency_hz)
# % synthesizer               the constructed Gfb_Synthesizer structure
# %
# % copyright: Universitaet Oldenburg
# % author   : tp
# % date     : Jan 2002, Nov 2003, Nov 2006, Jan 2007
#
# % filename : Gfb_Synthesizer_new.m
#
#
# synthesizer.type         = 'Gfb_Synthesizer';
# desired_delay_in_samples = round(desired_delay_in_seconds * ...
# 			         analyzer.sampling_frequency_hz);
# <<< MATLAB
    synthesizer = Gfb_Synthesizer()
    synthesizer.type = 'Gfb_Synthesizer'
    # MATLAB round(): half away from zero, unlike Python's banker's rounding.
    _delay = desired_delay_in_seconds * analyzer.sampling_frequency_hz
    desired_delay_in_samples = int(np.sign(_delay) * np.floor(np.abs(_delay) + 0.5))
# >>> MATLAB
# if (desired_delay_in_samples < 1)
#     error('delay must be at least 1/analyzer.sampling_frequency_hz');
# end
# <<< MATLAB
    if desired_delay_in_samples < 1:
        raise ValueError('delay must be at least 1/analyzer.sampling_frequency_hz')
# >>> MATLAB
#
# synthesizer.delay = Gfb_Delay_new(analyzer, desired_delay_in_samples);
# synthesizer.mixer = Gfb_Mixer_new(analyzer, synthesizer.delay);
# <<< MATLAB
    synthesizer.delay = Gfb_Delay_new(analyzer, desired_delay_in_samples)
    synthesizer.mixer = Gfb_Mixer_new(analyzer, synthesizer.delay)

    return synthesizer

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
