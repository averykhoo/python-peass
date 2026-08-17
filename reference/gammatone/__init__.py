"""Interlinear Python transcription of the gammatone toolbox bundled with
MATLAB PEASS v2.0.1 (``v2.0.1/gammatone/``).

This is a reference implementation: an independent second opinion on the
optimised code in ``peass/``.  It therefore

* imports nothing from ``peass`` -- numpy, scipy and the stdlib only,
* carries the complete MATLAB source of each ``.m`` file as comments, byte for
  byte, interleaved with the Python that implements it (see FORMAT.md), and
* is deliberately not optimised: the loops mirror the MATLAB loops.

The MATLAB function names are kept verbatim, so ``Gfb_Analyzer_process`` is
spelled that way here too.  Module names are the lowercased file names.

Two files of the MATLAB toolbox are deliberately not transcribed, because
nothing in the transcribed set calls them: ``Gfb_Delay_clear_state.m`` and
``Gfb_Synthesizer_clear_state.m`` (nor are the ``Example_*.m`` scripts,
``Gfb_plot.m`` or the MEX sources).  See NOTES.md.
"""

from .gfb_analyzer_clear_state import Gfb_Analyzer_clear_state
from .gfb_analyzer_new import Gfb_Analyzer, Gfb_Analyzer_new
from .gfb_analyzer_process import Gfb_Analyzer_process
from .gfb_analyzer_zresponse import Gfb_Analyzer_zresponse
from .gfb_center_frequencies import Gfb_center_frequencies
from .gfb_delay_new import Gfb_Delay, Gfb_Delay_new
from .gfb_delay_process import Gfb_Delay_process
from .gfb_erbscale2hz import Gfb_erbscale2hz
from .gfb_filter_clear_state import Gfb_Filter_clear_state
from .gfb_filter_new import Gfb_Filter, Gfb_Filter_new
from .gfb_filter_process import Gfb_Filter_process
from .gfb_filter_zresponse import Gfb_Filter_zresponse
from .gfb_hz2erbscale import Gfb_hz2erbscale
from .gfb_mixer_new import Gfb_Mixer, Gfb_Mixer_new
from .gfb_mixer_process import Gfb_Mixer_process
from .gfb_set_constants import Gfb_set_constants
from .gfb_synthesizer_new import Gfb_Synthesizer, Gfb_Synthesizer_new
from .gfb_synthesizer_process import Gfb_Synthesizer_process

__all__ = [
    # the 17 requested MATLAB functions
    "Gfb_Analyzer_clear_state",
    "Gfb_Analyzer_new",
    "Gfb_Analyzer_process",
    "Gfb_Analyzer_zresponse",
    "Gfb_Delay_new",
    "Gfb_Delay_process",
    "Gfb_Filter_clear_state",
    "Gfb_Filter_new",
    "Gfb_Filter_process",
    "Gfb_Filter_zresponse",
    "Gfb_Mixer_new",
    "Gfb_Mixer_process",
    "Gfb_Synthesizer_new",
    "Gfb_Synthesizer_process",
    "Gfb_center_frequencies",
    "Gfb_erbscale2hz",
    "Gfb_hz2erbscale",
    # plus the constants script they depend on
    "Gfb_set_constants",
    # and the classes standing in for the MATLAB structs
    "Gfb_Analyzer",
    "Gfb_Delay",
    "Gfb_Filter",
    "Gfb_Mixer",
    "Gfb_Synthesizer",
]
