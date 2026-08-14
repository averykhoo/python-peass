"""MATLAB built-ins the transcriptions lean on, plus struct-field access.

This module is NOT a transcription of any `.m` file -- it has no
`MATLAB_SOURCE` and `verify_transcription.py` skips it. It exists so the
per-file transcriptions can spell MATLAB semantics once instead of seven times.

Nothing here may import from `peass`.
"""

from collections.abc import Mapping

import numpy as np

_MISSING = object()


def matlab_round(x):
    """MATLAB's `round`: half away from zero.

    Python's built-in `round` and `numpy.round` both do banker's rounding
    (half to even), so `round(0.5)` is 0 there and 1 in MATLAB. The
    decomposition hits exact halves in real inputs -- e.g. the shade-window
    length `2*round(shadeInMs/1000*fs+1)` and the per-band frame/hop sizes --
    so the distinction is load-bearing, not pedantry.

    Returns a Python int for scalars and an integer array otherwise, matching
    how the call sites use the result (as lengths and indices).
    """
    a = np.asarray(x, dtype=float)
    rounded = np.sign(a) * np.floor(np.abs(a) + 0.5)
    if rounded.ndim == 0:
        return int(rounded)
    return rounded.astype(np.int64)


def hann_periodic(n):
    """MATLAB's `hann(n,'periodic')` == `window(@hann,n,'periodic')`.

    Both equal `0.5 - 0.5*cos(2*pi*(0:n-1)/n)`, which is
    `scipy.signal.windows.hann(n, sym=False)`. Spelled out here rather than
    imported so the identity is auditable at the one place it is asserted.
    """
    k = np.arange(int(n), dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / float(n))


def mstruct_get(struct, name, default=_MISSING):
    """Read field `name` off a MATLAB-struct stand-in.

    `reference/gammatone/` owns the analyzer/synthesizer objects, and a MATLAB
    struct maps equally naturally onto a dict or onto an attribute-carrying
    object. Rather than couple the decomposition layer to whichever was chosen,
    accept both. MATLAB structs also accept fields being added after
    construction, which `myPemoAnalysisFilterBank` does (`analyzer.fsOrig`,
    `analyzer.Ndec`, ...); see `mstruct_set`.
    """
    if isinstance(struct, Mapping):
        if name in struct:
            return struct[name]
    elif hasattr(struct, name):
        return getattr(struct, name)
    if default is _MISSING:
        raise AttributeError(f"struct has no field {name!r}")
    return default


def mstruct_set(struct, name, value):
    """Assign field `name`, creating it if absent (MATLAB structs allow this)."""
    if isinstance(struct, Mapping):
        struct[name] = value
    else:
        setattr(struct, name, value)
    return struct
