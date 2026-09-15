"""Interlinear Python transcription of MATLAB PEASS v2.0.1.

Every module here carries the complete MATLAB source of exactly one `.m` file
as comments, byte for byte, interleaved with the Python that implements it (see
FORMAT.md). `verify_transcription.py` checks that mechanically.

This package is a reference implementation -- an independent second opinion on
the optimised code in `peass/`. It therefore imports nothing from `peass`
(numpy, scipy and the stdlib only) and is deliberately not optimised: the loops
mirror the MATLAB loops.

Nothing is re-exported at package level; import the module you want, e.g.
`from reference.extractDistortionComponents import extractDistortionComponents`.
"""
