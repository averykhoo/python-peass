"""
PEASS Test Suite - Process-Wide Environment Setup
File path: conftest.py

This lives at the repository root rather than in `tests/` because pytest loads
rootdir conftests first, and the workaround below only has any effect if it runs
before the first `import torch` -- which `tests/conftest.py` itself performs.
"""

import os

# Intel OpenMP duplicate-runtime guard (Windows).
#
# conda's MKL ships `Library/bin/libiomp5md.dll` and the PyPI torch wheel ships its
# own copy in `site-packages/torch/lib`. Whichever initializes second aborts the
# interpreter outright:
#
#     OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already
#     initialized.
#
# There is no Python traceback for this -- the run dies as "Fatal Python error:
# Aborted" inside whatever MKL call happened to trigger the second initialization,
# which makes it look like a numerical bug in the backend rather than a link-time
# collision. The suite is exposed regardless of which backend a given test exercises,
# because `tests/conftest.py` imports torch during collection to decide what to skip --
# so by the time any NumPy test calls into MKL, both runtimes are already loaded.
#
# `KMP_DUPLICATE_LIB_OK` is Intel's own escape hatch and is documented as unsafe and
# unsupported, but it is the configuration this project's results are recorded under.
# `setdefault` so that an explicit setting in the environment still wins -- e.g. if
# you prefer `MKL_THREADING_LAYER=SEQUENTIAL`, which is supported but single-threads
# MKL. See "Intel OpenMP conflict on Windows" in README.md for the alternatives.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
