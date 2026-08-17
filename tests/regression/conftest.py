"""
PEASS Test Suite - Regression Fixtures
File path: tests/regression/conftest.py

The reference decomposition is the expensive fixture in this directory (~14 s by
design -- `reference/` mirrors the MATLAB loops rather than the vectorized
`peass/` implementation). Two modules need it: `test_reference_transcription.py`
compares it to the MATLAB gold WAVs, `test_reference_vs_peass_parity.py`
compares it to the fast backends. It is session-scoped here so the suite pays
for it once rather than once per module.
"""

import pytest

from reference.extractDistortionComponents import audioread
from reference.extractDistortionComponents import extractDistortionComponents


@pytest.fixture(scope="session")
def reference_decomposition(matlab_ref_resources):
    """The four reference components for the MATLAB reference clip set.

    Returns them in `_GOLD_FILES` order: true_target, target_distortion,
    interference, artifacts. `extractDistortionComponents` returns a fifth
    element (the output filenames), which is dropped.
    """
    target, fs = audioread(matlab_ref_resources / "targetSrc.wav")
    interference_1, _ = audioread(matlab_ref_resources / "interfSrc1.wav")
    interference_2, _ = audioread(matlab_ref_resources / "interfSrc2.wav")
    estimate, _ = audioread(matlab_ref_resources / "targetEstimate.wav")

    components = extractDistortionComponents(
        [target, interference_1, interference_2], estimate, fs
    )
    return components[:4]
