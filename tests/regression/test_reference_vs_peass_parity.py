"""
PEASS Test Suite - reference/ vs peass/ Head to Head
File path: tests/regression/test_reference_vs_peass_parity.py

Everything else compares the two implementations *through* the MATLAB gold WAVs:
`test_reference_transcription.py` measures `reference/` against them and
`test_matlab_regression.py` measures `peass/` against them, and both land on the
same digits. That is strong evidence, but it is indirect, and it is capped at the
gold WAVs' own resolution -- they are PCM16, so neither test can see anything
below ~1e-5. This file compares the two implementations *directly*, in float64,
with the gold WAVs out of the loop.

That turns out to be a far sharper instrument than either test above:

    component            ch      1 - corr    RMS ratio   rel L2 error
    true_target           0       1.1e-16    1.00000000       1.77e-15
    true_target           1       0.0e+00    1.00000000       1.77e-15
    target_distortion     0       5.6e-16    1.00000000       9.84e-11
    target_distortion     1       0.0e+00    1.00000000       9.84e-11
    interference          0       0.0e+00    1.00000000       8.80e-11
    interference          1       0.0e+00    1.00000000       9.15e-11
    artifacts             0       0.0e+00    1.00000000       5.33e-10
    artifacts             1       0.0e+00    1.00000000       5.37e-10

(measured 2026-08-15 on the reference platform, NumPy backend)

Two things worth reading off that table. First, the agreement is at float64
roundoff -- five orders of magnitude tighter than either implementation's ~3e-6
agreement with the gold WAVs, which confirms that gap is quantization in the WAVs
rather than error in the port. Second, the RMS ratio is 1.0 and not the
`_MATLAB_RESAMPLER_GAIN_OFFSET` the other two tests assert: the offset cancels
here, which is only possible if both implementations carry the *same* deviation.
Two independent gain errors that happened to have similar magnitude would not
cancel to nine digits.

The bars below are what makes `reference/` usable as a live fuzzing oracle (see
TODO.md, "Property-based testing against the reference") -- that item needs a
tolerance, and until this test existed the tolerance would have been a guess.
"""

import numpy as np
import pytest
import soundfile as sf

from peass import decompose_distortion_components
from peass.config import DecompositionConfiguration
from reference.extractDistortionComponents import audioread


# Relative L2 error, ||peass - reference|| / ||reference||, per component.
#
# Per-component for the same reason `test_matlab_regression.py`'s correlation
# floors are: `artifacts` is the residual of an ill-conditioned least-squares
# stage (minimum Gram eigenvalue ~3.4e-10, see ARCHIVE.md), so a 1-ULP difference
# upstream arrives there ~1e6 larger. A single bar would be set by `artifacts`
# and would let `true_target` degrade five decades without failing.
#
# Allowances are the next round number at least ~50x above the measured value in
# the docstring table, sized for platform LAPACK/FFT variation -- CI also runs
# ubuntu-latest, and ground rule 3 in TODO.md exists because one machine's
# numbers have been written down here as universal invariants before.
#
#     component            measured     allowed    margin
#     true_target           1.8e-15       1e-13      ~56x
#     target_distortion     9.8e-11        1e-8      ~102x
#     interference          9.2e-11        1e-8      ~109x
#     artifacts             5.4e-10        1e-7      ~186x
_MAX_RELATIVE_L2 = {
    "true_target": 1e-13,
    "target_distortion": 1e-8,
    "interference": 1e-8,
    "artifacts": 1e-7,
}

# The resampler-normalization offset cancels between the two implementations, so
# unlike the gold-WAV tests this one expects unity. Bounded well inside the L2
# bars above because an energy-only discrepancy is a different failure mode from
# a waveform one: a gain regression in exactly one implementation would show here
# first, and stating it separately makes the failure message say which it was.
_MAX_GAIN_ERROR = {
    "true_target": 1e-12,
    "target_distortion": 1e-9,
    "interference": 1e-9,
    "artifacts": 1e-8,
}

_COMPONENTS = ["true_target", "target_distortion", "interference", "artifacts"]


@pytest.fixture(scope="module")
def peass_decomposition(matlab_ref_resources):
    """The NumPy backend on the same clips, with the same 10 ms shading the
    reference's default options use."""
    target, fs = sf.read(matlab_ref_resources / "targetSrc.wav")
    interference_1, _ = sf.read(matlab_ref_resources / "interfSrc1.wav")
    interference_2, _ = sf.read(matlab_ref_resources / "interfSrc2.wav")
    estimate, _ = sf.read(matlab_ref_resources / "targetEstimate.wav")

    configuration = DecompositionConfiguration(
        shade_in_milliseconds=10.0, shade_out_milliseconds=10.0
    )
    waveforms = decompose_distortion_components(
        source_files=[target, interference_1, interference_2],
        estimate_file=estimate,
        configuration=configuration,
        sampling_frequency_hz=float(fs),
    ).waveforms
    return [waveforms.true_target, waveforms.target_distortion,
            waveforms.interference, waveforms.artifacts]


@pytest.mark.regression
@pytest.mark.parametrize("index,label", list(enumerate(_COMPONENTS)), ids=_COMPONENTS)
def test_peass_matches_reference_at_full_precision(reference_decomposition,
                                                   peass_decomposition,
                                                   index, label):
    reference = np.asarray(reference_decomposition[index])
    produced = np.asarray(peass_decomposition[index])

    length = min(len(reference), len(produced))
    for channel in range(reference.shape[1]):
        reference_channel = reference[:length, channel]
        produced_channel = produced[:length, channel]

        relative_l2 = (np.linalg.norm(produced_channel - reference_channel)
                       / np.linalg.norm(reference_channel))
        assert relative_l2 < _MAX_RELATIVE_L2[label], (
            f"{label} (CH{channel}): peass/ and reference/ differ by "
            f"{relative_l2:.3e} relative L2, above the "
            f"{_MAX_RELATIVE_L2[label]:.0e} bar. This is a direct "
            "implementation-vs-implementation comparison, so the gold WAVs' "
            "PCM16 floor is not an explanation -- one of the two moved."
        )

        # No `_MATLAB_RESAMPLER_GAIN_OFFSET` here: it is common to both, so it
        # cancels. A ratio drifting towards 1.0025651 would mean exactly one
        # implementation still has it.
        gain_error = (np.sqrt(np.mean(produced_channel ** 2))
                      / np.sqrt(np.mean(reference_channel ** 2))) - 1.0
        assert abs(gain_error) < _MAX_GAIN_ERROR[label], (
            f"{label} (CH{channel}): RMS ratio peass/reference is "
            f"{1.0 + gain_error:.10f}, expected 1.0 +/- "
            f"{_MAX_GAIN_ERROR[label]:.0e}"
        )


@pytest.mark.regression
def test_length_conventions_differ_only_by_a_zero_tail(reference_decomposition,
                                                       peass_decomposition,
                                                       matlab_ref_resources):
    """The two implementations disagree on output length, and that is by design.

    `reference/` reproduces MATLAB, which returns what the decimate / interpolate
    / trim arithmetic yields -- 79557 samples for this 80000-sample input.
    `peass/` pads back out to the input length so callers can align the
    components with what they passed in. Every other test in the suite absorbs
    this silently through a `min(len(a), len(b))`, so it has never actually been
    asserted anywhere; a change from zero-padding to *repeating* or *reflecting*
    the tail would pass all of them.

    Pinned here as: the overlap is what the test above checks, and the excess is
    exactly zero -- not merely small.
    """
    gold, _ = audioread(matlab_ref_resources / "targetEstimate_true.wav")
    estimate, _ = audioread(matlab_ref_resources / "targetEstimate.wav")

    for label, reference, produced in zip(_COMPONENTS, reference_decomposition,
                                          peass_decomposition):
        reference = np.asarray(reference)
        produced = np.asarray(produced)

        assert len(reference) == len(gold), (
            f"{label}: reference/ length {len(reference)} no longer matches the "
            f"MATLAB gold WAV's {len(gold)}"
        )
        assert len(produced) == len(estimate), (
            f"{label}: peass/ length {len(produced)} no longer matches the input "
            f"estimate's {len(estimate)}"
        )
        assert len(produced) >= len(reference)

        tail = produced[len(reference):]
        assert np.count_nonzero(tail) == 0, (
            f"{label}: peass/ emits {np.count_nonzero(tail)} nonzero samples "
            f"(peak {np.max(np.abs(tail)):.3e}) past the length MATLAB produces. "
            "The tail is padding and must stay exactly zero."
        )
