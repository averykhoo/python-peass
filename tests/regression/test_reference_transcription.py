"""
PEASS Test Suite - Interlinear MATLAB Transcription Checks
File path: tests/regression/test_reference_transcription.py

Three things are locked here:

1. every module under `reference/` embeds its MATLAB source byte for byte,
2. the set of deliberate `# !!! DEVIATION` markers is exactly the expected one,
   so a new silent deviation fails the build, and
3. the reference decomposition reproduces the MATLAB gold WAVs.

(3) is the real check on the port. It is slow (~15 s) by design: `reference/`
mirrors the MATLAB loops rather than the vectorized `peass/` implementation, and
that is the point -- it is an independent second opinion, so it must not share
`peass/`'s code, its shortcuts, or its bugs.

There is no `slow` marker registered in pyproject.toml (and this suite does not
add one), so these carry `regression` only.
"""

import numpy as np
import pytest

from reference import verify_transcription
from reference.extractDistortionComponents import audioread
from reference.extractDistortionComponents import extractDistortionComponents


# ---------------------------------------------------------------------------
# 1. Byte-exactness of the embedded MATLAB
# ---------------------------------------------------------------------------

def _module_ids(modules):
    return [module.name for module in modules]


_MODULES = verify_transcription.discover_modules()


@pytest.mark.regression
@pytest.mark.parametrize("module", _MODULES, ids=_module_ids(_MODULES))
def test_embedded_matlab_matches_source(module):
    """Each module's embedded MATLAB chunks reproduce its `.m` file exactly."""
    matlab_root = verify_transcription.default_matlab_root()
    if not matlab_root.is_dir():
        pytest.skip(
            f"MATLAB PEASS v2.0.1 source not present at {matlab_root} "
            "(it is gitignored, so this is expected on a fresh clone / in CI)"
        )

    result = verify_transcription.verify_module(module, matlab_root)
    if result.status == "SKIP":
        pytest.skip(result.detail)
    assert result.status == "PASS", f"{result.detail}\n\n{result.diff}"


@pytest.mark.regression
def test_reference_package_does_not_import_peass():
    """`reference/` is only worth having if it is independent of what it checks."""
    offenders = []
    for module in _MODULES:
        text = module.path.read_text(encoding="utf-8")
        for number, line in enumerate(text.split("\n"), start=1):
            stripped = line.strip()
            if stripped.startswith(("import peass", "from peass")):
                offenders.append(f"{module.name}:{number}: {stripped}")
    assert not offenders, (
        "reference/ must not import from peass -- it exists to be an independent "
        "second opinion:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# 2. The deliberate-deviation register
# ---------------------------------------------------------------------------

# How many `# !!! DEVIATION` blocks each module is allowed to declare. A new
# deviation -- anywhere under reference/ -- fails this test until it is added
# here, which is the point: departing from the MATLAB should be a decision, not
# an accident.
_EXPECTED_DEVIATION_COUNTS = {
    "reference.extractDistortionComponents": 3,
    "reference.myPemoAnalysisFilterBank": 2,
    "reference.myPemoSynthesisFilterBank": 2,
    "reference.gammatone.gfb_analyzer_process": 1,
    "reference.gammatone.gfb_center_frequencies": 1,
}

# For the decomposition layer the wording is pinned too, not just the count, so
# that swapping one deviation for a different one is also caught. The gammatone
# package is a separate deliverable, so only its counts are locked above.
_EXPECTED_DEVIATION_SUMMARIES = {
    "reference.extractDistortionComponents": [
        "the file I/O is inverted.",
        "`aux_segmentAndDecompose` and its helpers are not implemented.",
        "MATLAB's exact float-to-int16 rule for `audiowrite` is not",
    ],
    "reference.myPemoAnalysisFilterBank": [
        "MATLAB's `resample` is not reproducible from Octave or scipy",
        "`resample` again -- see the note above.",
    ],
    "reference.myPemoSynthesisFilterBank": [
        "MATLAB's `resample` is not reproducible from Octave or scipy",
        "MATLAB's `resample` -- see above.",
    ],
}


@pytest.mark.regression
def test_deviation_markers_match_expected_set():
    deviations = verify_transcription.collect_deviations()

    actual_counts = {}
    for deviation in deviations:
        actual_counts[deviation.module] = actual_counts.get(deviation.module, 0) + 1

    assert actual_counts == _EXPECTED_DEVIATION_COUNTS, (
        "the set of `# !!! DEVIATION` markers under reference/ changed.\n"
        f"expected: {_EXPECTED_DEVIATION_COUNTS}\n"
        f"actual:   {actual_counts}\n"
        "If the change is intended, update _EXPECTED_DEVIATION_COUNTS (and the "
        "summaries below) in this file."
    )

    for module_name, expected_summaries in _EXPECTED_DEVIATION_SUMMARIES.items():
        actual = [d.summary for d in deviations if d.module == module_name]
        assert len(actual) == len(expected_summaries)
        for summary, expected_prefix in zip(actual, expected_summaries):
            assert summary.startswith(expected_prefix), (
                f"{module_name}: deviation text changed.\n"
                f"expected it to start with: {expected_prefix!r}\n"
                f"actual: {summary!r}"
            )


@pytest.mark.regression
def test_every_deviation_explains_itself():
    """A marker with no explanation is not a deviation record, it is a shrug."""
    for deviation in verify_transcription.collect_deviations():
        assert len(deviation.summary) > 60, (
            f"{deviation.module}:{deviation.line}: a `# !!! DEVIATION` must say "
            f"what differs and why; got {deviation.summary!r}"
        )


# ---------------------------------------------------------------------------
# 3. The reference decomposition against the MATLAB gold WAVs
# ---------------------------------------------------------------------------

# The reference reproduces the gold WAVs up to one flat, frequency-independent
# gain, for the reason recorded in README ("Known deviations from the MATLAB
# reference") and re-derived independently here: SciPy's `firwin` normalizes the
# resampling filter to unit DC gain, MATLAB's does not, and four resamples per
# signal path compound to 1/0.997441484. `peass/`'s own regression test pins the
# same constant, and this file arriving at it through a completely separate
# implementation is corroboration, not duplication.
_MATLAB_RESAMPLER_GAIN_OFFSET = 1.0025651

# MEASURED, on the four components x two channels of the reference clip set:
#
#   component           ch  correlation   RMS ratio
#   true_target          0  0.999999956   1.002565626
#   true_target          1  0.999999956   1.002564210
#   target_distortion    0  0.999999752   1.002563812
#   target_distortion    1  0.999999757   1.002564109
#   interference         0  0.999999930   1.002565571
#   interference         1  0.999999926   1.002565426
#   artifacts            0  0.999996689   1.002553353
#   artifacts            1  0.999996727   1.002562157
#
# Worst correlation 0.999996689 -> 1-corr = 3.3e-6; the bound below allows 1e-5,
# ~3x headroom. Worst gain error against the constant above is -1.17e-5 (again
# `artifacts`, the quietest component and the one closest to the gold WAVs'
# PCM16 quantization floor); the bound allows 5e-5, ~4x headroom. Both are sized
# for platform LAPACK/FFT variation, not chosen to make the test pass -- the
# measured values are an order of magnitude inside them.
_MAX_CORRELATION_DEFICIT = 1e-5
_GAIN_TOLERANCE = 5e-5

_GOLD_FILES = [
    ("true_target", 0, "targetEstimate_true.wav"),
    ("target_distortion", 1, "targetEstimate_eTarget.wav"),
    ("interference", 2, "targetEstimate_eInterf.wav"),
    ("artifacts", 3, "targetEstimate_eArtif.wav"),
]


@pytest.fixture(scope="module")
def reference_decomposition(matlab_ref_resources):
    """Run the (deliberately slow) reference decomposition once for this module."""
    target, fs = audioread(matlab_ref_resources / "targetSrc.wav")
    interference_1, _ = audioread(matlab_ref_resources / "interfSrc1.wav")
    interference_2, _ = audioread(matlab_ref_resources / "interfSrc2.wav")
    estimate, _ = audioread(matlab_ref_resources / "targetEstimate.wav")

    components = extractDistortionComponents(
        [target, interference_1, interference_2], estimate, fs
    )
    return components[:4]


@pytest.mark.regression
def test_reference_output_length_matches_gold(reference_decomposition, matlab_ref_resources):
    """Length is set by the decimate/interpolate/trim arithmetic, so it is a
    sharp check on the 1-vs-0-indexed frame bookkeeping: off-by-one anywhere in
    LSDecompose_tv or the synthesis trim moves it."""
    gold, _ = audioread(matlab_ref_resources / "targetEstimate_true.wav")
    for component in reference_decomposition:
        assert component.shape == gold.shape


@pytest.mark.regression
@pytest.mark.parametrize("label,index,gold_filename", _GOLD_FILES,
                         ids=[row[0] for row in _GOLD_FILES])
def test_reference_reproduces_matlab_gold(reference_decomposition, matlab_ref_resources,
                                          label, index, gold_filename):
    produced = reference_decomposition[index]
    gold, _ = audioread(matlab_ref_resources / gold_filename)

    length = min(len(produced), len(gold))
    for channel in range(produced.shape[1]):
        produced_channel = produced[:length, channel]
        gold_channel = gold[:length, channel]

        correlation = np.corrcoef(produced_channel, gold_channel)[0, 1]
        assert 1.0 - correlation < _MAX_CORRELATION_DEFICIT, (
            f"{label} (CH{channel}): correlation with the MATLAB gold WAV is "
            f"{correlation:.9f}"
        )

        # Correlation is blind to gain, so pin the energy ratio against the
        # known, root-caused resampler-normalization offset as well.
        rms_ratio = (np.sqrt(np.mean(produced_channel ** 2))
                     / np.sqrt(np.mean(gold_channel ** 2)))
        gain_error = rms_ratio / _MATLAB_RESAMPLER_GAIN_OFFSET - 1.0
        assert abs(gain_error) < _GAIN_TOLERANCE, (
            f"{label} (CH{channel}): RMS ratio is {rms_ratio:.9f}, expected "
            f"{_MATLAB_RESAMPLER_GAIN_OFFSET} +/- {_GAIN_TOLERANCE:.0e} "
            f"(relative error {gain_error:+.2e})"
        )


@pytest.mark.regression
def test_reference_component_sum_matches_gold_sum(reference_decomposition,
                                                  matlab_ref_resources):
    """`eArtif` is defined as the residual, so the four components sum back to
    the resynthesized estimate by construction. Per-component errors therefore
    cancel in the sum, making it a much tighter check than any single component:
    a mis-transcribed sign or a dropped term in extractTSIA's final subtraction
    chain survives the per-component tolerances but not this one."""
    total = sum(reference_decomposition)
    gold_total = sum(audioread(matlab_ref_resources / filename)[0]
                     for _, _, filename in _GOLD_FILES)

    for channel in range(total.shape[1]):
        correlation = np.corrcoef(total[:, channel], gold_total[:, channel])[0, 1]
        assert 1.0 - correlation < _MAX_CORRELATION_DEFICIT, (
            f"CH{channel}: component sum correlates {correlation:.9f} with the "
            "gold sum"
        )
        rms_ratio = (np.sqrt(np.mean(total[:, channel] ** 2))
                     / np.sqrt(np.mean(gold_total[:, channel] ** 2)))
        assert abs(rms_ratio / _MATLAB_RESAMPLER_GAIN_OFFSET - 1.0) < _GAIN_TOLERANCE

    # And no component was silently zeroed.
    for component in reference_decomposition:
        assert np.sqrt(np.mean(component ** 2)) > 1e-4
