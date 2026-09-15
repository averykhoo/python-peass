"""
PEASS Test Suite - reference/ as a Live Fuzzing Oracle
File path: tests/regression/test_reference_oracle_property.py

Every invariant the suite has so far constrains the FORM of the decomposition and is
satisfied by a wrong answer. `true_target + target_distortion + interference +
artifacts == estimate` holds for *any* partition of the estimate; gain invariance
holds for any homogeneous map; the gammatone round trip says nothing at all about how
the estimate was split. Only a reference constrains the CONTENT -- and until this file
existed the reference was consulted on exactly one input: the 5 s / 16 kHz / stereo /
3-source clip in `tests/resources/matlab_reference/`, via the session-scoped
`reference_decomposition` fixture in this directory's conftest.

This file uses `reference/` as a LIVE oracle instead: hypothesis draws an input, the
reference decomposes it, and the fast backends are asserted to agree. It is the only
test in the suite whose expected values are *computed* rather than stored, which is
also why it is the only one that can cover inputs no gold WAV exists for -- short
clips, 8 kHz, mono, a 2-source Gram, a synthetic nonlinear estimate.

WHY IT IS OPT-IN
    Every example costs one full `reference/` decomposition, and the reference is slow by
    design -- it mirrors the MATLAB loops rather than the vectorised `peass/`
    implementation. Inside this file's strategy box (mono, 2 sources, 8/16 kHz,
    0.25-0.75 s) a reference call measures ~0.3-1.2 s, so the default 40 examples plus the
    two trap pins is a ~1 minute job. That is a rough wall clock taken as a budget
    estimate, NOT a benchmark -- do not quote it as a perf number.
    (For scale: the 5 s / 16 kHz / stereo / 3-source gold clip costs ~12-15 s, roughly 10x
    a worst-case draw from this box. An earlier revision of this file used the GOLD CLIP
    figure as the per-example cost and overstated the budget by ~10x. It is fixed here.)
    It is opt-in because it is a LIVE oracle whose cost scales with `PEASS_ORACLE_EXAMPLES`
    and whose value is in being run deliberately, not because it is expensive. It carries
    the `oracle` marker, which `pyproject.toml` deselects by default, and NOTHING in
    `.github/workflows/` reselects it -- that is intentional (decision 2026-08-17). Opt in
    with `pytest -m oracle`; size it with `PEASS_ORACLE_EXAMPLES`. See "RUNNING IT" below.

WHAT IS DELIBERATELY EXCLUDED, AND WHY
    Hypothesis finds the degenerate cases immediately and they are traps, not bugs.
    Both were hit while building `benchmarks/measure.py`; both are re-measured here and
    pinned as characterization tests at the bottom of this file so that nobody
    "simplifies" the strategy back into one.

    1. A LINEAR estimate lies exactly in the span of the sources, so `artifacts` and
       `target_distortion` come out at ~1e-12 of the estimate RMS and comparing them is
       noise against noise -- measured 1.1e-2 and 2.4e-2 relative L2, i.e. a hard fail
       against any sane bar, for a decomposition that is entirely correct.
       EXCLUDED BY: the estimate is always `tanh(drive*mix)/drive` with `drive >= 1`
       plus independent shaped additive noise at -40..-25 dB. The additive term alone
       is statistically independent of both sources, so the estimate provably cannot
       lie in their span. This is `benchmarks/measure.py`'s FROZEN CONVENTION for the
       nonlinear estimate, with drive and noise level lifted into the strategy.

    2. Sources that are FIR-related to each other make the per-frame Gram
       rank-deficient, so `target_distortion` and `interference` become minimum-norm
       garbage: the two implementations pick *different* minimum-norm solutions and
       disagree by 3.6e-1 / 3.9e-1 relative L2 while their peaks agree to ~3%.
       measure.py hit the stereo form of this (a second channel that is a pure delay
       against a 640-tap filter). NOTE THAT GOING MONO DOES NOT SAVE YOU: measured
       here, `sources = [x, 0.5*x]` in mono reproduces it exactly.
       EXCLUDED BY: independent RNG streams per source AND -- the load-bearing part --
       an independent white-noise floor at -60 dB in each source before normalisation,
       which is what puts independent energy in every band and makes the Gram full
       rank. Exact analogue of measure.py's `_STEREO_DECORR`, for the same reason.

    3. Odd sample rates. `myPemoAnalysisFilterBank` resamples by `round(1.5*fs)/fs`.
       For even `fs` that reduces to up=3/down=2. For `fs = 11025` it reduces to
       16538/11025 with `gcd == 1`: a ~330k-tap filter and a 16538*n intermediate.
       A `st.integers(8000, 48000)` strategy appears to hang on its first odd draw.
       EXCLUDED BY: `st.sampled_from((8000, 16000))`.

    4. Amplitude. The decomposition is homogeneous in the input gain, so level is not
       an interesting axis and a tiny one just turns every comparison into a denormal
       contest. Both sources are normalised to a fixed RMS; hypothesis varies content.

WHY THE TOLERANCES ARE PER COMPONENT, AND PER BACKEND
    Same reason `test_matlab_regression.py` and `test_reference_vs_peass_parity.py`
    are: `artifacts` is the residual of an ill-conditioned least-squares stage
    (minimum Gram eigenvalue ~3.4e-10, see ARCHIVE.md), so a 1-ULP difference upstream
    arrives there ~1e6 larger. Measured worst-case spread across the four components on
    the numpy backend is nearly FIVE orders of magnitude (2.1e-15 on `true_target`
    against 9.8e-11 on `artifacts`, 120 draws, 2026-08-17), and the bars that follow it
    span six. One bar is therefore meaningless: set by `artifacts` it would let
    `true_target` rot six decades unnoticed.

    And per BACKEND -- but for a far smaller reason than when this file landed. It then
    read "torch disagrees with the reference by ~2e-6 relative where numpy disagrees by
    ~1e-15", nine orders of magnitude, and blamed the float32 `torch.fft.fftfreq` grid.
    That grid was fixed (float64, 6e0162d) and the gammatone modulation phase was then
    range-reduced, and together they closed almost the whole gap. Re-measured
    2026-08-17 over 120 draws, torch meets the SAME per-component bars as numpy on
    three of the four components; only `true_target` still needs a looser one, and only
    by ~100x (2.0e-13 against numpy's 2.1e-15, which is machine epsilon). The two
    tables stay separate because that last factor is real and because the backends can
    regress independently -- not because torch is broadly less accurate. It is not, any
    more.

WHY THE BARS ARE PROVED LIVE
    A tolerance that cannot fail is worse than no tolerance, because it reads as
    coverage. Every one of the eight bars below is mutation-proved by
    `test_every_tolerance_bar_is_live` at the bottom of this file, which perturbs each
    component by a scaled fraction of its own bar and asserts the comparison passes
    just inside and fails just outside. The resolving power of a bar B is a relative
    gain error of ~B on that one component, and that test is what keeps it true.

RUNNING IT
    `pytest -m oracle` is the opt-in, and it is the ONLY way these tests run: the
    `-m "not oracle"` in `pyproject.toml`'s `addopts` deselects them in every other
    invocation, and a command-line `-m` replaces that one rather than adding to it
    (pytest's `-m` is single-valued, last wins).

    pytest -m oracle                                  # the default 40 examples, ~1 min
    PEASS_ORACLE_EXAMPLES=5 pytest -m oracle          # smoke it first
    PEASS_ORACLE_EXAMPLES=250 pytest -m oracle -n auto        # a long soak
    pytest -m oracle -k "linear_mix or fir_related"   # just the characterization pins
    pytest -m oracle -k bar_is_live                   # just the mutation proof, ~3 s
    pytest -m ""                                      # clear the filter entirely

    `--hypothesis-show-statistics` PRINTS NOTHING for this file, and that is expected.
    The `@given` callable is `_check_one_drawn_case`, which the collected test drives
    by hand so that it can check the example counts afterwards; hypothesis' pytest
    plugin gates statistics on `is_hypothesis_test(item.obj)`, and the collected item
    is the plain wrapper. The one thing those statistics were used for here -- the
    invalid-example count -- is now ASSERTED instead, by the acceptance-rate floor in
    that wrapper, so it fails the run rather than waiting to be read. Falsifying
    examples, shrinking and the `@reproduce_failure` blob are unaffected: they come
    from hypothesis' core reporting, not the plugin, and are verified working.

    PowerShell: `$env:PEASS_ORACLE_EXAMPLES = "5"; pytest -m oracle`
    PyCharm (verified 2026-08-17, and the wart is real): running a single test by node
    ID alone reports `0 collected, 1 deselected` -- `addopts` applies to a node ID too,
    and pytest deselects by marker after collecting the ID. Put `-m oracle` (or `-m ""`)
    in the run configuration's Additional Arguments or nothing runs.

    `hypothesis` is a dev/test extra (`pip install -e ".[test]"`, or it is already in
    requirements.txt); the whole module skips cleanly without it.
"""

import os
import warnings

import numpy as np
import pytest
import scipy.signal

from peass import decompose_distortion_components
from peass.config import DecompositionConfiguration
from reference.extractDistortionComponents import extractDistortionComponents

pytest.importorskip(
    "hypothesis",
    reason="hypothesis is a dev/test extra; install with `pip install -e \".[test]\"`",
)

from hypothesis import HealthCheck  # noqa: E402
from hypothesis import assume  # noqa: E402
from hypothesis import given  # noqa: E402
from hypothesis import settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised by the remove-torch CI leg
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

_COMPONENTS = ("true_target", "target_distortion", "interference", "artifacts")

# Relative L2 error, ||peass - reference|| / ||reference||, per component, NumPy
# backend.
#
# MEASURED 2026-08-17 with `.scratch/hypothesis-2026-08-17/prove_oracle_bars.py`, which
# imports THIS module, so the strategy, the builder and the bars measured are the ones
# that ship. Three independent runs of 40 draws (120 total) over the strategy box
# (mono, 2 sources, 8/16 kHz, 0.25-0.75 s):
#
#     component            run 1     run 2     run 3     WORST     bar    margin
#     true_target        2.06e-15  1.96e-15  1.94e-15  2.06e-15  1e-13     48.5x
#     target_distortion  3.75e-11  8.47e-12  3.52e-11  3.75e-11   1e-8    266.5x
#     interference       1.12e-11  7.00e-11  3.86e-11  7.00e-11   1e-8    142.8x
#     artifacts          3.54e-11  4.59e-11  9.83e-11  9.83e-11   1e-7   1017.7x
#
# READ THAT COLUMN CORRECTLY. `WORST` is the maximum over 120 draws: a max-of-N
# OBSERVATION, not a bound. The per-run columns are printed precisely to show how
# unstable it is -- `interference` spans 1.12e-11 to 7.00e-11 across three runs of the
# identical strategy, a factor of 6.3, and a different component is worst in each run.
# An earlier revision of this comment carried ONE run's figures under the heading
# "worst any draw"; two independent re-runs each exceeded it, on different components.
# Three runs do not fix that, they only shrink the understatement. So each bar carries
# an explicit MARGIN over the observation (48x-1018x above), sized so that a run
# several times worse than any yet seen still passes.
#
# NO BAR HAS EVER BEEN BREACHED, in any measurement of this suite: 2 x 14 draws before
# the torch solver ridge, and 3 x 40 draws after it and after P5. The four values below
# are UNCHANGED from the day the file landed -- only the evidence under them is new.
#
# These happen to be the same four numbers as
# `test_reference_vs_peass_parity.py::_MAX_RELATIVE_L2`, which is corroboration rather
# than coincidence -- two different input regimes landing on the same floor. They are
# re-declared rather than imported ON PURPOSE: that file's bars are justified by one
# 5 s stereo clip, these by a fuzzed mono box, and coupling them would mean tightening
# one silently tightens the other and makes this suite flaky.
#
# All four are mutation-proved live by `test_every_tolerance_bar_is_live` below.
_MAX_RELATIVE_L2_NUMPY = {
    "true_target": 1e-13,
    "target_distortion": 1e-8,
    "interference": 1e-8,
    "artifacts": 1e-7,
}

# Same quantity, torch backend. GRADED PER COMPONENT since 2026-08-17; it used to be
# flat at 1e-3 and that is no longer defensible.
#
# MEASURED the same way, in the same three 40-draw runs as the numpy table above:
#
#     component            run 1     run 2     run 3     WORST     bar    margin
#     true_target        9.57e-14  1.29e-13  1.99e-13  1.99e-13  1e-10    503.8x
#     target_distortion  5.07e-11  9.12e-12  4.69e-11  5.07e-11   1e-8    197.2x
#     interference       1.60e-11  1.45e-10  5.46e-11  1.45e-10   1e-8     69.1x
#     artifacts          4.75e-11  9.60e-11  1.27e-10  1.27e-10   1e-7    790.0x
#
# Same reading rule as the numpy table: `WORST` is a max-of-120 OBSERVATION, not a
# bound, the per-run columns show the spread (up to 9.0x, on `interference`), and the
# margin is what does the work. No torch bar has ever been breached either.
#
# WHY THE FLAT 1e-3 IS GONE. It was set when torch sat uniformly ~2-4e-6 off the
# reference -- a common-mode upstream error, correctly blamed at the time on the
# float32 `torch.fft.fftfreq` grid. That grid was fixed (float64, 6e0162d), and the
# gammatone modulation phase was then range-reduced, removing a further ~4e-11 error.
# Torch's disagreement with the reference fell four to five decades and now sits at the
# SAME MAGNITUDE as numpy's. Measured against the old 1e-3 the margins came back at
# 5.0e9x, 2.0e7x, 6.9e6x and 7.9e6x -- seven to ten decades of slack, a bar that could
# not fail for anything short of total breakage. That is vacuous coverage, which is the
# one thing this file must not ship.
#
# The replacements are numpy's own bars, unchanged, on three of the four components:
# torch now reproduces the reference as well as numpy does on `target_distortion`,
# `interference` and `artifacts`. Only `true_target` still needs a looser bar, and only
# by ~100x -- torch 2.0e-13 against numpy 2.1e-15, the latter being machine epsilon on
# the one component that is not an output of the ill-conditioned solve. That surviving
# factor is the honest reason these two tables are still SEPARATE rather than collapsed
# into one, and it is the number to watch if torch accuracy is worked on again.
#
# `true_target` is bounded at 1e-10 rather than 1e-11 on purpose. 1e-11 would still
# clear the observation, but only by 50x -- the very floor of the 50x-500x band this
# repo uses -- against an observed run-to-run spread of 2.1x on this component. The
# eighth decade of tightening buys little and costs flake risk.
#
# All four are mutation-proved live by `test_every_tolerance_bar_is_live` below.
_MAX_RELATIVE_L2_TORCH = {
    "true_target": 1e-10,
    "target_distortion": 1e-8,
    "interference": 1e-8,
    "artifacts": 1e-7,
}

# A component whose energy falls below this fraction of the estimate's is not being
# compared, it is being compared to rounding noise -- that is the linear-mix trap.
#
# MEASURED 2026-08-17 over 120 draws of the shipped strategy (3 runs x 40), component
# RMS / estimate RMS, minimum per component:
#
#     true_target 5.92e-1 | target_distortion 7.69e-2 | interference 6.10e-2 |
#     artifacts 7.74e-2
#
# These are min-of-N OBSERVATIONS, not bounds, and this constant has now been bitten by
# that twice. An early revision quoted a single "healthy minimum" of 1.39e-1 that its
# own harness did not reproduce; the 14-draw correction that replaced it recorded
# 1.11e-1 for `interference`, and 120 draws duly went 1.8x lower still, to 6.10e-2. A
# sweep of the strategy CORNERS (drive 1.0/6.0 x artifact_db -40/-25 x leakage -12/0)
# bottoms out at 1.71e-1, which is HIGHER than the fuzzed minimum -- the corners are not
# where this quantity is smallest, so do not treat that sweep as the bound either.
#
# What is robust, and what the floor rests on, is the SEPARATION rather than any one
# figure: the linear-mix trap collapses target_distortion and artifacts to 1.24e-12 and
# 1.21e-12, so the floor sits ~4.8 decades below the worst healthy component observed
# and ~5.9 decades above the trap. It has ample room to absorb another 1.8x drift in
# either direction. Re-measure with
# `.scratch/hypothesis-2026-08-17/prove_oracle_bars.py` if you change the builder.
#
# It is an `assert`, not an `assume`: if it fires, the STRATEGY is broken and that must
# be loud. Note the linear-mix characterization test below deliberately CONSTRUCTS an
# input that breaches this floor and asserts that it does -- that is the pin, not a
# contradiction.
_DEGENERACY_FLOOR = 1e-6

# Pre-oracle preconditions, both cheap. They exist as executable documentation of the
# two traps -- if a future edit to the builder reintroduces one, hypothesis filters the
# example out instead of the suite going red for the wrong reason, and the acceptance
# rate that `_MIN_VALID_EXAMPLE_FRACTION` guards is what makes the filtering visible.
#
# MEASURED 2026-08-17 over 120 draws (3 runs x 40): the nonlinear residual bottoms out
# at 3.91e-2 against the 1e-3 bar, 39x clear, and every run completed 40 of 40 draws,
# so neither `assume` fired even once.
#
# The cross-correlation margin is the thin one and is deliberately not tightened:
# measured peak 2.86e-1 against the 0.5 bar, only 1.75x clear, because two independent
# partial stacks can land partials near each other by chance. So this one CAN fire on a
# long run. That is the designed behaviour -- it is an `assume`, so a correlated draw is
# filtered rather than failed, and filtering is the right answer for an input whose Gram
# is genuinely closer to singular. Do not "fix" an occasional invalid-example count by
# raising the bar; raising it is what lets trap 2 back in. If it starts firing OFTEN,
# `_MIN_VALID_EXAMPLE_FRACTION` below is what turns that from silence into a failure.
_MIN_NONLINEAR_RESIDUAL = 1e-3
_MAX_SOURCE_CROSS_CORRELATION = 0.5


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

_EPS = 1e-300

# Even rates only -- see the module docstring, exclusion 3.
_SAMPLE_RATES_HZ = (8000, 16000)

# 0.25-0.75 s. Short is BOTH cheaper and (measured) tighter: the worst
# reference-vs-peass disagreement anywhere came from the LONGEST draw. It is also the
# regime the reflection-padding TODO item is about and the one no gold WAV covers.
#
# Note this box straddles a structural threshold on purpose. `LSDecompose_tv`'s frame
# loop needs `Lb + flen - 1 >= Lw/2`, and because `Lw ~= 133-141` samples in every band
# while the decimated band length grows at ~59 samples per second of audio regardless
# of `fs`, the 6-7 LOWEST bands get ZERO frames below ~0.957 s and their projection is
# identically zero. Measured, the two implementations agree there anyway (3.2e-11 at
# 0.5 s / 8 kHz), so this regime is INCLUDED rather than excluded -- it is exactly
# where an oracle has never been consulted.
_MIN_DURATION_SECONDS = 0.25
_MAX_DURATION_SECONDS = 0.75

_SOURCE_RMS = 0.2
_SOURCE_NOISE_FLOOR_DB = -60.0  # load-bearing; see exclusion 2
_ARTIFACT_CUTOFF_HZ = 4000.0  # matches benchmarks/measure.py's _NOISE_CUTOFF_HZ
_PARTIAL_LOW_HZ = 60.0
_PARTIAL_HIGH_FRACTION = 0.40  # of the sample rate, i.e. 0.8 of Nyquist


@st.composite
def _decomposition_cases(draw):
    """A DESCRIPTION of an input, not the input itself.

    Drawing 20000 raw float64s with `hypothesis.extra.numpy` would be slow to
    generate, miserable to shrink, and would spend almost every example inside one
    degenerate corner or another. Drawing nine scalars that a deterministic builder
    turns into audio keeps the search space small, the shrinks readable, and every
    draw inside the well-posed region by construction.
    """
    sampling_frequency_hz = draw(st.sampled_from(_SAMPLE_RATES_HZ))
    return {
        "sampling_frequency_hz": sampling_frequency_hz,
        "n_samples": draw(st.integers(
            min_value=int(round(_MIN_DURATION_SECONDS * sampling_frequency_hz)),
            max_value=int(round(_MAX_DURATION_SECONDS * sampling_frequency_hz)),
        )),
        "seed": draw(st.integers(min_value=0, max_value=2 ** 32 - 1)),
        "partial_count": draw(st.integers(min_value=2, max_value=6)),
        "target_tilt": draw(st.floats(min_value=-1.0, max_value=1.0,
                                      allow_nan=False, allow_infinity=False)),
        "interferer_tilt": draw(st.floats(min_value=-1.0, max_value=1.0,
                                          allow_nan=False, allow_infinity=False)),
        # How much of the interferer survives into the estimate: 0 dB is a total
        # separation failure, -12 dB a good separation.
        "leakage_db": draw(st.floats(min_value=-12.0, max_value=0.0,
                                     allow_nan=False, allow_infinity=False)),
        # tanh drive. 1.0 is the mildest nonlinearity the strategy will produce; it is
        # NOT 0, because 0 is the linear trap.
        "drive": draw(st.floats(min_value=1.0, max_value=6.0,
                                allow_nan=False, allow_infinity=False)),
        # Additive shaped noise, relative to the mix RMS. measure.py pins this at
        # -40 dB; here it is a strategy parameter over a range that brackets it.
        "artifact_db": draw(st.floats(min_value=-40.0, max_value=-25.0,
                                      allow_nan=False, allow_infinity=False)),
    }


def _root_mean_square(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _source_signal(rng, n_samples, sampling_frequency_hz, partial_count, spectral_tilt):
    """An amplitude-modulated partial stack, plus coloured noise, plus a noise floor.

    The partials give band-limited structure the least-squares fit can actually latch
    onto; the amplitude modulation makes it non-stationary so the time-varying framing
    is exercised rather than degenerating to a single stationary fit; the coloured
    noise keeps it from being purely tonal.

    The last line is the one that matters. Without an INDEPENDENT white floor in each
    source there is no guarantee that both sources carry energy in every band, and a
    band where one source is numerically silent -- or worse, where the two are FIR
    images of one another -- is a rank-deficient Gram whose minimum-norm solution the
    two implementations are free to disagree about by 40%. See exclusion 2 in the
    module docstring, and `benchmarks/measure.py`'s `_STEREO_DECORR` note, which is the
    same lesson on the channel axis instead of the source axis.
    """
    time_seconds = np.arange(n_samples, dtype=np.float64) / sampling_frequency_hz
    signal = np.zeros(n_samples, dtype=np.float64)

    highest_hz = _PARTIAL_HIGH_FRACTION * sampling_frequency_hz
    for _ in range(partial_count):
        frequency_hz = float(np.exp(rng.uniform(np.log(_PARTIAL_LOW_HZ), np.log(highest_hz))))
        amplitude = (frequency_hz / 1000.0) ** spectral_tilt
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        depth = float(rng.uniform(0.0, 0.7))
        modulation_hz = float(rng.uniform(0.5, 5.0))
        envelope = 1.0 - depth * (0.5 + 0.5 * np.cos(2.0 * np.pi * modulation_hz * time_seconds + phase))
        signal += amplitude * envelope * np.sin(2.0 * np.pi * frequency_hz * time_seconds + phase)

    numerator, denominator = scipy.signal.butter(4, 0.35, btype="low")
    coloured = scipy.signal.lfilter(numerator, denominator, rng.standard_normal(n_samples))

    signal = signal / (_root_mean_square(signal) + _EPS)
    signal = signal + 0.5 * coloured / (_root_mean_square(coloured) + _EPS)
    signal = signal + (10.0 ** (_SOURCE_NOISE_FLOOR_DB / 20.0)) * rng.standard_normal(n_samples)

    # Fixed RMS: level is not an interesting axis (the decomposition is homogeneous in
    # it) and a tiny one turns every comparison into a denormal contest.
    return (_SOURCE_RMS * signal / (_root_mean_square(signal) + _EPS))[:, np.newaxis]


def _build_case(case):
    """Turn a drawn description into `(sources, estimate, mixture)`, all (N, 1)."""
    sampling_frequency_hz = case["sampling_frequency_hz"]
    n_samples = case["n_samples"]
    seed = case["seed"]

    # Adjacent integer seeds are fine: `default_rng(int)` routes through SeedSequence,
    # whose entropy mixing makes the three streams independent.
    target = _source_signal(np.random.default_rng(seed), n_samples,
                            sampling_frequency_hz, case["partial_count"],
                            case["target_tilt"])
    interferer = _source_signal(np.random.default_rng(seed + 1), n_samples,
                                sampling_frequency_hz, case["partial_count"],
                                case["interferer_tilt"])

    mixture = target + (10.0 ** (case["leakage_db"] / 20.0)) * interferer

    # ARCHIVE.md fixes the tanh form; benchmarks/measure.py's FROZEN CONVENTIONS block
    # pins `tanh(3*mix)/3` plus shaped noise. `drive` generalises the 3.
    drive = case["drive"]
    estimate = np.tanh(drive * mixture) / drive

    cutoff_hz = min(_ARTIFACT_CUTOFF_HZ, 0.45 * sampling_frequency_hz)
    numerator, denominator = scipy.signal.butter(
        4, cutoff_hz / (sampling_frequency_hz / 2.0), btype="low")
    noise = scipy.signal.lfilter(
        numerator, denominator,
        np.random.default_rng(seed + 2).standard_normal(mixture.shape), axis=0)
    noise = noise * (10.0 ** (case["artifact_db"] / 20.0)) * _root_mean_square(mixture) \
        / (_root_mean_square(noise) + _EPS)

    return [target, interferer], estimate + noise, mixture


def _relative_nonlinear_residual(estimate, mixture):
    """How far the estimate sits off the raw mixture. A cheap, necessary (not
    sufficient) proxy for 'the estimate is not in the span of the sources'."""
    return float(np.linalg.norm(estimate - mixture) / (np.linalg.norm(estimate) + _EPS))


def _peak_normalised_cross_correlation(first, second):
    """Peak |cross-correlation| over all lags, normalised to [0, 1]. FFT-based, so
    O(N log N) -- cheap enough to run as a precondition on every example."""
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    first = first - first.mean()
    second = second - second.mean()
    length = 1 << int(np.ceil(np.log2(2 * len(first))))
    correlation = np.fft.irfft(
        np.fft.rfft(first, length) * np.conj(np.fft.rfft(second, length)), length)
    denominator = np.linalg.norm(first) * np.linalg.norm(second) + _EPS
    return float(np.max(np.abs(correlation)) / denominator)


# ---------------------------------------------------------------------------
# Running the two sides
# ---------------------------------------------------------------------------

# `reference/`'s own defaults, restated so the `peass/` call is provably configured the
# same way: extractDistortionComponents' defaultOptions has frameLength .5,
# filterLength .04, shadeInMs 10, shadeOutMs 10, FLAG_2PROJ false, segmentationFactor 1.
_CONFIGURATION = DecompositionConfiguration(
    shade_in_milliseconds=10.0,
    shade_out_milliseconds=10.0,
)


def _reference_components(sources, estimate, sampling_frequency_hz):
    """The oracle. ~0.3-1.2 s in this file's strategy box. Everything else here is
    cheap by comparison -- the two `peass/` calls together cost a fraction of it.

    Rough wall clock, a budget estimate only. NOT a `benchmarks/ab.py` measurement and
    not quotable as a perf number.
    """
    return extractDistortionComponents(sources, estimate, float(sampling_frequency_hz))[:4]


def _peass_components(sources, estimate, sampling_frequency_hz, backend):
    if backend == "torch":
        sources = [torch.tensor(s, dtype=torch.float64) for s in sources]
        estimate = torch.tensor(estimate, dtype=torch.float64)
    waveforms = decompose_distortion_components(
        source_files=sources,
        estimate_file=estimate,
        configuration=_CONFIGURATION,
        sampling_frequency_hz=float(sampling_frequency_hz),
    ).waveforms
    return [waveforms.true_target, waveforms.target_distortion,
            waveforms.interference, waveforms.artifacts]


def _as_numpy(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _relative_l2(produced, reference):
    return float(np.linalg.norm(produced - reference) / (np.linalg.norm(reference) + _EPS))


_DEFAULT_MAX_EXAMPLES = 40


def _max_examples():
    """`PEASS_ORACLE_EXAMPLES`, parsed defensively.

    This runs at IMPORT time (a `settings` object is built at module scope), so anything
    that raises here is a COLLECTION error and takes down the whole session, not just this
    file. A malformed or out-of-range value therefore falls back to the default with a
    warning rather than raising: an unusable env var must not be able to break `pytest`
    for tests that have nothing to do with this one.

    Fallback: `_DEFAULT_MAX_EXAMPLES` (40). Values below 1 are invalid to hypothesis
    (`max_examples=0 must be at least one`) and are clamped up to 1, not silently dropped,
    so `PEASS_ORACLE_EXAMPLES=0` still means "run as little as possible".
    """
    raw = os.environ.get("PEASS_ORACLE_EXAMPLES")
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_EXAMPLES
    try:
        value = int(raw)
    except ValueError:
        warnings.warn(
            f"PEASS_ORACLE_EXAMPLES={raw!r} is not an integer; falling back to "
            f"{_DEFAULT_MAX_EXAMPLES}.",
            RuntimeWarning, stacklevel=2,
        )
        return _DEFAULT_MAX_EXAMPLES
    if value < 1:
        warnings.warn(
            f"PEASS_ORACLE_EXAMPLES={value} is below hypothesis' minimum of 1; using 1.",
            RuntimeWarning, stacklevel=2,
        )
        return 1
    return value


_MAX_EXAMPLES = _max_examples()

# `deadline=None` is mandatory: hypothesis' 200 ms default would fail every example.
# `too_slow` and `filter_too_much` are suppressed because the health checks time
# generation against a budget this test blows through by three orders of magnitude for
# reasons that have nothing to do with the strategy.
# `print_blob=True` so a failure comes with a `@reproduce_failure` blob that can be
# pasted straight into a local run.
#
# `filter_too_much` is suppressed, which means a strategy that drifts into filtering MOST
# (not all) of its draws would pass silently on a handful of real examples. Total
# filtering still raises `Unsatisfiable`, but partial does not -- MEASURED on 6.165.9, a
# strategy that filters 95% of its draws PASSES, green and quiet, having run 2 real
# examples out of 40, and nothing in the output distinguishes that from a healthy run.
# `_MIN_VALID_EXAMPLE_FRACTION` below is what turns that case into a failure. Do not
# raise `_MAX_SOURCE_CROSS_CORRELATION` to make an invalid count go away; raising it is
# what lets trap 2 back in.
#
# `database=None` is deliberate (decision 2026-08-17). Without it hypothesis persists
# failing examples under `.hypothesis/` and replays them FIRST on the next run, so the
# `PEASS_ORACLE_EXAMPLES=5` smoke run this file's docstring recommends would spend its
# whole budget re-running one stale example from a previous session instead of exploring.
# The trade is real -- a failure does not auto-replay -- and it is paid for by
# `print_blob=True`, which prints a `@reproduce_failure` blob that reproduces it exactly.
_ORACLE_SETTINGS = settings(
    max_examples=_MAX_EXAMPLES,
    deadline=None,
    print_blob=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

# The vacuity floor. With `filter_too_much` suppressed (above), a strategy that drifts
# into filtering nearly all of its draws reports a clean pass having tested almost
# nothing. This is what makes that loud instead.
#
# THE OBVIOUS DETECTOR DOES NOT WORK, and the reason is worth knowing before anyone
# "simplifies" this. Counting the examples that reach the assertions and requiring, say,
# half of `max_examples` looks sufficient and is nearly blind: MEASURED 2026-08-17,
# forcing 95% of draws to be filtered still produced the FULL 40 valid examples, because
# hypothesis compensates for filtering by simply drawing more. A count-of-valid floor
# only fires once the filtering is severe enough to exhaust hypothesis' own generation
# budget, by which point the problem has been present for a long time.
#
# The quantity that is actually diagnostic is the ACCEPTANCE RATE -- examples that got
# past both `assume`s, over examples entered -- because `assume` raises from inside the
# property body, so both counts are visible from here and neither is affected by
# hypothesis re-drawing. Under the same forced 95% filtering the rate reads ~5% and
# fails immediately. Both checks are kept: the rate is the detector, the count is the
# backstop for the case where hypothesis gives up early.
#
# Sized at one half, which is enormous headroom: 3 runs x 40 draws in the 2026-08-17
# measurement accepted 40 of 40 every time, a 100% rate -- neither `assume` fired once.
# It is not tighter because the cross-correlation precondition is only 1.75x clear and
# is DESIGNED to be allowed to fire; this floor is meant to catch structural drift, not
# to police the odd filtered draw.
_MIN_VALID_EXAMPLE_FRACTION = 0.5

# The rate check needs a denominator worth dividing by. Below this many draws only the
# absolute floor applies, so that `PEASS_ORACLE_EXAMPLES=1` stays a legal smoke run and
# a single filtered draw in a 2-draw run cannot fail it.
_MIN_DRAWS_FOR_RATE_CHECK = 10

# Both incremented by the property, reset and checked by the test that drives it.
_examples_drawn = 0
_examples_reaching_the_assertions = 0


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------
#
# Every test here carries `oracle` and ONLY `oracle` -- deliberately. This is the one
# file in `tests/regression/` without a `regression` marker, so that `pytest -m
# regression` does not silently pull a live-oracle fuzz run into a routine invocation.
# Do not "tidy up" by adding `regression` to these.

def _compare_against_reference(backend, bars, reference, produced, case):
    """Check one backend's four components against the oracle's.

    Returns a list of human-readable lines, one per component that exceeded its bar;
    an empty list means every bar held. The three STRUCTURAL checks (length, ordering,
    zero tail) stay `assert`s inside here, because none of them is a tolerance -- they
    are either true or the comparison itself is meaningless.

    Extracted from the property on purpose rather than inlined: this is the code
    `test_every_tolerance_bar_is_live` mutation-proves. If the liveness proof called a
    re-implementation of the comparison, it would prove that re-implementation is live
    and say nothing about the one that ships.
    """
    failures = []
    for label, reference_component, produced_component in zip(
            _COMPONENTS, reference, produced):
        # `peass/` pads back out to the input length, `reference/` returns what
        # MATLAB's decimate/interpolate/trim arithmetic yields. Compare the
        # overlap, and pin the excess as exactly zero -- which
        # test_reference_vs_peass_parity.py asserts for one clip and this
        # generalises across the whole strategy box.
        assert len(produced_component) == case["n_samples"], (
            f"{backend}/{label}: peass returned {len(produced_component)} samples "
            f"for a {case['n_samples']}-sample input"
        )
        assert len(produced_component) >= len(reference_component), (
            f"{backend}/{label}: peass returned {len(produced_component)} samples "
            f"but the reference returned MORE ({len(reference_component)}), which "
            f"inverts the expected relation. `reference/` returns only what MATLAB's "
            f"decimate/interpolate/trim arithmetic in `myPemoAnalysisFilterBank` "
            f"yields, which is always SHORTER than the input -- the filterbank "
            f"resamples by round(1.5*fs)/fs, decimates per band, and trims the "
            f"filter transients, so a 4000-sample input comes back as 3413 samples "
            f"(587 short, ~15%). `peass/` pads back out to the full input length, so "
            f"produced >= reference always holds. If this fires, either `peass/` "
            f"stopped padding or the reference's trim arithmetic changed. Case: "
            f"{case}"
        )
        tail = produced_component[len(reference_component):]
        assert np.count_nonzero(tail) == 0, (
            f"{backend}/{label}: {np.count_nonzero(tail)} nonzero samples (peak "
            f"{np.max(np.abs(tail)):.3e}) past the length MATLAB produces. The "
            f"tail is padding and must stay exactly zero. Case: {case}"
        )

        length = len(reference_component)
        error = _relative_l2(produced_component[:length, 0],
                             reference_component[:length, 0])
        if not error < bars[label]:
            failures.append(
                f"  {backend}/{label}: relative L2 {error:.3e} exceeds the "
                f"{bars[label]:.0e} bar"
            )
    return failures


@_ORACLE_SETTINGS
@given(case=_decomposition_cases())
def _check_one_drawn_case(case):
    """The property body. Driven by
    `test_every_backend_reproduces_the_reference_decomposition` below rather than
    collected directly, so that the number of examples which actually reached these
    assertions can be checked once `@given` has finished -- see
    `_MIN_VALID_EXAMPLE_FRACTION`.
    """
    global _examples_drawn, _examples_reaching_the_assertions
    # Counted before the `assume`s: this is the denominator of the acceptance rate.
    _examples_drawn += 1

    sources, estimate, mixture = _build_case(case)

    # --- preconditions, BEFORE the expensive call -------------------------------
    # Trap 1: an estimate inside the span of the sources. Does not fire as written
    # (measured minimum 3.9e-2, bar 1e-3); here so that a future weakening of
    # `_build_case` is filtered out rather than reported as a backend bug.
    assume(_relative_nonlinear_residual(estimate, mixture) > _MIN_NONLINEAR_RESIDUAL)
    # Trap 2: FIR-related sources. This is the one that can genuinely fire (measured
    # peak 2.9e-1 against the 0.5 bar) -- see the constant's comment.
    assume(_peak_normalised_cross_correlation(sources[0], sources[1])
           < _MAX_SOURCE_CROSS_CORRELATION)

    # Past both `assume`s, so this example WILL reach the assertions below. Counted
    # here rather than at the end because an example that fails an assertion still
    # reached it. See `_MIN_VALID_EXAMPLE_FRACTION`.
    _examples_reaching_the_assertions += 1

    reference = [np.asarray(component, dtype=np.float64)
                 for component in _reference_components(
                     sources, estimate, case["sampling_frequency_hz"])]

    # --- guard: is this comparison meaningful at all? ---------------------------
    # An `assert`, not an `assume`. If a component came out at the float64 noise floor
    # the input was degenerate, which means the strategy is broken, and a broken
    # strategy must be loud rather than silently filtered into a green run.
    estimate_rms = _root_mean_square(estimate)
    for label, component in zip(_COMPONENTS, reference):
        component_rms = _root_mean_square(component)
        assert component_rms > _DEGENERACY_FLOOR * estimate_rms, (
            f"STRATEGY BUG, not a backend bug: the reference's {label} came out at "
            f"{component_rms / (estimate_rms + _EPS):.3e} of the estimate RMS, below "
            f"the {_DEGENERACY_FLOOR:.0e} floor. A component at the noise floor means "
            f"the drawn input is degenerate -- almost certainly an estimate that lies "
            f"in the span of the sources -- and comparing it to the backends is "
            f"noise against noise. Fix `_build_case`, do not widen the tolerance. "
            f"Case: {case}"
        )

    backends = [("numpy", _MAX_RELATIVE_L2_NUMPY)]
    if _HAS_TORCH:
        backends.append(("torch", _MAX_RELATIVE_L2_TORCH))

    failures = []
    for backend, bars in backends:
        produced = [_as_numpy(component) for component in _peass_components(
            sources, estimate, case["sampling_frequency_hz"], backend)]
        failures.extend(
            _compare_against_reference(backend, bars, reference, produced, case))

    assert not failures, (
        "peass/ and reference/ disagree on a fuzzed input:\n"
        + "\n".join(failures)
        + f"\n  case: {case}\n"
        "The reference is a line-by-line transcription of MATLAB PEASS v2.0.1 and is "
        "validated against the gold WAVs by test_reference_transcription.py, so the "
        "prior is strongly that peass/ moved. Reproduce with the printed hypothesis "
        "blob, then diff the per-band subband output rather than the synthesised "
        "waveform -- the decomposition is band-independent and the failing band is "
        "usually the whole story."
    )


@pytest.mark.oracle
def test_every_backend_reproduces_the_reference_decomposition():
    """The one property this file exists for: on an arbitrary well-posed input, the
    fast backends produce the SAME four components as the MATLAB transcription.

    Both backends are checked from a SINGLE oracle call. That is the cost decision in
    this file -- a `parametrize` over backend would double the oracle bill for no extra
    coverage, since the oracle does not depend on which backend is being checked.

    This wrapper exists only so that the `@given` run can be followed by a VACUITY
    check. `filter_too_much` is suppressed, so a strategy that starts filtering nearly
    everything would otherwise report a pass having tested almost nothing -- measured,
    95% filtering runs 2 real examples out of 40 and is green. Counting the examples
    that got past both `assume`s and requiring a floor is what makes that visible.
    """
    global _examples_drawn, _examples_reaching_the_assertions
    _examples_drawn = 0
    _examples_reaching_the_assertions = 0

    _check_one_drawn_case()

    drawn = _examples_drawn
    reached = _examples_reaching_the_assertions
    diagnosis = (
        f"Something is filtering the strategy: most likely `_build_case` changed and "
        f"the sources are now correlated enough to trip "
        f"`_MAX_SOURCE_CROSS_CORRELATION` ({_MAX_SOURCE_CROSS_CORRELATION}), or the "
        f"estimate drifted toward the span of the sources and is tripping "
        f"`_MIN_NONLINEAR_RESIDUAL` ({_MIN_NONLINEAR_RESIDUAL:.0e}). Print those two "
        f"quantities per draw to see which, then fix the BUILDER. "
        f"(`--hypothesis-show-statistics` will not help: it reports nothing for this "
        f"file, see the module docstring.) "
        f"Do NOT widen either precondition to make this go away -- they "
        f"are what keep the two degenerate traps out, and widening them readmits the "
        f"trap instead of removing the filtering."
    )

    # The detector. See `_MIN_VALID_EXAMPLE_FRACTION`: hypothesis compensates for
    # filtering by drawing more, so the rate is what reveals it, not the count.
    if drawn >= _MIN_DRAWS_FOR_RATE_CHECK:
        assert reached >= _MIN_VALID_EXAMPLE_FRACTION * drawn, (
            f"VACUOUS RUN, not a backend bug: only {reached} of {drawn} examples "
            f"entered got past the two preconditions, an acceptance rate of "
            f"{reached / drawn:.1%} against a floor of "
            f"{_MIN_VALID_EXAMPLE_FRACTION:.0%}. Everything this run did assert "
            f"passed, and without this check it would have reported a clean pass "
            f"while most of the search was being thrown away.\n" + diagnosis
        )

    # The backstop, for the case where hypothesis gives up generating early enough
    # that the rate above still looks acceptable.
    floor = max(1, int(_MIN_VALID_EXAMPLE_FRACTION * _MAX_EXAMPLES))
    assert reached >= floor, (
        f"VACUOUS RUN, not a backend bug: only {reached} examples reached the "
        f"assertions out of a requested {_MAX_EXAMPLES}, below the floor of {floor} "
        f"({_MIN_VALID_EXAMPLE_FRACTION:.0%} of `max_examples`); hypothesis entered "
        f"the property {drawn} times to get there.\n" + diagnosis
    )


# ---------------------------------------------------------------------------
# Characterization pins for the two excluded traps
# ---------------------------------------------------------------------------
#
# These are NOT fuzzed -- one fixed input each, one oracle call each. They exist so
# that the exclusions above are testable claims rather than comments, and so that
# anyone who "simplifies" the strategy back into a trap gets a red test that says
# exactly which trap.

_TRAP_SAMPLE_RATE_HZ = 8000
_TRAP_N_SAMPLES = 10000
_TRAP_SEED = 11


def _trap_sources():
    target = _source_signal(np.random.default_rng(_TRAP_SEED), _TRAP_N_SAMPLES,
                            _TRAP_SAMPLE_RATE_HZ, 4, -0.3)
    interferer = _source_signal(np.random.default_rng(_TRAP_SEED + 1), _TRAP_N_SAMPLES,
                                _TRAP_SAMPLE_RATE_HZ, 4, 0.2)
    return target, interferer


@pytest.mark.oracle
def test_a_linear_mix_collapses_two_components_to_the_noise_floor():
    """Trap 1, pinned. A linear estimate lies exactly in the span of the sources, so
    `target_distortion` and `artifacts` are ~1e-12 of the estimate and any comparison
    of them is noise against noise. This is CORRECT behaviour, which is why the
    strategy excludes the input rather than the tolerance absorbing it.

    MEASURED: both land at ~1.1e-12 of the estimate RMS, and peass-vs-reference
    relative L2 on them is 1.1e-2 and 2.4e-2 -- five decades past the `artifacts` bar,
    on a decomposition with nothing wrong with it.
    """
    target, interferer = _trap_sources()
    estimate = target + 0.5 * interferer  # exactly linear: the trap

    reference = _reference_components([target, interferer], estimate,
                                      _TRAP_SAMPLE_RATE_HZ)
    estimate_rms = _root_mean_square(estimate)
    ratios = {label: _root_mean_square(component) / estimate_rms
              for label, component in zip(_COMPONENTS, reference)}

    for label in ("target_distortion", "artifacts"):
        assert ratios[label] < _DEGENERACY_FLOOR, (
            f"{label} is {ratios[label]:.3e} of the estimate RMS on an exactly linear "
            f"mix, above the {_DEGENERACY_FLOOR:.0e} degeneracy floor. Either the "
            f"decomposition changed or `_source_signal` did. If a linear mix no longer "
            f"collapses these components, the exclusion in the module docstring is "
            f"stale and the strategy could be widened -- but check why first."
        )
    for label in ("true_target", "interference"):
        assert ratios[label] > 1e-2, (
            f"{label} should be unaffected by the linearity of the mix, but it is "
            f"{ratios[label]:.3e} of the estimate RMS"
        )


@pytest.mark.oracle
def test_two_fir_related_sources_make_the_projection_ill_posed():
    """Trap 2, pinned -- and note this is MONO.

    TODO.md and benchmarks/measure.py describe this trap as a second CHANNEL that is an
    FIR-realisable image of the first. It arises just as readily between two SOURCES,
    with one channel: `sources = [x, 0.5*x]` makes the per-frame Gram singular, and the
    two implementations then pick different minimum-norm solutions.

    MEASURED: `target_distortion` and `interference` disagree by 3.6e-1 and 3.9e-1
    relative L2 -- seven decades past their bars -- while `true_target` (1.7e-15) and
    `artifacts` (2.3e-13) are untouched, because those two do not depend on how the
    singular system's null space is resolved. The components are NOT small; their peaks
    agree to ~3%. They are simply different, and both are right.
    """
    target, _ = _trap_sources()
    duplicate = 0.5 * target  # exactly FIR-realisable from `target`: the trap
    mixture = target + duplicate
    estimate = np.tanh(3.0 * mixture) / 3.0

    reference = [np.asarray(component, dtype=np.float64) for component in
                 _reference_components([target, duplicate], estimate,
                                       _TRAP_SAMPLE_RATE_HZ)]
    produced = [_as_numpy(component) for component in _peass_components(
        [target, duplicate], estimate, _TRAP_SAMPLE_RATE_HZ, "numpy")]

    errors = {}
    for label, reference_component, produced_component in zip(
            _COMPONENTS, reference, produced):
        length = len(reference_component)
        errors[label] = _relative_l2(produced_component[:length, 0],
                                     reference_component[:length, 0])

    for label in ("target_distortion", "interference"):
        assert errors[label] > 1e-4, (
            f"{label} now agrees to {errors[label]:.3e} on a rank-deficient Gram. "
            f"That would be GOOD news -- it would mean the two implementations resolve "
            f"the singular system identically and the independent-noise-floor "
            f"exclusion in `_source_signal` could be relaxed -- but verify it before "
            f"relaxing anything, and update the module docstring."
        )
    for label in ("true_target", "artifacts"):
        assert errors[label] < _MAX_RELATIVE_L2_NUMPY[label], (
            f"{label} should be immune to the Gram's rank -- it does not depend on how "
            f"the null space is resolved -- but it disagrees by {errors[label]:.3e}"
        )


# ---------------------------------------------------------------------------
# Liveness proof for the eight bars
# ---------------------------------------------------------------------------
#
# THE PROBLEM THIS SOLVES. Every bar in this file clears its measured worst case by
# 48x-1018x. Those margins are what keep the suite from flaking, but they are also why
# nothing in a green run distinguishes "the backends agree" from "the comparison is not
# actually reachable". Until this test existed there was NO evidence that any of the
# eight bars could fail at all, and a tolerance that cannot fail is worse than no
# tolerance because it reads as coverage. The flat torch 1e-3 that Stage B replaced is
# the cautionary case: it survived every measurement for weeks while carrying seven to
# ten decades of slack.
#
# HOW IT WORKS, AND WHAT THE NUMBERS MEAN. Scaling a component by `1 + eps` moves its
# relative L2 against the reference to `sqrt(e0^2 + eps^2 + cross) ~= eps` whenever
# `eps >> e0`, which every bar here satisfies by at least 48x. So a bar B resolves a
# relative gain error of ~B on its own component, and perturbing by a stated fraction
# of B either side of 1.0 brackets the trip point without depending on the absolute
# scale of anything. THAT is the bar's resolving power, and it is the number worth
# reporting: `_MAX_RELATIVE_L2_TORCH["true_target"] = 1e-10` means this suite notices a
# 1e-10 relative error in torch's `true_target` and does not notice 5e-11.
#
# MEASURED 2026-08-17 by bisection on the fixed case below
# (`.scratch/hypothesis-2026-08-17/bisect_bar_liveness.py`): all eight bars trip at
# 1.0000x-1.0026x their own value -- the closed-form prediction above, confirmed. The
# worst baseline-to-bar ratio on this case is 1.7e-2 (numpy `true_target`), so the
# 0.5x / 2.0x bracket asserted here is not marginal: it is a factor of two either side
# of a threshold that is sharp to a fraction of a percent.
#
# WHAT THIS TEST DOES NOT PROVE, and it matters. Because the perturbation is scaled BY
# each bar, this test is invariant to the bar's absolute size: it would have passed
# just as happily against the vacuous flat torch 1e-3 it now guards. It proves the
# comparison is REACHABLE and that the four components are INDEPENDENT. It says nothing
# about whether a bar is tight enough to be useful -- that is what the measured tables
# above are for, and the two checks are complements, not substitutes.
#
# It is NOT fuzzed: one fixed case, one oracle call, both backends. Fuzzing it would
# multiply the oracle bill by `max_examples` to re-prove the same arithmetic.
_LIVENESS_CASE = {
    "sampling_frequency_hz": 8000,
    "n_samples": 4000,
    "seed": 20260817,
    "partial_count": 4,
    "target_tilt": -0.3,
    "interferer_tilt": 0.2,
    "leakage_db": -6.0,
    "drive": 3.0,
    "artifact_db": -35.0,
}

# The bracket, as a multiple of each bar. 0.5x must pass, 2.0x must fail.
_LIVENESS_PASSING_FRACTION = 0.5
_LIVENESS_FAILING_FRACTION = 2.0


@pytest.mark.oracle
def test_every_tolerance_bar_is_live():
    """Mutation-prove all eight bars: each one catches an error just over its own size
    and tolerates one just under.

    Perturbs ONE component at a time and asserts that exactly that component's bar
    reports, which also proves the bars are independent -- a component's error does not
    leak into its neighbours' comparisons.

    This drives `_compare_against_reference`, the same function the property uses, so
    it is the shipped comparison being proved and not a copy of it.
    """
    sources, estimate, _mixture = _build_case(_LIVENESS_CASE)
    sampling_frequency_hz = _LIVENESS_CASE["sampling_frequency_hz"]
    reference = [np.asarray(component, dtype=np.float64)
                 for component in _reference_components(
                     sources, estimate, sampling_frequency_hz)]

    backends = [("numpy", _MAX_RELATIVE_L2_NUMPY)]
    if _HAS_TORCH:
        backends.append(("torch", _MAX_RELATIVE_L2_TORCH))

    for backend, bars in backends:
        produced = [_as_numpy(component) for component in _peass_components(
            sources, estimate, sampling_frequency_hz, backend)]

        # Control. If the unperturbed case does not pass, everything below is
        # measuring the wrong thing.
        unperturbed = _compare_against_reference(
            backend, bars, reference, produced, _LIVENESS_CASE)
        assert not unperturbed, (
            f"the unperturbed liveness case must PASS before any mutation of it means "
            f"anything, but it does not:\n" + "\n".join(unperturbed)
        )

        for index, label in enumerate(_COMPONENTS):
            bar = bars[label]

            just_inside = list(produced)
            just_inside[index] = produced[index] * (
                1.0 + _LIVENESS_PASSING_FRACTION * bar)
            survived = _compare_against_reference(
                backend, bars, reference, just_inside, _LIVENESS_CASE)
            assert not survived, (
                f"BAR TOO TIGHT: {backend}/{label} rejected a relative gain error of "
                f"{_LIVENESS_PASSING_FRACTION} x its own {bar:.0e} bar, which should "
                f"sit comfortably inside it:\n" + "\n".join(survived) + "\n"
                f"Either the bar was tightened without re-measuring, or the backend's "
                f"baseline error on this fixed case has grown until it is a "
                f"significant fraction of the bar -- check the baseline first, because "
                f"if it has grown the fuzzed property is about to start flaking."
            )

            just_outside = list(produced)
            just_outside[index] = produced[index] * (
                1.0 + _LIVENESS_FAILING_FRACTION * bar)
            caught = _compare_against_reference(
                backend, bars, reference, just_outside, _LIVENESS_CASE)
            assert len(caught) == 1 and f"{backend}/{label}" in caught[0], (
                f"BAR IS DEAD: {backend}/{label} was given a relative gain error of "
                f"{_LIVENESS_FAILING_FRACTION} x its own {bar:.0e} bar and the "
                f"comparison reported {caught!r} instead of exactly that one "
                f"component.\n"
                f"If the list is EMPTY the comparison for this component is not "
                f"reachable at all -- the bar is not being applied, and the fuzzed "
                f"property above is passing vacuously on it no matter what the "
                f"backend does. Check `_compare_against_reference` first; a bar that "
                f"is merely too GENEROUS cannot show up here, because the perturbation "
                f"is scaled by the bar itself. Re-measure sizes with "
                f"`.scratch/hypothesis-2026-08-17/prove_oracle_bars.py`.\n"
                f"If the list has MORE than one entry the bars are no longer "
                f"independent: perturbing one component moved another's comparison, "
                f"which means `_compare_against_reference` or `_peass_components` "
                f"started sharing state between components."
            )
