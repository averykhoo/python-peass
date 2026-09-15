"""
Pairwise A/B timing of two callables, interleaved along the Thue-Morse sequence.

Usage:  python benchmarks/ab.py [--samples N] [--warmup K] ...      (runs a demo)
        from benchmarks.ab import compare_ab                        (as a library)

Why not "time the old build, time the new build"
------------------------------------------------
Wall-clock before/after cannot resolve a change worth less than ~10% on this
machine: it drifts 6-8% between runs, and on three separate occasions an
*untouched* backend appeared to move more than the touched one (TODO.md ground
rule 3, ARCHIVE.md "The 2026-08-12 pass"). Anything that changes slowly during a
session -- turbo/thermal state, another process waking up, the page cache
filling -- lands entirely on whichever candidate happened to run during it.

The method
----------
Run both candidates alternately in one process, assigning run `i` by the
Thue-Morse sequence `t(i) = popcount(i) & 1`:

    t = 0 1 1 0 1 0 0 1 1 0 0 1 0 1 1 0 ...
    phase 1:  t(i) == 0 -> A, else B
    phase 2:  the opposite mapping

Thue-Morse is the standard "fair sharing" order: every prefix of length 2^k
splits exactly evenly, and it is balanced not just in counts but against
low-order polynomial trends, so a slow ramp (thermal drift) or a burst
(background process) is split near-equally between the candidates rather than
landing on one. Plain ABABAB... is balanced in counts but locks each candidate to
one parity, which aliases against any period-2 effect (e.g. an allocator or
cache state that alternates); Thue-Morse has no such fixed parity.

Then the whole thing is run again with A and B swapped. Any residual
position-in-sequence effect -- "the run that goes first is slower", "whoever
follows B pays for B's cache footprint" -- changes sign between the phases, so
comparing the two phase estimates *is* the measurement's own error bar. If the
phases disagree beyond their bootstrap intervals, the result is reported
UNRELIABLE: either the machine was too busy or the effect is order-dependent, and
in both cases the number is not a property of the code.

Design choices
--------------
* Warm-up samples are **discarded, not down-weighted** (`warmup`, default 3 per
  candidate per phase). The first calls differ in kind, not just in noise: numba
  compiles, torch spins up its thread pool, the allocator grows its arenas, pages
  fault in. Down-weighting keeps a biased sample in the estimate at reduced
  weight; discarding removes it. The discarded count is reported so the choice is
  visible, and if a candidate's steady state genuinely needs more than `warmup`
  runs to appear, the phase estimates will disagree and say so.
* The central estimate is the **median**, not the min. `min` is the usual choice
  for benchmark noise that is purely additive, but ARCHIVE.md records that on
  this machine it was actively misleading -- "min is the wrong statistic under
  this machine's outlier structure" (2026-08-12): the outliers are one-sided but
  frequent enough that min-of-N samples the luckiest scheduling window rather
  than the typical cost, and it read 0.98x on a change the paired difference put
  at ~3.2 sigma positive. The median discards the same one-sided tail without
  betting everything on one sample.
* Uncertainty is a **percentile bootstrap** over the retained samples (resampling
  each candidate independently, `bootstrap` resamples, default 2000). It makes no
  normality assumption, which matters for a one-sided-tailed distribution.
* Reported ratio is `median_a / median_b`, i.e. "B is this many times faster than
  A", plus the percentage change of B relative to A (negative = B is faster).

Honest limitations
------------------
* It needs a lot of runs. Resolving a few percent wants tens of samples per
  candidate per phase; the defaults here (48 samples per phase) are a starting
  point, not a guarantee.
* On a fully idle machine it buys much less than on a busy one -- the noise it
  spreads fairly is largely not there, and you mostly pay the extra runtime. Its
  value is that it degrades gracefully when the machine *is* busy, and tells you
  when it could not cope.
* It controls for time-varying and position-in-sequence effects. It does NOT
  control for effects that are genuinely properties of running the two together:
  shared BLAS thread pools, shared caches, one candidate leaving the allocator in
  a state that helps or hurts the other. Both candidates are also measured under
  each other's memory pressure, which is not the same as either running alone.
* It measures wall time only. It says nothing about accuracy -- pair it with
  `measure.py` / `compare.py`, which is the actual landing bar (TODO.md rule 2).
* Both callables must be independently callable with no shared mutable state and
  no accumulating side effects (don't let one grow a cache the other reads), and
  should do enough work to clear the timer's resolution -- microsecond-scale
  callables are dominated by call overhead.
"""

import argparse
import dataclasses
import math
import os
import pathlib
import statistics
import sys
import time
from typing import Callable
from typing import Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


# --------------------------------------------------------------- the sequence --
def thue_morse(length: int) -> list[int]:
    """`t(i) = popcount(i) & 1` for i in [0, length)."""
    return [bin(i).count("1") & 1 for i in range(length)]


# ----------------------------------------------------------------- containers --
@dataclasses.dataclass
class CandidateStats:
    """One candidate's timings within one phase, in seconds."""
    label: str
    samples: list[float]  # every sample, in the order it was taken
    discarded: int  # leading warm-up samples not used below
    retained: list[float]
    median: float
    ci_low: float
    ci_high: float

    @property
    def spread_pct(self) -> float:
        if not self.retained:
            return float("nan")
        return (max(self.retained) - min(self.retained)) / min(self.retained) * 100.0


@dataclasses.dataclass
class PhaseResult:
    """One pass over the Thue-Morse sequence, with a fixed candidate mapping."""
    phase: int  # 1 = t(i)==0 -> A, 2 = swapped
    a: CandidateStats
    b: CandidateStats
    ratio: float  # median_a / median_b, ">1 means B is faster"
    ratio_ci: tuple[float, float]

    @property
    def pct_change(self) -> float:
        """Change of B relative to A, in percent. Negative = B is faster."""
        return (1.0 / self.ratio - 1.0) * 100.0


@dataclasses.dataclass
class ABResult:
    label_a: str
    label_b: str
    phase1: PhaseResult
    phase2: PhaseResult
    agree: bool
    verdict: str
    pooled_ratio: float
    pooled_ratio_ci: tuple[float, float]

    @property
    def pooled_pct_change(self) -> float:
        return (1.0 / self.pooled_ratio - 1.0) * 100.0


# ------------------------------------------------------------------ machinery --
def _bootstrap_ratio_ci(
        samples_a: Sequence[float],
        samples_b: Sequence[float],
        resamples: int,
        rng: np.random.Generator,
        confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for median(a)/median(b).

    The two candidates are resampled independently because the design does not
    pair individual runs -- Thue-Morse interleaves them, it does not couple run
    `i` of A to run `i` of B.
    """
    if len(samples_a) < 2 or len(samples_b) < 2:
        return (float("nan"), float("nan"))
    arr_a = np.asarray(samples_a, dtype=np.float64)
    arr_b = np.asarray(samples_b, dtype=np.float64)
    draws_a = rng.choice(arr_a, size=(resamples, arr_a.size), replace=True)
    draws_b = rng.choice(arr_b, size=(resamples, arr_b.size), replace=True)
    ratios = np.median(draws_a, axis=1) / np.median(draws_b, axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    return (float(np.percentile(ratios, tail)), float(np.percentile(ratios, 100.0 - tail)))


def _summarize(label: str, samples: list[float], warmup: int,
               rng: np.random.Generator, resamples: int) -> CandidateStats:
    """Drop the leading warm-up samples, then median + bootstrap CI on the rest."""
    discarded = min(warmup, max(len(samples) - 1, 0))
    retained = samples[discarded:]
    ci_low = ci_high = float("nan")
    if len(retained) >= 2:
        draws = rng.choice(np.asarray(retained, dtype=np.float64),
                           size=(resamples, len(retained)), replace=True)
        medians = np.median(draws, axis=1)
        ci_low = float(np.percentile(medians, 2.5))
        ci_high = float(np.percentile(medians, 97.5))
    return CandidateStats(
        label=label,
        samples=samples,
        discarded=discarded,
        retained=retained,
        median=float(statistics.median(retained)) if retained else float("nan"),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _run_phase(
        phase: int,
        label_a: str, fn_a: Callable[[], object],
        label_b: str, fn_b: Callable[[], object],
        samples: int,
        warmup: int,
        resamples: int,
        rng: np.random.Generator,
        verbose: bool,
) -> PhaseResult:
    swap = 0 if phase == 1 else 1
    order = [t ^ swap for t in thue_morse(samples)]
    functions = {0: fn_a, 1: fn_b}
    collected: dict[int, list[float]] = {0: [], 1: []}

    for side in order:
        fn = functions[side]
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        del result  # free before the next call so neither side pays the other's peak
        collected[side].append(elapsed)

    stats_a = _summarize(label_a, collected[0], warmup, rng, resamples)
    stats_b = _summarize(label_b, collected[1], warmup, rng, resamples)

    ratio = stats_a.median / stats_b.median if stats_b.median else float("nan")
    ratio_ci = _bootstrap_ratio_ci(stats_a.retained, stats_b.retained, resamples, rng)
    phase_result = PhaseResult(phase=phase, a=stats_a, b=stats_b, ratio=ratio, ratio_ci=ratio_ci)

    if verbose:
        mapping = f"t(i)==0 -> {label_a}" if phase == 1 else f"t(i)==0 -> {label_b}  (swapped)"
        print(f"\nphase {phase}  [{mapping}]")
        for stats in (stats_a, stats_b):
            print(f"  {stats.label:24s} n={len(stats.retained):3d} (+{stats.discarded} warm-up)"
                  f"  median={stats.median * 1e3:9.3f} ms"
                  f"  95% CI [{stats.ci_low * 1e3:.3f}, {stats.ci_high * 1e3:.3f}]"
                  f"  spread={stats.spread_pct:6.1f}%")
        print(f"  ratio {label_a}/{label_b} = {ratio:.4f}x"
              f"  95% CI [{ratio_ci[0]:.4f}, {ratio_ci[1]:.4f}]"
              f"   ({phase_result.pct_change:+.2f}% for {label_b})")
    return phase_result


def _intervals_overlap(one: tuple[float, float], two: tuple[float, float]) -> bool:
    if any(math.isnan(v) for v in (*one, *two)):
        return False
    return one[1] >= two[0] and two[1] >= one[0]


def compare_ab(
        label_a: str,
        fn_a: Callable[[], object],
        label_b: str,
        fn_b: Callable[[], object],
        *,
        samples_per_phase: int = 48,
        warmup: int = 3,
        bootstrap: int = 2000,
        seed: int = 20260813,
        verbose: bool = True,
) -> ABResult:
    """Time `fn_a` against `fn_b`, Thue-Morse interleaved, in two swapped phases.

    Both callables take no arguments and their return value is discarded; close
    over whatever inputs they need. `samples_per_phase` is the total number of
    calls per phase, so each candidate gets about half of it, of which the first
    `warmup` are dropped. Returns an :class:`ABResult`; when `verbose`, also
    prints the per-phase table and the verdict.

    The verdict is the point of the method: two phases whose bootstrap intervals
    do not overlap mean the measurement did not reproduce under a swapped order,
    and the ratio must not be quoted.
    """
    if samples_per_phase < 4 * (warmup + 1):
        raise ValueError(
            f"samples_per_phase={samples_per_phase} is too small for warmup={warmup}: "
            f"each candidate gets ~{samples_per_phase // 2} samples and drops {warmup}"
        )
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"A/B: {label_a}  vs  {label_b}")
        print(f"     {samples_per_phase} calls per phase, 2 phases, "
              f"{warmup} warm-up samples dropped per candidate per phase, "
              f"{bootstrap} bootstrap resamples")

    phase1 = _run_phase(1, label_a, fn_a, label_b, fn_b,
                        samples_per_phase, warmup, bootstrap, rng, verbose)
    phase2 = _run_phase(2, label_a, fn_a, label_b, fn_b,
                        samples_per_phase, warmup, bootstrap, rng, verbose)

    agree = _intervals_overlap(phase1.ratio_ci, phase2.ratio_ci)

    pooled_a = phase1.a.retained + phase2.a.retained
    pooled_b = phase1.b.retained + phase2.b.retained
    pooled_ratio = float(statistics.median(pooled_a) / statistics.median(pooled_b))
    pooled_ci = _bootstrap_ratio_ci(pooled_a, pooled_b, bootstrap, rng)

    both_null = (phase1.ratio_ci[0] <= 1.0 <= phase1.ratio_ci[1]
                 and phase2.ratio_ci[0] <= 1.0 <= phase2.ratio_ci[1])
    if not agree:
        verdict = "UNRELIABLE"
    elif both_null:
        verdict = "AGREE (no measurable difference)"
    else:
        verdict = "AGREE"

    result = ABResult(
        label_a=label_a, label_b=label_b, phase1=phase1, phase2=phase2,
        agree=agree, verdict=verdict,
        pooled_ratio=pooled_ratio, pooled_ratio_ci=pooled_ci,
    )

    if verbose:
        print("\n" + "=" * 78)
        print(f"phase 1 ratio {phase1.ratio:.4f}x  CI [{phase1.ratio_ci[0]:.4f}, {phase1.ratio_ci[1]:.4f}]")
        print(f"phase 2 ratio {phase2.ratio:.4f}x  CI [{phase2.ratio_ci[0]:.4f}, {phase2.ratio_ci[1]:.4f}]")
        if not agree:
            print("!!" + " " + "*" * 72)
            print("!! UNRELIABLE: the two phases DISAGREE -- their intervals do not overlap.")
            print("!! Swapping which candidate sits on which Thue-Morse parity changed the")
            print("!! answer, so this is not a property of the code. Either the machine was")
            print("!! too busy, or the effect is order-dependent (shared cache/allocator/BLAS")
            print("!! state). Do NOT quote a speedup from this run: close background work,")
            print("!! raise --samples, and re-run.")
            print("!!" + " " + "*" * 72)
        else:
            print(f"VERDICT: {verdict} -- phases reproduce under swapping.")
            print(f"  {label_b} vs {label_a}: {pooled_ratio:.4f}x"
                  f"  95% CI [{pooled_ci[0]:.4f}, {pooled_ci[1]:.4f}]"
                  f"   ({result.pooled_pct_change:+.2f}% time for {label_b})")
            if both_null:
                print("  (both phase intervals contain 1.0: no difference resolved at this "
                      "sample count -- that is a null result, not a confirmation of equality)")
        print("=" * 78)
    return result


# ----------------------------------------------------------------------- demo --
def _demo(samples_per_phase: int, warmup: int, bootstrap: int, seed: int) -> ABResult:
    """A/B two real, cheap code paths in this repo.

    `fast_resample_poly` has two interchangeable implementations behind one
    entry point -- the numba polyphase kernel and the SciPy `upfirdn` fallback
    (`USE_NUMBA_RESAMPLER`) -- computing the same quantity from the same filter.
    That makes it an honest A/B: same inputs, same output to ~1e-15, a few
    milliseconds a call, and no dependence on anything outside this repo.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from peass.backend_numpy import gammatone

    if not gammatone._HAS_NUMBA:
        print("numba is not installed: both candidates would take the SciPy path, so this")
        print("demo would A/B a function against itself. That is still a valid smoke test")
        print("of the harness (expect ~1.00x and a null verdict).")

    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((8, 16_000))  # 8 channels x 1 s at 16 kHz, float64
    up, down = 2, 1

    def call(use_numba: bool):
        # The toggle is a module-level flag read inside the call; setting it here
        # costs a single attribute store, ~1e-7 s against a ~1e-2 s call.
        gammatone.USE_NUMBA_RESAMPLER = use_numba
        return gammatone.fast_resample_poly(signal, up, down, axis=-1)

    reference = call(False)
    numba_out = call(True)
    deviation = float(np.max(np.abs(numba_out - reference)))
    peak = float(np.max(np.abs(reference)))
    print(f"demo: fast_resample_poly({signal.shape}, up={up}, down={down})")
    print(f"      the two paths agree to {deviation:.3e} absolute "
          f"({deviation / peak:.3e} relative to peak) -- same quantity, different code\n")
    del reference, numba_out

    try:
        return compare_ab(
            "scipy_upfirdn", lambda: call(False),
            "numba_polyphase", lambda: call(True),
            samples_per_phase=samples_per_phase,
            warmup=warmup,
            bootstrap=bootstrap,
            seed=seed,
        )
    finally:
        gammatone.USE_NUMBA_RESAMPLER = True  # restore the library default


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thue-Morse interleaved pairwise A/B timing. Run bare for a demo "
                    "on this repo's resampler; import compare_ab to point it at your own "
                    "pair of callables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=48,
                        help="calls per phase (each candidate gets about half)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="leading samples discarded per candidate per phase")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="bootstrap resamples for the confidence intervals")
    parser.add_argument("--seed", type=int, default=20260813,
                        help="seed for the bootstrap and the demo signal")
    args = parser.parse_args()
    outcome = _demo(args.samples, args.warmup, args.bootstrap, args.seed)
    raise SystemExit(0 if outcome.agree else 1)
