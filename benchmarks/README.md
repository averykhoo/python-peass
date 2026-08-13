# benchmarks

Developer tooling for performance and accuracy work. Not shipped (`pyproject.toml`
excludes `benchmarks/*` from the sdist), not collected by pytest (`norecursedirs`),
not imported by the library.

| script | what it does |
| --- | --- |
| `measure.py` | captures one **tag**: accuracy vs the MATLAB gold WAVs, the 8 locked scores/ratios, every decomposed component waveform as float64 `.npy`, and coarse decomposition timings for {numpy,torch} x {mono,stereo}. |
| `compare.py` | diffs two tags: timing, correlation and gain-error deltas vs MATLAB, score deltas, and the worst waveform deviation relative to each component's own peak. |
| `ab.py` | pairwise A/B timing of two callables, Thue-Morse interleaved, run twice with the candidates swapped. This is the only timing number worth quoting. |

## Running

```bash
python benchmarks/measure.py baseline           # freeze a reference, ~75-85 s
python benchmarks/measure.py current            # after your change
python benchmarks/compare.py baseline current   # judge the change
python benchmarks/ab.py --samples 96            # demo A/B on this repo's resampler
```

Results land in `benchmarks/results/` (gitignored). Override with `--results-dir DIR`
or `$PEASS_BENCH_RESULTS`; `measure.py` and `compare.py` resolve it identically.
A tag costs about **25 MB** (24 float64 component waveforms) and `measure.py` takes
about **75-85 s**. Keep the frozen baseline, delete the rest when the series is done.

`measure.py` imports the gain-offset constant, the gain tolerance and the locked
score/ratio dicts from `tests/regression/test_matlab_regression.py` rather than
copying them, so the harness always reproduces the bar the test actually asserts.

## The workflow (this is the part that matters)

Both halves come from `TODO.md`'s ground rules; `ARCHIVE.md` has the incidents that
produced them.

**Freeze the baseline before you start the series, and compare everything to that
frozen tag — never to the previous commit.** (Ground rule 2.) Each release only ever
being compared to the one before it is exactly how drift ratchets in: every step looks
clean and the total does not. The bar is deviation from the *MATLAB reference audio*,
which is what `measure.py`'s section (A) and `compare.py`'s section 2 record. You
cannot reconstruct the pre-series capture afterwards, so take it first:

```bash
git stash          # or check out the pre-series commit
python benchmarks/measure.py baseline
```

**Use `ab.py` for any timing claim.** (Ground rule 3.) Wall-clock before/after cannot
resolve a change worth less than ~10% on this machine: it drifts 6-8% between runs, and
on three separate occasions an *untouched* backend appeared to move more than the
touched one — a numpy-only change read 1.19x by before/after and 1.08x when measured
properly. `compare.py`'s section 1 is kept as a sanity check only; treat it as such.
`ab.py` interleaves the two candidates along the Thue-Morse sequence so that background
load and thermal drift are split near-equally between them, then repeats the whole run
with the candidates swapped and checks that the two phases agree. If they do not, it
reports **UNRELIABLE** — which means the machine was too busy or the effect is
order-dependent, and there is no number to quote. Note also that `min` is the wrong
statistic here (the outlier structure is one-sided); `ab.py` reports medians with
bootstrap intervals, and its module docstring explains the rest of the method and its
limits.

Point it at your own pair of callables:

```python
from benchmarks.ab import compare_ab

result = compare_ab("old", lambda: old_path(x), "new", lambda: new_path(x), samples_per_phase=96)
if result.agree:
    print(result.pooled_ratio, result.pooled_ratio_ci)
```

The A/B is a timing tool only. Accuracy is still judged by `compare.py` against the
frozen tag, and the landing gate is still the full suite from the repo root.
