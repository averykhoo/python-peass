# todo

Open work only. Settled items — landed fixes, closed investigations, declined options —
live in `ARCHIVE.md`.

> ## ⚠ PROCESS STATE — the branch is PUSHED; everything from 2026-08-18 is UNCOMMITTED
>
> **This replaces the "COMMITTED, NOT PUSHED" banner, which is stale in both halves: the
> branch *is* pushed, and the tree is no longer fully committed.**
>
> **What is on the branch, and it is pushed.** `perf/2026-08-12-decomposition`, base
> `e960c5e`, **six commits, all of them pushed** — verified 2026-08-18,
> `git ls-remote origin perf/2026-08-12-decomposition` returns `9f78864`:
>
> | commit | contents |
> | --- | --- |
> | `f847817` | the `1e-15` solver ridge + the analysis-modulation fold |
> | `6e0162d` | the float64 gammatone frequency grid |
> | `72992d9` | the three `utils.py` items (Kaiser dedup, padless polyphase, fused split/pad) |
> | `12f32ea` | the structural half-spectrum negative tests |
> | `cd80f01` | the four re-derived parity bars |
> | `9f78864` | docs — the 2026-08-17 verification pass |
>
> The perf items are separated from the correctness fixes, so the near-exact `72992d9` can be
> reverted on its own.
>
> **What is not.** **Everything from 2026-08-18 is in the working tree only, and nothing has
> been committed or pushed this session.** Three independent pieces of work:
>
> - **P5, both halves** — `peass/backend_torch/{utils,decomposition,gammatone}.py`,
>   `tests/unit/backend_torch/{test_torch_utils,test_torch_decomposition,test_torch_gammatone}.py`
> - **the `resample_filter_half_length_factor` guard** — `peass/config.py`,
>   `tests/unit/test_config.py`
> - **the `hypothesis` oracle suite** — `tests/regression/test_reference_oracle_property.py`
>   (**untracked**), `pyproject.toml`, `requirements.txt`
>
> Plus `TODO.md`, `ARCHIVE.md` and `README.md`. **Do not `git add -A`** if you intend to
> commit these separately — they are three unrelated changes that happen to share a tree.
>
> **Green**: `655 passed, 24 skipped, 4 deselected`; `pytest -m oracle -q` →
> `4 passed, 679 deselected in 37.48s`.
>
> **All three pieces are gate-verified and settled**; their write-ups are in `ARCHIVE.md`
> (2026-08-18). P5 passed ground rule 2 (numpy exactly zero movement, all 8 torch
> correlations improved) and ground rule 3 (**1.1673x** mono / **1.1571x** stereo, six A/Bs,
> all AGREE first time), and removes 117.9 MB of resident cache.
>
> The frozen baseline at `benchmarks/results/` **still must not be retaken**. It has now been
> used twice; keep it, so the next change measures against the same fixed point. The `p5` tag
> now sits beside `baseline` there as the post-P5 capture.

**Ground rules for performance work**, since they have decided every call so far and are
not obvious from the code:

1. **The library's own code spawns no threads, no subprocesses, no multiprocessing.**
   Speed comes from efficiency — SIMD/AVX, better memory traffic, fewer allocations,
   less redundant work, better algorithms. Numba `prange` was measured at 1.8x and
   declined on exactly this basis (`ARCHIVE.md`). BLAS threading inherited from the
   user's NumPy is a separate matter and is fine.
2. **Near-exact changes can land. Measure the deviation against the MATLAB reference
   audio, not against the last build.** Reassociation and solver substitutions that
   compute the *same quantity* are landable; a genuine *approximation* is not.
   Bit-identical is still preferred, and the README table still marks which changes
   are.

   The bar is `tests/regression/test_matlab_regression.py`, which decomposes
   `tests/resources/matlab_reference/` and asserts per-component correlation against
   MATLAB PEASS v2.0.1's gold WAVs plus the locked resampler gain offset. Diffing a
   change against the previous release is a fine *development* signal, but it is not
   the bar: each release only ever gets compared to the one before it, so drift
   ratchets in a step at a time and every step looks clean. A fixed reference cannot
   drift with you.

   This rule previously required bit-identical outright, and cited the torch
   decomposition stack as measured, prototyped and deliberately *unlanded* at
   2.10-2.45x. That stack landed on 2026-08-10; see `ARCHIVE.md`, "Decomposition:
   torch ~2.2x, numpy ~1.47x", for what it cost in accuracy and how that was verified.
   The 2026-08-12 pass is the first done the way this rule actually asks — every change
   compared to a capture frozen *before* the whole series, not to the commit before it.
   See `ARCHIVE.md`, "The 2026-08-12 pass, measured cumulatively against the fixed
   reference". Freeze the capture before you start; you cannot reconstruct it after.
3. **Measure, don't estimate** — and be careful what you claim from one machine. Two
   entries in `ARCHIVE.md` exist because a platform-specific observation was written
   down as a universal invariant.

   **Wall-clock before/after does not resolve a change worth less than ~10%.** Learned
   the hard way on 2026-08-12: this machine drifts 6-8% between runs, and on three
   separate occasions an *untouched* backend appeared to move more than the touched one
   — a numpy-only change read 1.19x by before/after and 1.08x when measured properly,
   and one run showed 41.9% spread within six repeats of a single configuration. Use
   `benchmarks/ab.py`: it interleaves the two candidates along the Thue-Morse sequence,
   repeats the whole run with them swapped, and refuses to give a number when the two
   phases disagree. Note also that `min` was the wrong statistic here — the outlier
   structure is one-sided, so medians with bootstrap intervals beat min-of-N.

   **Sharpened 2026-08-18: absolute medians are not comparable ACROSS processes either.**
   The identical `new` code read 995 ms in one P5 A/B and 882 ms in another minutes later
   on the same idle machine — **11.4% apart**. Only *within-run* ratios are quotable; never
   cross-quote the millisecond columns of two `ab.py` runs against each other.

   `benchmarks/` also holds `measure.py` and `compare.py`, which is how you take the
   frozen capture rule 2 asks for. See `benchmarks/README.md`.

**Picking this up? Start here.**

1. **This series' "before" snapshot already exists — do not retake it.**
   `benchmarks/results/baseline.json` + `baseline_wav/` were captured at `e960c5e`
   *before any edit* in the current working tree. Recapturing now would capture the
   changed code and silently destroy the comparison. The ground-rule-2 gate against it has
   now been run **twice** and passed both times (2026-08-17, and again for P5 on
   2026-08-18); keep the capture anyway, so the next change in this tree can be measured
   against the same fixed point. `p5` is the post-P5 capture and sits beside it, so a new
   change can be read against either.

   For a *new* series, take your own capture at the commit you start from, and note
   that `benchmarks/results/` is gitignored so a previous session's capture is not
   yours. **A capture is not the same thing as `reference/`, and one does not replace
   the other.** `reference/` is frozen *code*: an oracle that answers "is this output
   **correct**?". A capture is frozen *data*: the actual waveforms and timings at one
   commit, answering "what did my change **move**?". You need the capture because the
   reference cannot resolve small changes — two different implementations differ from
   each other by far more than a reassociation does, so a 1e-9 shift is invisible
   against it. A capture compares your code to *itself before the edit*, where the only
   difference is your edit, which is how the 2026-08-12 pass measured a 1.18e-9 move on
   `artifacts` and then attributed it. The capture also records timings; the reference
   says nothing about speed.
2. **Read `ARCHIVE.md`'s rejected lists before picking an item.** They are much longer than
   this file's open list, and they include ideas that look obviously right — a numpy
   polyphase GEMM mirroring the torch win measured **3.3x slower**, routing torch's resampler
   at the numpy Numba kernels measured **0.461x**, and P5's own widened real `dgemm` variant
   measured **0.717x**.
3. **Quote no timing number that did not come from `benchmarks/ab.py`.** Wall-clock
   before/after cannot resolve anything under ~10% on this machine; see rule 3. **And do
   not size a change from an isolated kernel harness either** — three measured cases
   overstate by 1.5x or more against the same change measured in situ (`ARCHIVE.md`,
   2026-08-15), and P5 is now a fourth, milder one: 1.351x/1.426x isolated against
   1.105x/1.05-1.07x in situ. Sizing from the wrong profile cuts both ways: the same entry
   records an item that was *under*-sized 2.2x because it counted kernel calls from a
   decomposition profile when the metric path makes six times as many.

   **And measure overlapping changes in BOTH directions** (learned 2026-08-17). Removing one
   item at a time from the current tree reported all three `utils.py` perf items as null,
   point estimates multiplying to ~0.999, against 1.0394x measured for the three together —
   two of them attack the same copy, so each is sufficient and neither is necessary.

   **The rule was re-run on 2026-08-18 and came out the OPPOSITE way, which is what makes it
   a rule rather than an anecdote.** P5's two halves measure 1.1047x / 1.0514x turned ON from
   an all-old baseline and 1.1062x / 1.0735x removed OFF from the finished tree — the two
   directions AGREE, the absolute savings sum to 160-172 ms against 161 ms measured together,
   and the ON-direction product reproduces the headline to 0.5%. So the halves are genuinely
   independent. **You cannot tell the overlapping case from the independent case without
   measuring both directions**, and which one you are in decides whether the per-item numbers
   mean anything.
4. **Know what the last passes took.** 2026-08-15 worked the *auditory/metric* path
   (`ARCHIVE.md`): metric 1.75x, whole pipeline 1.24x, from four changes. The cheap
   structural wins there are spent. Compose those ratios and the metric path went from ~45%
   of torch `predict` to ~32%, so the **decomposition is the larger share** — and it has now
   been worked over by the 2026-08-09/10/12 passes, the 2026-08-15 correctness/utils pass
   (**1.0394x** mono / 1.0334x stereo in situ) and P5 (**1.1673x** mono / **1.1571x** stereo,
   2026-08-18). That is why what is left below is harder than what just landed. 2026-08-17
   added no optimisation — it ran the gates, re-verified the bit-identity claims and fixed
   two test defects.
5. **THE HEADLINE IS NOW THE LEAST-SQUARES ASSEMBLY, OPTION 2(c) — and it is the last sized
   perf candidate in the file.** P5 was the headline and is done (`ARCHIVE.md`, 2026-08-18);
   the P-numbered list is now entirely archived. What is left on the torch side is
   `perform_time_varying_least_squares_projection_torch`, 21% of mono at 0.247 s cumulative
   of which only 0.015 s is `cholesky_ex` — so it is almost entirely data movement and the
   Gram/RHS GEMMs. See the item under "perf ideas not yet taken".

   **Be honest about what is known: no ratio exists for it at all.** P5 at least had an
   isolated-harness figure to discount; 2(c) has never been prototyped, so there is not even
   an overstated number. It is near-exact (~7e-16), so it needs the full ground-rule-2
   treatment and its own commit. **If it does not pay, this file has no perf headline left**
   and item 6 becomes the headline outright.
6. **The torch gammatone's wrap guard is the dominant accuracy term left, and P5 has settled
   that by elimination.** 5.4e-6 relative against a 4x-padded oracle, driven by a 0.2 s guard
   against a designed `pad_len` of 4800. This file used to call that "very likely *the*
   dominant term", which was an inference; it can now be stated. The float64 frequency grid
   removed one competitor (2026-08-15) and P5 removed another — the unreduced modulation
   phase, worth ~4e-11 — and **everything else now measured on the torch decomposition path
   sits at 1e-12 to 1e-15**, five to nine orders below the wrap guard. Nothing is within
   reach of it. See "correctness, not perf".
7. **The cheapest decisive next step is Numba neutrality, and it is three passes overdue.**
   It is not a code change: one scoring run with `numba` importable and one without. The
   "3 of 8 fields" set under "correctness, not perf" and README's table have been stale since
   2026-08-15, were flagged stale again on 2026-08-17, and P5 has now moved torch numerics a
   third time. **Neither gate measures this** — the frozen-capture gate compares the tree
   against itself before the edits, which says nothing about Numba-present versus
   Numba-absent. Do it before anything else; it is nearly free and two documents are wrong
   until it is done. Then **do not write a test asserting which fields move** — see the item
   for why.
8. **Two items became unblocked on 2026-08-15 and neither has been started**: generating
   reference files for inputs we have no gold for, and configurable reflection padding.
   Both were gated on root-causing the onset transient, which is done — it was a missing
   solver ridge, not a warm-up.
9. **One thing to watch on the next CI run**, not a work item:
   `_MAX_AUDITORY_REPRESENTATION_DEFICIT` was tightened to 1e-12 against a deviation that
   measures exactly zero here. See "one number to watch" at the bottom.

- train a different model on the peass data (which was removed in PR #3, commit `7ad923b3` on 2026-06-06 17:01)
  - get more data from https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS
- use peass decomposition as an ablation for haspi metrics
- note to self: add `-n auto` for pycharm to speed up tests,
  examples here: https://www.jetbrains.com/help/pycharm/performing-tests.html#run-tests-in-parallel
- **add reflection padding milliseconds as a configurable option** — helpful for short
  files (especially under 1 s), see
  `test_torch_decomposition.py::test_torch_decomposition_gain_invariance_with_padding`.
  **Newly unblocked 2026-08-15.** This item was gated on root-causing the backend
  divergence on short clips, on the grounds that tuning padding on top of an unexplained
  divergence would tune against a bug. That divergence is now root-caused and fixed — it
  was torch's missing `1e-15` solver ridge, not a padding or warm-up effect
  (`ARCHIVE.md`, 2026-08-15) — so the short-clip regime is a sane place to work again.

## deferred from the 2026-07 code review (not release-blocking)

- perf/backprop: the torch metrics (auditory adaptation loop) still dominate
  backprop — ~1.2x backward/forward, ~39s for 2s audio (2026-07-30, after the
  cascade-collapse work in `ARCHIVE.md`) — because it is BPTT through the sequential
  recurrence (the decomposition's solve backward is cheap by comparison — measured
  when that solve was a `pinv`; since 2026-08-10 it is a Cholesky, so cheaper still).
  The gradient path cannot use the fast forward paths since it needs the
  straight-through max, so it runs far slower than the no-grad path. A custom
  `autograd.Function` with a hand-derived analytic backward for the 5-stage cascade
  would make training-scale backprop practical; it's substantial and needs careful
  gradient validation. Note the 2026-08-08 Numba kernel does **not** help here — it is
  forward-only by design, and neither does the 2026-08-15 fusion that absorbed the
  haircell stage into it, for the same reason. **Re-checked 2026-08-15 and this item is
  now relatively more urgent, not less**: the no-grad path around it got ~1.75x faster on
  the metric path while the gradient path did not move at all, and the 2026-08-15
  `utils.py` work moved the no-grad path further ahead again while the gradient path
  still has not moved. **Note P5 does NOT widen the gap** (2026-08-18): its decimation fold
  is applied on the grad route as well as the no-grad one, deliberately, so both paths took
  the same win — the alternative would have given training and inference different numerics.
  Re-measure the ratio before quoting it.
- perf/deprecation: `torch.jit.script` is deprecated in favour of
  `torch.compile`/`torch.export`. Only one call site remains (the adaptation-loop
  fallback in `backend_torch/auditory_model.py`), it is worth ~2x where it is still
  used, and its warning is filtered at that one site. Neither replacement is viable
  here — re-verified 2026-08-08: inductor's CPU backend still fails with
  `InductorError: Compiler: cl is not found` (no `cl`/`gcc`/`clang` on this box), and
  `torch.export` fully unrolls the loop at ~9 graph nodes per timestep — ~437k nodes
  and ~30 min export for 2s of audio, re-exported per input length. Revisit only on a
  machine with a working inductor backend or a CUDA runtime. Note this is now much
  less urgent: the Numba kernel already delivers the fusion this item was chasing,
  on the CPU path, at ~1e-14 (exactly zero on the reference platform — see
  `ARCHIVE.md` for why that is not a portable guarantee). **Less urgent again as of
  2026-08-15**: the kernel now also absorbs the relu, haircell, clamp and dB affine, so
  the scripted fallback covers proportionally less work than it did — and it is reached
  by fewer callers than the gate reads, since `float32` callers do *not* fall back
  (`GammatoneAnalyzerTorch` promotes unconditionally; see README). It is still the one
  remaining `jit.script` site and still worth ~2x where it runs. Unchanged 2026-08-18.
- the `_EXPECTED_SCORES` characterization values in `test_matlab_regression.py`
  are Python-reference numbers (the decomposition now matches MATLAB to ~0.9999,
  but we still don't have MATLAB's published OPS/TPS/IPS/APS to assert against);
  replace with MATLAB's actual scores for the example clips if/when available.
  There is now a second route to this that does not need MATLAB: transcribe the score
  path into `reference/` — see the ground-truth reference section below.
  Re-checked 2026-08-18: still open and unchanged, and the tolerances absorbed P5 without
  edits. Note these values are weakly **Numba-dependent** — some scores differ by ~1e-9 or
  less between Numba-present and Numba-absent runs (see "correctness, not perf") — so
  whichever numbers eventually replace them must be asserted with a tolerance, never `==`.

## ground-truth reference — decomposition done, score path open

`reference/` transcribes MATLAB PEASS v2.0.1's **decomposition path**; landed 2026-08-14,
written up in `ARCHIVE.md`. It is also now a *live* oracle: the `hypothesis` property suite
fuzzes both backends against it (applied 2026-08-18, `ARCHIVE.md`), which closes the gap the
old tests structurally could not — every invariant we had constrains the *form* of the answer
and is satisfied by a wrong one. Remaining work, in rough priority order:

- **Generate reference files for inputs we have no gold for.** This was the original
  motivation. **Newly unblocked 2026-08-15**: it was explicitly gated on root-causing the
  onset transient first, so that generated ground truth would not be generated on top of
  a backend divergence — and that root cause is now found and fixed (the missing `1e-15`
  solver ridge, `ARCHIVE.md`). The reference produces ground truth for short clips,
  unusual rates and stereo, none of which the single MATLAB gold clip covers. The
  reflection-padding item above is the obvious first customer. Caution worth keeping:
  the reference is validated on *one* clip, so a transcription bug invisible there would
  silently poison everything generated. Prefer generating for inputs close to the
  validated regime first, and sanity-check invariants (algebraic reconstruction, gain
  invariance) on whatever comes out. Note the property suite now covers a fuzzed *box*
  around that clip rather than the clip alone, which narrows but does not close this risk.

  Three things to design around if you extend the strategies, all learned the hard way and
  all pinned by characterization tests today (details in `ARCHIVE.md`, 2026-08-18):
  the reference is deliberately slow, so constrain hard and cap `max_examples`; a *linear*
  mix collapses `target_distortion` **and** `artifacts` to ~1.2e-12 of the estimate RMS; and
  FIR-related sources make the per-frame Gram rank-deficient — **in mono, between two
  sources**, so "constrain to mono" does not buy you out of it.

- **Transcribe the score path** — `audioQualityFeatures.m`, `pemo_internal.m`,
  `PEASS_ObjectiveMeasure.m`, `map2SubjScale.m`, `myMapping.m`, `pemo_metric.m`,
  `ISR_SIR_SAR_fromNewDecomposition.m`. That would let us finally replace the
  `_EXPECTED_SCORES` characterization values above with numbers derived from MATLAB
  rather than from our own output. Note `haircell`/`adapt` are MEX with pure-MATLAB
  fallbacks, and the adaptation loop is a sequential recurrence, so a naive transcription
  will be *very* slow — which is acceptable for a reference. Still untouched.
- **Use it as a refactor gate for the queued perf work.** The least-squares assembly
  item below has no independent oracle today; `reference/LSDecompose_tv.py` is one now,
  and it is what caught the missing solver ridge in the first place.

Deliberately NOT done: a reference for the resampler. There is no route to an independent
one short of real MATLAB — see the Octave investigation in `ARCHIVE.md` — so all four
`resample` call sites are declared deviations rather than transcriptions.

## perf ideas not yet taken

From the decomposition-focused profiles of 2026-08-09, 2026-08-10 and 2026-08-12, plus
the A/Bs run on 2026-08-15. Measured on `tests/resources/database/exp01_*`, and subject to
rule 1 — efficiency and SIMD, no fanning out.

**One item is left here, and it is the only one in this file that was never prototyped.**
The whole P-numbered list is now settled: P1-P4 are archived — P4 landed, P3 was rejected
(the batching experiment the polyphase GEMM obsoleted) — and **P5 landed on 2026-08-18**
(1.1673x mono / 1.1571x stereo, −117.9 MB, `ARCHIVE.md`).

**Check `ARCHIVE.md` before reviving anything.** What landed and what was measured and
rejected both live there, and the rejected list is far longer than this one — notably FIR
symmetry folding, Levinson, the batching experiment the polyphase GEMM obsoleted, a numpy
polyphase GEMM that measured 3.3x *slower*, the whole Numba-resampler-for-torch idea at
0.461x, and P5's own widened real `dgemm` at 0.717x.

**Re-measure before trusting any share-of-runtime number below.** These were sized against a
decomposition in which resampling was 60% of runtime. cProfile on a warm torch mono
decomposition put `fast_resample_poly_torch` at 198 calls and 37% cumulative (2026-08-12) —
and **that block has since given up ~3-4% to the 2026-08-15 `utils.py` items and another
~14% of the whole decomposition to P5** (mono phase-1 median 1161 ms → 1000 ms). Treat 37% as
a stale upper bound; the resampler is materially thinner than any figure below assumes, and
the least-squares stage is correspondingly a larger share than it was.

### torch

- **The least-squares assembly, option 2(c) — fold the window into the Toeplitz in place**
  (`decomposition.py:293-301`). `perform_time_varying_least_squares_projection_torch`
  is 21% of mono at 0.247 s cumulative, of which only 0.015 s is `cholesky_ex` — so it is
  almost entirely data movement and the Gram/RHS GEMMs. The `window_sq * toeplitz_batched`
  temporary at `:300` and the duplicate synthesis-window pass could both be folded into
  the Toeplitz as it is built.

  **Measure first, and it wants its own commit.** It is near-exact (~7e-16 relative), not
  bit-identical, so it needs the full ground-rule-2 treatment: frozen capture, before/after
  against the MATLAB gold WAVs, and its own accuracy paragraph. **And note it has never been
  prototyped** — unlike every other item that ever sat here, there is no harness figure to
  discount, so the first honest job is sizing it at all.

  **Its stated dependency may have evaporated — re-check before sizing it.** It was
  recorded as needing the `.reshape` at `:295` to become a view, and that was to have been
  delivered by replacing `.flip(-1)` with `index_select`. That replacement was attempted on
  2026-08-15 and **refuted** — it is not bit-identical, see `ARCHIVE.md` — so it is not
  coming. Work out whether 2(c) still needs a view at all, or whether it only ever needed
  one because of how the flip item was going to be written.

  Options 2(a) (flip the solved weights instead of the Toeplitz block) and 2(b) (batched
  HERK for the Hermitian Gram) were both settled on 2026-08-15 and are in `ARCHIVE.md`.
  Do not re-derive them.

### correctness, not perf

- **The torch gammatone's accuracy floor is its wrap guard, not its FFT length.**
  Against a 4x-padded oracle the current transform is 5.4e-6 relative, driven by a
  0.2 s guard (`backend_torch/gammatone.py:292` and `:347`) against a designed `pad_len`
  of 4800. Raising `pad_len` matters far more than any FFT sizing change (see the dropped
  sizing item in `ARCHIVE.md`).

  **Re-read 2026-08-18: this is now the dominant torch accuracy term by elimination, not by
  inference.** The float32 frequency grid, which used to sit near it in magnitude, was fixed
  on 2026-08-15; P5 has now removed the other candidate, the unreduced modulation phase at
  ~4e-11. Everything else measured on that path — the folds themselves, the range reduction,
  the padless polyphase — is at 1e-12 to 1e-15. There is nothing else within five orders of
  this. Worth re-sizing the guard deliberately rather than leaving it at a round 0.2 s.

- **Micro-item, considered and not taken (2026-08-18): reduce `turns` into `(-0.5, 0.5]`
  before the exponential.** `_reduced_phase_exp` in `backend_torch/utils.py` currently
  reduces into `[0, 1)`. Reducing into `(-0.5, 0.5]` halves the polar argument and measures
  **5.16e-16 against 9.14e-16** on a 50-digit `mpmath` reference — a real **1.8x**, and
  orthogonal to the `limit_denominator` defect P5 fixed. Not taken because both forms are
  already on the float64 floor and ~4 orders below anything the pipeline can see. Filed with
  its number so it is not re-derived; take it only if something downstream ever needs that
  decade.

- **Numba is no longer numerically neutral on the torch path.** As of 2026-08-15, **3 of
  the 8 reported scores differed** between Numba-present and Numba-absent runs; before the
  auditory-path pass it was 0 of 8. Roundoff on a 0-100 scale, but it is a difference where
  there was none, so installing or removing the optional `numba` extra now changes the last
  digits of a score.

  **That 3-of-8 field set is stale and probably wrong now, and it is staler than it was.**
  Torch numerics have moved three times since it was measured: the 2026-08-15 pass (solver
  ridge, float64 frequency grid, ~1 ULP padless polyphase), and now P5's range reduction and
  folds — after which `overall_perceptual_score` is **identical on both backends to twelve
  digits**. README's table is stale for the same reason and needs re-measuring, not editing
  from memory. **This is the cheapest open job in the file; see "start here" item 7.**

  Recorded here for one reason: **do not write a test that asserts a specific set of
  moved fields.** Which fields move is itself unstable under unrelated ULP-level changes
  upstream — one pass alone took it from 0 of 8 to 2 of 8 to 3 of 8, with the field set
  changing, and `target_perceptual_score` landed on the *identical* value under two
  different changes and under both together (see `ARCHIVE.md`). The durable invariant is
  that some scores differ and that the differences are ~1e-9 or smaller; a field-set
  assertion would be fragile and would fail on the next unrelated ULP change.

- **Undocumented behaviour change: complex input to the torch auditory path now raises.**
  `GammatoneAnalyzerTorch.process_real` raises `ValueError` for a complex input, where the
  auditory path previously routed through `process` and computed `.real` of a
  complex-filtered *complex* signal. Investigated 2026-08-15: the unreachability was
  verified empirically (the resampler in front of it raises first), and the numpy backend
  was checked for the same divergence. Failing loud is the better behaviour, so nothing was
  changed — but the item is still open, because the decision was never made. Decide
  deliberately whether it wants a test pinning the unreachability, or a note in the
  docstring, or nothing.

  There is now a precedent to decide it against, and it points the same way: the
  `resample_filter_half_length_factor` guard (`ARCHIVE.md`, 2026-08-18) took exactly this
  shape of question — silently wrong output versus a loud error through a public surface —
  and was resolved as **error and halt**, with acceptance tests pinning the boundary. The
  difference is that this one is already loud; what it lacks is a decision and a test.

## one number to watch on the first CI run

Not a defect — a knowingly accepted risk from 2026-08-17, recorded so it is not discovered as
a mystery CI failure. (The three defects that used to sit here are all closed: two on
2026-08-17 and the `hypothesis` oracle suite on 2026-08-18; see `ARCHIVE.md` for their
lessons.)

- **`_MAX_AUDITORY_REPRESENTATION_DEFICIT` is 1e-12 against a deviation that measures exactly
  zero here.** The bar was tightened 1e-5 → **1e-12** on a re-derivation where the measured
  deviation is **0.0 exactly** on this machine, i.e. resolution-limited. **Headroom that
  cannot be measured is not headroom**, and ground rule 3 exists because this repo has twice
  written a one-machine observation down as a universal invariant. It is the one number in
  that pass that is not margin-justified.

  Mitigations already in place: a new companion bar on the auditory max deviation at
  `1e-10 * peak(numpy)` with a real **257x** margin (measured 6.0163e-10), and 1e-12 still
  catches the float32-grid regression by **4258x**. The exposure is
  ubuntu-latest 3.10/3.12/3.14 and windows-latest 3.14. If it fails on another platform,
  loosen *this* bar to a measured margin — do not touch the `1e-10 * peak` companion, which is
  the one doing the real work.

## process state — read before touching anything

- **The frozen baseline exists and must not be retaken.**
  `benchmarks/results/baseline.json` + `baseline_wav/` were captured at `e960c5e` before
  any edit — 24 float64 component waveforms and all 8 scores. `git stash` + recapture does
  not reproduce it. It has now been used twice (2026-08-17, 2026-08-18); keep it, so the next
  change in this tree measures against the same fixed point. **`p5` sits beside it** as the
  post-P5 capture, taken through `measure.py p5` on the current working tree.
- **The ground-rule-2 gate has been run twice and passed twice.** 2026-08-17 for the five
  committed changes, and 2026-08-18 for P5 — the latter against a capture verified genuine
  first (`git_dirty = false`). Both times: numpy exactly zero movement, all 8 torch
  correlations improved, worst correlation drop `+0.000e+00`. The one growth `compare.py`
  flags both times (+6.814e-07 on torch/`target_distortion` gain error) is **convergence onto
  numpy**, not regression — numpy's own value is the one torch moved to. Full tables in
  `ARCHIVE.md`.
- **The ground-rule-3 A/Bs have been run for both perf passes.** The three `utils.py` items:
  1.0394x mono [1.0310, 1.0467] / 1.0334x stereo, AGREE. P5: **1.1673x** mono
  [1.1631, 1.1751] / **1.1571x** stereo [1.1486, 1.1643], six A/Bs, all AGREE on the first
  attempt. Read the attribution method notes in `ARCHIVE.md` before quoting any per-item
  figure from either — one pass overlapped and one did not, and only measuring both
  directions told them apart.
- **Numba-neutrality has still not been re-measured.** Neither gate touches it, so the
  3-of-8 field set under "correctness, not perf" and README's table are both still stale —
  and staler than they were, since P5 moved torch numerics again.
- **Nothing from 2026-08-18 is committed or pushed.** The branch itself
  (`perf/2026-08-12-decomposition`, six commits above `e960c5e`) **is** pushed — see the
  banner for the mapping and the verification. Today's three pieces of work are in the
  working tree only, the suite is green at **`655 passed, 24 skipped, 4 deselected`**, and
  they are three unrelated changes sharing a tree: commit by path, not with `git add -A`.
  Some archive entries were written while their work was still uncommitted and carry dated
  notes where that matters.
