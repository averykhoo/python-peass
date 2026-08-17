# todo

Open work only. Settled items — landed fixes, closed investigations, declined options —
live in `ARCHIVE.md`.

> ## ⚠ COMMITTED, NOT PUSHED — verified and measured
>
> A 2026-08-15 session landed **six changes across four files**: the missing `1e-15` solver
> ridge, the analysis-modulation fold into the band gather, the float64 gammatone frequency
> grid, and three `utils.py` items (Kaiser dedup, padless polyphase, fused split/merge). A
> 2026-08-17 pass then verified them and fixed two test defects they had created. **Green
> (`606 passed, 24 skipped`).**
>
> All of it is now **committed on `perf/2026-08-12-decomposition` above base `e960c5e`, in
> five commits, and nothing is pushed**: `f847817` (ridge + modulation fold), `6e0162d`
> (float64 grid), `72992d9` (the three `utils.py` items), `12f32ea` (structural half-spectrum
> negative tests), `cd80f01` (the four re-derived parity bars). The perf items are separated
> from the correctness fixes, so the near-exact `72992d9` can be reverted on its own.
>
> **Both gates the tree was missing have now been run** (2026-08-17, `ARCHIVE.md`, "The
> 2026-08-17 verification pass"):
>
> - **Ground rule 2 — passed.** numpy exactly zero movement; **all 8 torch correlations
>   improved**; worst correlation drop `+0.000e+00`; every deficit and gain error 30x-230x
>   inside its bar. Largest cumulative torch score move **3.643e-04** against a 1.0 bar,
>   which *corrects* the ≤2.2e-7 figure the 2026-08-15 archive entry implied.
> - **Ground rule 3 — the tree has a quotable in-situ number.** The three `utils.py` perf
>   items measure **1.0394x** mono [1.0310, 1.0467] / **1.0334x** stereo [1.0198, 1.0446]
>   together, AGREE. Do not attribute that per item from a leave-one-out A/B; see the method
>   note in the archive entry.
>
> The frozen baseline at `benchmarks/results/` **cannot be retaken** once the tree is dirty —
> it has now been used, but keep it, so the next change measures against the same fixed point.
> See "process state" at the bottom of this file.

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

   `benchmarks/` also holds `measure.py` and `compare.py`, which is how you take the
   frozen capture rule 2 asks for. See `benchmarks/README.md`.

**Picking this up for performance work? Start here.**

1. **This series' "before" snapshot already exists — do not retake it.**
   `benchmarks/results/baseline.json` + `baseline_wav/` were captured at `e960c5e`
   *before any edit* in the current working tree. Recapturing now would capture the
   changed code and silently destroy the comparison. The ground-rule-2 gate against it
   **has been run** (2026-08-17, passed — see the banner above); keep the capture anyway, so
   the next change in this tree can be measured against the same fixed point.

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
2. **Read `ARCHIVE.md`'s rejected lists before picking an item.** They are longer than
   this file's open list, and they include ideas that look obviously right — a numpy
   polyphase GEMM mirroring the torch win measured **3.3x slower**, and as of 2026-08-15
   routing torch's resampler at the numpy Numba kernels measured **0.461x**.
3. **Quote no timing number that did not come from `benchmarks/ab.py`.** Wall-clock
   before/after cannot resolve anything under ~10% on this machine; see rule 3. **And do
   not size a change from an isolated kernel harness either** — three measured cases now
   overstate by 1.5x or more against the same change measured in situ (`ARCHIVE.md`,
   2026-08-15). Sizing from the wrong profile cuts both ways: the same entry records an
   item that was *under*-sized 2.2x because it counted kernel calls from a decomposition
   profile when the metric path makes six times as many.

   **And measure overlapping changes in BOTH directions** (learned 2026-08-17,
   `ARCHIVE.md`). Removing one item at a time from the current tree reported all three
   `utils.py` perf items as null, point estimates multiplying to ~0.999, against 1.0394x
   measured for the three together. Two of them attack the same copy, so each is sufficient
   and neither is necessary. Leave-one-out measures *marginal* value against an
   already-optimised baseline; turn each item on from the un-optimised baseline as well, and
   report both.
4. **Know what the last passes took.** 2026-08-15 worked the *auditory/metric* path —
   `ARCHIVE.md`, "The 2026-08-15 auditory-path pass": metric 1.75x, whole pipeline 1.24x,
   from four changes. The cheap structural wins there (one fused kernel instead of six
   full-array passes; a real-output transform where the imaginary half was discarded) are
   spent. Compose those two ratios and the metric path went from ~45% of torch `predict`
   to ~32%, so the **decomposition is now decisively the larger share** — and it has
   already been worked over by the 2026-08-09/10/12 passes and again by the 2026-08-15
   correctness/utils pass, which is why what is left below is harder than what
   just landed. That last pass is now measured: **1.0394x** mono / **1.0334x** stereo on a
   full torch decomposition (2026-08-17), so the decomposition has given up another ~3-4% and
   the resampler block is that much thinner than the profiles below assume. 2026-08-17 itself
   added no optimisation — it ran the two gates, re-verified the bit-identity claims and fixed
   two test defects; see `ARCHIVE.md`, "The 2026-08-17 verification pass".
5. **The headline recommendation is now P5**, folding the modulation into the polyphase
   filters — the largest known remaining win. It is measured, it wins, and it is
   unimplemented: 1.351x analysis / 1.426x synthesis on full replacement. Effort is high and
   it has two hard implementation constraints; see the item under "perf ideas not yet taken"
   before starting.

   It became the headline on 2026-08-17 by elimination — the cheap test-and-measurement
   jobs that used to sit in front of it (the blinded gammatone test, the four fragile
   tolerances) are **done**, and the two gates the tree was missing are **run**. Two
   caveats for planning: three attempts on 2026-08-17 died on transient API 529 errors
   without writing a checkpoint, so P5 is untouched and starts from scratch; and the
   resampler path it overlaps has now been measured 1.0394x faster in situ, so ground rule
   3's "kernel harnesses overstate" warning applies to those 1.351x/1.426x figures.
6. **Cheapest remaining test-and-measurement job: the `hypothesis` oracle property suite**,
   under "outstanding defects". The design and a ~700-line patch exist, the dependency
   decision is made, and the opt-in-versus-CI question was decided on 2026-08-17 (opt-in
   only). What is left is installing `hypothesis`, fixing the five reviewed defects and
   applying it — a "finish it", not a "start it".
7. **Two items became unblocked on 2026-08-15** and neither has been started: generating
   reference files for inputs we have no gold for, and configurable reflection padding.
   Both were gated on root-causing the onset transient, which is now done — it was a
   missing solver ridge, not a warm-up.
8. **One thing to watch on the next CI run**, not a work item:
   `_MAX_AUDITORY_REPRESENTATION_DEFICIT` was tightened to 1e-12 against a deviation that
   measures exactly zero here. See the last item under "outstanding defects".

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
  still has not moved. The gap between training and scoring is wider than the ~1.2x
  backward/forward figure above implies. Re-measure the ratio before quoting it.
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
  remaining `jit.script` site and still worth ~2x where it runs. Unchanged 2026-08-15.
- the `_EXPECTED_SCORES` characterization values in `test_matlab_regression.py`
  are Python-reference numbers (the decomposition now matches MATLAB to ~0.9999,
  but we still don't have MATLAB's published OPS/TPS/IPS/APS to assert against);
  replace with MATLAB's actual scores for the example clips if/when available.
  There is now a second route to this that does not need MATLAB: transcribe the score
  path into `reference/` — see the ground-truth reference section below.
  Re-checked 2026-08-15: still open and unchanged, and the tolerances absorbed that
  pass without edits. But note these values are now weakly **Numba-dependent** — some
  scores differ by ~1e-9 or less between Numba-present and Numba-absent runs (see
  "correctness, not perf"), so whichever numbers eventually replace them must be
  asserted with a tolerance, never with `==`.

## ground-truth reference — decomposition done, score path open

`reference/` transcribes MATLAB PEASS v2.0.1's **decomposition path**; landed 2026-08-14,
written up in `ARCHIVE.md`. Remaining work, in rough priority order:

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
  invariance) on whatever comes out.
- **Property-based testing against the reference, with `hypothesis`** — the stronger
  version of the item above: instead of freezing a fixed set of generated golden files,
  use `reference/` as a *live oracle* and fuzz the input space, asserting the fast
  backends agree with it. This closes the gap the existing tests structurally cannot:
  every invariant we have — algebraic reconstruction, gain invariance, gammatone round
  trip — constrains the *form* of the answer and is satisfied by a **wrong** one.
  `true_target + target_distortion + interference + artifacts == estimate` holds for any
  partition of the estimate. Only a reference constrains the *content*, and today it is
  only checked on one clip.

  **Two things changed on 2026-08-15 and this entry is now a "finish it", not a
  "start it".** First, **`hypothesis` is approved as a dev/test extra** (user decision) —
  but it is **not installed** and no dependency file lists it yet. Second, the suite has
  been **designed and written as an unapplied patch**; see the outstanding-defects
  section below for the list of things a reviewer found in it *before* it was applied,
  all of which must be fixed as part of applying it.

  Three things to design around, all learned the hard way and none obvious:

  - **The reference is slow** (deliberately). Naive fuzzing is far too slow for the
    default gate. Constrain the strategies hard — short signals, low sample rates, two
    sources, mono — and cap `max_examples` with `deadline=None`. Note the shipped design's
    own cost claim is ~10x too high: a reference call measures ~1.1-1.5 s, not the 13-15 s
    the patch and this entry previously assumed. That changes the budget arithmetic, so
    re-decide opt-in versus default on the real number.
  - **Hypothesis will find the degenerate cases immediately, and they are traps rather
    than bugs.** A *linear* mix lies exactly in the span of the sources, so `artifacts`
    comes out at ~1e-14 and any comparison of it becomes noise-against-noise. And a
    second channel that is an exactly FIR-realizable image of the first (say a pure
    delay, against a 640-tap filter) makes the per-frame Gram rank-deficient, so
    `target_distortion` and `interference` become minimum-norm garbage — peaks ~1.7, and
    the two backends disagreeing by 9e-2. Both were hit while building
    `benchmarks/measure.py`; see the FROZEN CONVENTIONS block there for the nonlinear
    estimate and decorrelated stereo that avoid them. Either exclude these from the
    strategy or assert something weaker on them, but do it deliberately.
  - **Comparison tolerance must be per component**, for the same reason
    `test_matlab_regression.py` now is: `artifacts` amplifies ~1e6 through the
    ill-conditioned least-squares stage, so a single bar is meaningless across the four.

  Worth noting `reference/` runs without the MATLAB sources — only
  `verify_transcription.py` needs them — so this works on a fresh clone and in CI.

- **Transcribe the score path** — `audioQualityFeatures.m`, `pemo_internal.m`,
  `PEASS_ObjectiveMeasure.m`, `map2SubjScale.m`, `myMapping.m`, `pemo_metric.m`,
  `ISR_SIR_SAR_fromNewDecomposition.m`. That would let us finally replace the
  `_EXPECTED_SCORES` characterization values above with numbers derived from MATLAB
  rather than from our own output. Note `haircell`/`adapt` are MEX with pure-MATLAB
  fallbacks, and the adaptation loop is a sequential recurrence, so a naive transcription
  will be *very* slow — which is acceptable for a reference. Deliberately out of scope
  for the 2026-08-15 session; untouched.
- **Use it as a refactor gate for the queued perf work.** The least-squares assembly
  item below has no independent oracle today; `reference/LSDecompose_tv.py` is one now,
  and it is what caught the missing solver ridge in the first place.

Deliberately NOT done: a reference for the resampler. There is no route to an independent
one short of real MATLAB — see the Octave investigation in `ARCHIVE.md` — so all four
`resample` call sites are declared deviations rather than transcriptions.

## perf ideas not yet taken

From the decomposition-focused profiles of 2026-08-09, 2026-08-10 and 2026-08-12, plus
the A/Bs run on 2026-08-15. All prototyped and measured on
`tests/resources/database/exp01_*` unless marked unprototyped, and all subject to rule 1
— efficiency and SIMD, no fanning out.

**Check `ARCHIVE.md` before reviving anything here.** What landed and what was measured
and rejected both live there, and the rejected list is far longer than this one — notably
FIR symmetry folding, Levinson, the batching experiment the polyphase GEMM obsoleted, a
numpy polyphase GEMM that measured 3.3x *slower*, and as of 2026-08-15 the whole
Numba-resampler-for-torch idea at 0.461x. P5 below is the last survivor of the original
P-numbered list; P1-P4 are archived.

**Re-measure before trusting any share-of-runtime number below.** These were sized
against a decomposition in which resampling was 60% of runtime. It is not 60% any more,
but it is still the largest single block: cProfile on a warm torch mono decomposition puts
`fast_resample_poly_torch` at 198 calls and 37% cumulative (2026-08-12). Treat 37% as an
upper bound — cProfile inflates call-heavy Python frames and this is 198 calls — but do
not assume resampling has stopped mattering.

**Updated 2026-08-17: the 2026-08-15 `utils.py` work now HAS an in-situ number**, where
this preamble previously said it had none at all. The three items together (Kaiser dedup,
padless polyphase interior, fused split/merge) measure **1.0394x** mono
`ab.py` [1.0310, 1.0467] and **1.0334x** stereo [1.0198, 1.0446] on a full torch
decomposition, both AGREE. So ~3-4% of the resampler block is already taken, and the two
items that delivered it (split/pad 1.0327x, padless 1.0230x from an all-old baseline)
removed full-signal copies — meaning what P5 stands to win from the same block is smaller
than the 37% share above and the isolated 1.351x/1.426x harness figures together suggest.
The Kaiser dedup is a **speed null in both A/B directions** and stands on its memory result
only.

### torch

- **P5, measured and unimplemented — fold the modulation into the polyphase filters**
  (`decomposition.py:484` on the analysis side and the synthesis alignment matrix).
  Complex-exponential modulation distributes through convolution exactly, so the
  full-length modulation multiplies fold into the 21-tap filters and the cached modulation
  matrices (61 MB + 62 MB) can be dropped. **This is now the largest known remaining
  win.**

  **The caveat that this file previously called "probably fatal" is not fatal.** It was
  A/B'd on 2026-08-15 on an idle machine, all phases AGREE under swapping:

  | A/B | ratio |
  | --- | --- |
  | analysis, GEMM only (the caveat mechanism) | 1.221x |
  | **analysis, full replacement** | **1.351x** |
  | synthesis, GEMM only | 1.452x |
  | **synthesis, full replacement** | **1.426x** |

  The doubled complex-GEMM flops cost *less* than the 61/62 MB streaming passes they
  remove. Harness: `.scratch/handoff-2026-08-15/harnesses/ab_p5_complex_gemm.py`. These
  are isolated-harness numbers, so ground rule 3 applies — expect less in situ.

  **Two 2026-08-17 updates.** First, "expect less in situ" is now quantified from the
  neighbouring evidence rather than asserted: the same resampler path was measured 1.0394x
  faster in situ by the `utils.py` A/B, and the two items that delivered that were removing
  full-signal copies — some of the streaming traffic P5's 61/62 MB argument rests on has
  already been taken. Second, **three implementation attempts were made and none started**:
  all three died on transient **API 529 Overloaded** errors, a server-side capacity issue
  with nothing to do with the work. The tree was verified byte-for-byte untouched after each
  (`peass/` at 86/29/330 changed lines, no untracked files) and no checkpoint was written, so
  this item is exactly where 2026-08-15 left it. Both hard constraints below are intact and
  unrevised.

  **Two hard implementation constraints, both measured, both non-negotiable:**

  1. **Implement it as a true complex `zgemm`, not as a widened real `dgemm`.** The
     variant that keeps the folded kernel on the real split path measured **0.717x** —
     i.e. it loses. The real-FIR split (`_split_real_imag`, `utils.py`) is what makes the
     current polyphase GEMM cheap, and a modulated kernel is complex, so the choice is
     between a genuine complex GEMM and four real ones. The complex GEMM wins; four real
     GEMMs lose.
  2. **Range-reduce the phase mod one turn before the exponential.** Naive folding
     measures **3.7e-11** (analysis) / **4.4e-11** (synthesis), because `2*pi*fc*n/fs`
     reaches ~2.2e8 rad at `T = 120000` and `torch.exp` loses ~8 digits of phase there.
     With exact `Fraction` range reduction the same fold measures **~7e-16**. See
     `.scratch/handoff-2026-08-15/harnesses/verify_exp.py`.

  The other original caveat is confirmed as described: the modulation does not vanish, it
  shrinks. `m[n]·(h*x)[n] = Σ (h[k]m[k])·(m[n-k]x[n-k])`, so folding into the taps leaves
  a residual modulation on the *decimated*-rate side — `m[mD]` on the analysis output, the
  mirror of it on the synthesis input. That is still 32 multiplies of length `T/D_b`;
  cheap, but the saving is smaller than the dropped multiplies suggest. Note also that the
  synthesis half of this item overlaps the fused alignment matrix P4 introduced.
  Effort: **high** — the index algebra needs careful per-band validation.

- **torch: the least-squares assembly, option 2(c) — fold the window into the Toeplitz in
  place** (`decomposition.py:293-301`). `perform_time_varying_least_squares_projection_torch`
  is 21% of mono at 0.247 s cumulative, of which only 0.015 s is `cholesky_ex` — so it is
  almost entirely data movement and the Gram/RHS GEMMs. The `window_sq * toeplitz_batched`
  temporary at `:300` and the duplicate synthesis-window pass could both be folded into
  the Toeplitz as it is built.

  **Measure first, and it wants its own commit.** It is near-exact (~7e-16 relative), not
  bit-identical, so it needs the full ground-rule-2 treatment: frozen capture, before/after
  against the MATLAB gold WAVs, and its own accuracy paragraph.

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

- **`resample_filter_half_length_factor = 0` is reachable and silently produces wrong
  output — decide on the guard.** Found 2026-08-17 while correcting a docstring; measured,
  not speculative. Full write-up in `ARCHIVE.md`, "The 2026-08-17 verification pass", §5.

  `DecompositionConfiguration.__post_init__` (`peass/config.py`) validates **only**
  `segmentation_factor`, so `resample_filter_half_length_factor=0` is accepted through the
  public API. At `hf = 0` the fast `_polyphase_decimate` does **not** raise: it returns finite
  numbers that disagree with its own `_polyphase_decimate_padded` reference by **O(1)** —
  deviations **0.39 to 2.13** across 33 of the swept `(down, in_len, rows, dtype)`
  combinations, against 2.22e-16 for `hf >= 1`. The gradient path routes to the padded
  reference, so the **no-grad and grad paths disagree** there. Root cause: the fast path's
  grid algebra needs `right_pad >= (hf-1)*down + 1 > 0`, i.e. `hf >= 1`.

  **The docstring claimed `hf = 0` was "not reachable" and that was false on both counts** —
  reachable through the public API, and unguarded rather than guarded. The comment now states
  the measured truth; the guard does not exist yet.

  **Recommended fix: one check in `config.py`'s `__post_init__` raising on `< 1`, mirroring
  the `segmentation_factor` check. Deliberately NOT applied, pending a decision**, because it
  is a public behaviour change — a loud error where callers currently get silently wrong
  output. Nothing in the library passes 0 (default 10; the config comment contemplates
  lowering only to ~3), so it has never been hit. Reusable point worth keeping either way:
  "nothing in the library passes that" is a statement about the library, not about the API,
  and a `__post_init__` that validates one field of several reads as though it validates all
  of them.

- **The torch gammatone's accuracy floor is its wrap guard, not its FFT length.**
  Against a 4x-padded oracle the current transform is 5.4e-6 relative, driven by a
  0.2 s guard (`backend_torch/gammatone.py:292` and `:347`) against a designed `pad_len`
  of 4800. Raising `pad_len` matters far more than any FFT sizing change (see the dropped
  sizing item in `ARCHIVE.md`).

  **Re-prioritised 2026-08-15: this is now very likely *the* dominant torch gammatone
  accuracy term.** The float32 frequency grid, which used to sit near it in magnitude, is
  fixed in the working tree, so nothing else of comparable size is left in front of it.
  Worth re-sizing the guard deliberately rather than leaving it at a round 0.2 s.

- **Numba is no longer numerically neutral on the torch path.** As of 2026-08-15, **3 of
  the 8 reported scores differed** between Numba-present and Numba-absent runs; before the
  auditory-path pass it was 0 of 8. Roundoff on a 0-100 scale, but it is a difference where
  there was none, so installing or removing the optional `numba` extra now changes the last
  digits of a score.

  **That 3-of-8 field set is stale and probably wrong now.** The 2026-08-15 work changed
  torch numerics again in three places (the solver ridge, the float64 frequency grid, the
  ~1 ULP padless polyphase), so *which* fields move has very likely shifted. README's table
  is stale for the same reason and needs re-measuring, not editing from memory.

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

## outstanding defects — one left open, plus one number to watch

Not perf items and not speculative. Two of the three defects found by adversarial review on
2026-08-15 were **fixed on 2026-08-17** and have moved to `ARCHIVE.md`, "The 2026-08-17
verification pass" §3, which keeps their lessons: the blinded gammatone test (and the fact
that this file's "catches neither" conclusion was itself half wrong — trap B was always
caught by a structural assert next to the bar), and the four fragile tolerances (re-derived,
tightened, units fixed, and the `rtol` principle question settled as *never* `rtol`). Full
briefs and repro scripts are in `.scratch/handoff-2026-08-15/`.

- **The `hypothesis` oracle property suite is written and not applied.** The design and a
  ~700-line patch exist in `.scratch/handoff-2026-08-15/`; the dependency decision is made
  (dev/test extra, approved) but `hypothesis` is not installed and the patch is not
  applied. Review found five things wrong with it *before* applying, all of which must be
  fixed as part of applying it:

  - The shipped tolerance tables understate their own re-run — numpy `interference`
    measured 1.347e-10 against a table entry of 7.90e-11; torch `target_distortion`
    6.583e-6 against 4.52e-6.
  - **The suite ships dead.** `addopts` deselects the marker everywhere and there is no CI
    leg that reselects it, yet three separate places in the patch describe it as "a
    nightly". **Decided 2026-08-17: make it opt-in only and delete the three "nightly"
    claims** — do not wire it into CI. Nothing reselects the marker today, and a reference
    call is only ~1.1-1.5 s, so a local opt-in run is practical and a CI leg is not worth
    buying.
  - `_MAX_EXAMPLES` parses an environment variable at module scope, so a malformed value
    raises at *collection* time and takes the whole suite down, not just this file.
  - The cost claim is ~10x too high — a reference call is ~1.1-1.5 s, not 13-15 s.
  - The `_DEGENERACY_FLOOR` comment is contradicted by the suite's own output.

  Re-measure all of its tolerances against **ridge-fixed** torch, not against the numbers
  in the patch.

- **WATCH ON THE FIRST CI RUN: `_MAX_AUDITORY_REPRESENTATION_DEFICIT` is 1e-12 against a
  deviation that measures exactly zero here.** Not a defect — a knowingly accepted risk from
  2026-08-17, recorded so it is not discovered as a mystery CI failure.

  The bar was tightened 1e-5 → **1e-12** on a re-derivation where the measured deviation is
  **0.0 exactly** on this machine, i.e. resolution-limited. **Headroom that cannot be measured
  is not headroom**, and ground rule 3 exists because this repo has twice written a
  one-machine observation down as a universal invariant. It is the one number in that pass
  that is not margin-justified.

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
  not reproduce it. It has now been *used* (2026-08-17); keep it, so the next change in this
  tree measures against the same fixed point.
- **The ground-rule-2 gate HAS been run (2026-08-17) and it passed.** `measure.py current`
  (80.4 s) + `compare.py baseline current`, against a capture verified genuine first
  (`git_commit = e960c5e4fa…`, `git_dirty = False`). numpy exactly zero movement on every
  correlation, gain error and score; **all 8 torch correlations improved**, worst correlation
  drop `+0.000e+00`; deficits and gain errors 30x-230x inside their bars; largest torch score
  move 3.643e-04 against 1.0; worst waveform move 6.144e-05 of peak on `artifacts`, toward
  MATLAB. The one growth `compare.py` flags (+6.814e-07 on torch/`target_distortion` gain
  error) is **convergence onto numpy**, not regression. Full tables in `ARCHIVE.md`.
- **The in-situ A/B for the `utils.py` perf items HAS been run (2026-08-17).** The three
  together: **1.0394x** mono [1.0310, 1.0467] and **1.0334x** stereo [1.0198, 1.0446], both
  AGREE. From an all-old baseline: fused split/pad 1.0327x, padless interior 1.0230x, Kaiser
  dedup 1.0053x (null). **Removed one at a time from the current tree all three read null** —
  read the method note in `ARCHIVE.md` before quoting any per-item figure. P5's numbers above
  are still kernel-harness numbers, and ground rule 3 records three cases where such
  harnesses overstated by 1.5x or more in situ.
- **Numba-neutrality has still not been re-measured.** The gate compares the tree against the
  capture, not Numba-present against Numba-absent, so the 3-of-8 field set under "correctness,
  not perf" and README's table are both still stale.
- **Committed, not pushed.** Branch `perf/2026-08-12-decomposition`, five commits above base
  `e960c5e` (`f847817`, `6e0162d`, `72992d9`, `12f32ea`, `cd80f01` — see the banner for the
  mapping), suite green at **`606 passed, 24 skipped`** (603 before the 2026-08-17 gammatone
  fix added three tests). The docs in this repo were written while the work was still in the
  working tree, so anything still reading "uncommitted" is stale — the archive entries carry
  dated notes where that matters.
