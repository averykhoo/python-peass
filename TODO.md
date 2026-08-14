# todo

Open work only. Settled items — landed fixes, closed investigations, declined options —
live in `ARCHIVE.md`.

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

- train a different model on the peass data (which was removed in PR #3, commit `7ad923b3` on 2026-06-06 17:01)
  - get more data from https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS
- use peass decomposition as an ablation for haspi metrics
- note to self: add `-n auto` for pycharm to speed up tests,
  examples here: https://www.jetbrains.com/help/pycharm/performing-tests.html#run-tests-in-parallel
- add reflection padding milliseconds as a configurable option, it's helpful for short files (especially under 1s), 
  see `test_torch_decomposition.py:test_torch_decomposition_gain_invariance_with_padding` for details

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
  forward-only by design.
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
  `ARCHIVE.md` for why that is not a portable guarantee).
- the `_EXPECTED_SCORES` characterization values in `test_matlab_regression.py`
  are Python-reference numbers (the decomposition now matches MATLAB to ~0.9999,
  but we still don't have MATLAB's published OPS/TPS/IPS/APS to assert against);
  replace with MATLAB's actual scores for the example clips if/when available.
  There is now a second route to this that does not need MATLAB: transcribe the score
  path into `reference/` — see the ground-truth reference section below.

## ground-truth reference — decomposition done, score path open

`reference/` transcribes MATLAB PEASS v2.0.1's **decomposition path**; landed 2026-08-14,
written up in `ARCHIVE.md`. Remaining work, in rough priority order:

- **Generate reference files for inputs we have no gold for.** This was the original
  motivation and it is now unblocked: the reference produces ground truth for short
  clips, unusual rates and stereo, none of which the single MATLAB gold clip covers. The
  reflection-padding item above is the obvious first customer. Caution worth keeping:
  the reference is validated on *one* clip, so a transcription bug invisible there would
  silently poison everything generated. Prefer generating for inputs close to the
  validated regime first, and sanity-check invariants (algebraic reconstruction, gain
  invariance) on whatever comes out.
- **Property-based testing against the reference, with `hypothesis`.** The stronger
  version of the item above: instead of freezing a fixed set of generated golden files,
  use `reference/` as a *live oracle* and fuzz the input space, asserting the fast
  backends agree with it. This closes the gap the existing tests structurally cannot:
  every invariant we have — algebraic reconstruction, gain invariance, gammatone round
  trip — constrains the *form* of the answer and is satisfied by a **wrong** one.
  `true_target + target_distortion + interference + artifacts == estimate` holds for any
  partition of the estimate. Only a reference constrains the *content*, and today it is
  only checked on one clip.

  **`hypothesis` is not currently a dependency** — this needs a decision first, presumably
  as a dev/test extra rather than a runtime one.

  Three things to design around, all learned the hard way and none obvious:

  - **The reference is slow** (~13 s per decomposition, deliberately). Naive fuzzing is
    far too slow for CI. Constrain the strategies hard — short signals, low sample rates,
    two sources, mono — and cap `max_examples` with `deadline=None`. This probably wants
    to be a nightly or opt-in marker rather than part of the default gate.
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
  will be *very* slow — which is acceptable for a reference.
- **Use it as a refactor gate for the queued perf work.** The least-squares assembly
  items below have no independent oracle today; `reference/LSDecompose_tv.py` is one now.

Deliberately NOT done: a reference for the resampler. There is no route to an independent
one short of real MATLAB — see the Octave investigation in `ARCHIVE.md` — so all four
`resample` call sites are declared deviations rather than transcriptions.

## perf ideas not yet taken

From the decomposition-focused profiles of 2026-08-09, 2026-08-10 and 2026-08-12. All
prototyped and measured on `tests/resources/database/exp01_*` unless marked unprototyped,
and all subject to rule 1 — efficiency and SIMD, no fanning out.

**Check `ARCHIVE.md` before reviving anything here.** What landed and what was measured
and rejected both live there, and the rejected list is longer than this one — notably FIR
symmetry folding, Levinson, the batching experiment the polyphase GEMM obsoleted, and a
numpy polyphase GEMM that measured 3.3x *slower*. P5 below is the last survivor of the
original P-numbered list; P1-P4 are archived.

**Re-measure before trusting the number below.** It was sized against a decomposition
in which resampling was 60% of runtime. It is not 60% any more, but it is still the
largest single block: cProfile on a warm torch mono decomposition puts
`fast_resample_poly_torch` at 198 calls and 37% cumulative (2026-08-12). Treat 37% as an
upper bound — cProfile inflates call-heavy Python frames and this is 198 calls — but do
not assume resampling has stopped mattering.

### torch

- **P5, unprototyped, was sized at 109 ms — fold the modulation into the polyphase
  filters** (`decomposition.py:407` and the synthesis alignment matrix). Complex-
  exponential modulation distributes through convolution exactly, so the full-length
  modulation multiplies fold into the 21-tap filters and the cached modulation matrices
  (61 MB + 62 MB) can be dropped. Exact in real arithmetic, expected ~1 ULP.
  Effort: high — the index algebra needs careful per-band validation.

  **Two caveats found on 2026-08-12 that the original sizing missed, and the second is
  probably fatal.** First, the modulation does not vanish, it shrinks:
  `m[n]·(h*x)[n] = Σ (h[k]m[k])·(m[n-k]x[n-k])`, so folding it into the taps leaves a
  residual modulation on the *decimated*-rate side — `m[mD]` on the analysis output,
  the mirror of it on the synthesis input. That is still 32 multiplies of length
  `T/D_b`; cheap, but 109 ms is an upper bound, not the saving. Second, a modulated
  kernel is **complex**, which kills `_split_real_imag` (`utils.py`). The entire complex
  win in the polyphase GEMM is that the FIR is real — one real multiply per tap instead
  of four — so a complex kernel turns each polyphase GEMM into four real GEMMs, and
  those GEMMs are now the dominant remaining cost. Measure that before writing any of
  the index algebra: it plausibly costs more than it saves. Note also that the synthesis
  half of this item now overlaps the fused alignment matrix P4 introduced.

### from the 2026-08-12 profiles

New, and separate from the P-numbered list above. Each was found while landing that
pass; the things that were prototyped and *lost* are in `ARCHIVE.md` under
"Decomposition optimisations that measured worse, or not at all (2026-08-12)" — read
that first, it is longer than this list.

- **numpy: `_numba_gfb_analyze` loop order** (`backend_numpy/gammatone.py:20-61`). The
  kernel runs `for sample: for band:`, so every input sample writes 32 output cells
  `num_samples * 16` bytes apart — 32 distinct cache lines per sample into a C-order
  `(bands, samples)` array. The bands are independent 4th-order recurrences, so the
  order is arbitrary; `for band: for sample:` writes each row contiguously and keeps the
  four states in registers across the row. Prototyped on real coefficients: 71.85 ->
  62.12 ms at n=120336 (1.16x), 5.86 -> 3.83 ms at n=8572 (1.53x). The kernel is 10.6%
  of a mono decomposition over 3 calls, so **~29 ms, ~1.8% end-to-end** — small, and
  worth stating as small. **Bit-identical** (verified: `tobytes()` equal on output and
  mutated states; reordering independent recurrences reassociates nothing).
  Effort: low, ~15 lines, no call-site change.
- **torch: the complex real/imag round trip**, `utils.py:235` and `:251`. Found
  independently by two profiling passes, which agree on the shape and disagree on the
  size: 0.071 s and 0.155 s per torch mono decomposition, so call it **4-8%**, and
  re-measure before sizing anything on it. `_split_real_imag` gathers the strided
  `.real`/`.imag` views into a fresh 2N buffer and `_merge_real_imag` allocates a new
  complex tensor, on every resample call. On the heavily decimated bands the copies
  dominate the arithmetic — for 1/1229, split+pad is 46% of the call.

  The tempting fix is `view_as_real`/`view_as_complex`, which are free views. **It
  probably does not work, and this is the specific reason**: `view_as_real` puts the
  real/imag axis *last*, which is exactly where `unfold` needs the time axis, so the
  permute forces the copy straight back. Recovering this likely means restructuring the
  kernel layout rather than swapping two calls. Class: reassociation (~1 ULP, GEMM shape
  changes). Effort: medium-high, confidence it pays: low.
- **torch: `F.pad` copies the whole signal to add a small margin**, `utils.py:285`,
  `:298`, `:300`, `:384`. Also found twice, at 0.051 s and 0.092 s over 298 calls
  (~4-5%). For 1/1229 the leading pad is ~13.5k samples against 121.5k of signal. Each
  call allocates a full padded copy purely to give `unfold` `half_length_factor` zeros
  of headroom at each end; an interior region needs no pad at all, with only the ~20
  edge samples handled separately. Done carefully each output sample keeps its tap
  order, so bit-identical — in practice the GEMM blocking shifts, so ~1 ULP. Effort:
  medium, and the edge cases are fiddly in the two hottest routines.
- **torch: fold the analysis modulation into the band gather** (`decomposition.py:407`
  then `:420`). The mirror of what P4 just fixed on the synthesis side: a full pass over
  a 62 MB mono / 123 MB stereo block immediately followed by a gather-copy of every band
  into per-group blocks. `block = subbands_output[..., band_indices, :] *
  modulation_matrix[band_indices, :]` is **bit-identical** — each band is in exactly one
  decimation group, so every element gets the same single multiply, asserted with
  `torch.equal` on all four components in both layouts.

  **It is a measured speed dud** — ten paired interleaved A/B replications came out
  mixed-sign and mostly inside 1 sem of zero, medians 0.999x/1.002x. Take it only for
  the memory: one fewer full-block complex128 temporary, 61.6 MB mono / 123.2 MB stereo,
  which is the same ground `_FFT_CHUNK_BUDGET_BYTES` was landed on. Effort: trivial.
  Recorded mostly so nobody re-prototypes it expecting P4's 2.6x.
- **torch: call the numpy backend's Numba resampler kernels on the no-grad CPU path.**
  Precedented, not speculative: the torch adaptation recurrence already does exactly
  this (`ARCHIVE.md`, 2026-08-08, ~2.2-2.3x, bit-identical on the reference platform),
  so the pattern, the optional-dependency handling and the forward-only constraint are
  all already solved in this repo.

  **Do not size this from the 3.3x in the rejected list.** That number is the ratio
  between two *numpy* implementations — numpy's Numba AXPY kernel against a numpy
  polyphase GEMM. The separate torch result is that a polyphase GEMM beat an *FFT
  convolution*. The two comparisons share no common leg, so they do not chain into an
  ordering, and torch's GEMM is not even the same algorithm as the numpy GEMM that lost
  (different layout, split real/imag rows, shifted-diagonal sum). Nobody has ever
  benchmarked numpy's Numba kernel against torch's GEMM head to head.

  The ceiling is the resampler's share of torch runtime — 37% by cProfile, itself an
  upper bound — so even an infinitely fast kernel caps near 1.6x. Constraints inherited
  from the adaptation-loop precedent: forward-only, so the differentiable torch path
  must stay for training; CPU-only; and Numba is an optional dependency, so this is a
  fast path and not a replacement.

  **The deciding measurement is cheap and nobody has run it**: A/B
  `_numba_polyphase_interpolate`/`_decimate` against torch's `_polyphase_interpolate`/
  `_decimate` at the real filterbank rates, with `benchmarks/ab.py`. Do that before
  writing any integration. Note one discouraging data point already in hand — the
  `benchmarks/ab.py` demo A/Bs the Numba polyphase path against SciPy's `upfirdn` at
  `up=2` on `(8, 16000)` and the Numba kernel is **0.57x, i.e. slower**. That is a small
  pure-interpolation case at a rate the filterbank never uses, so it is not decisive,
  but it does mean "the Numba kernel is simply faster" is not a safe assumption at every
  shape. Effort: low to measure, medium to integrate.
- **torch: the least-squares *assembly*, not the solve** (`decomposition.py:250-258`).
  `perform_time_varying_least_squares_projection_torch` is 21% of mono at 0.247 s
  cumulative, of which only 0.015 s is `cholesky_ex` — so it is almost entirely data
  movement and the Gram/RHS GEMMs. Two angles, both weak, both distinct from the
  archived Levinson and sliding-Gram entries (those attack the solve). `:250`
  materializes the big Toeplitz block flipped (0.022 s); flipping the far smaller solved
  weights instead is a pure permutation and bit-identical, but the `.reshape` at `:252`
  would force a materializing copy anyway, so the win is plausibly zero. And the Gram at
  `:257` is Hermitian PSD built with a full batched GEMM, ~2x the necessary flops — but
  torch exposes no batched HERK, so it needs a custom kernel. Note the numpy backend has
  the identical opportunity and it was reasoned out as a near-certain loss there for the
  same reason (`ARCHIVE.md`, 2026-08-12 rejected list).
- **torch: complex input designs the Kaiser filter twice per rate** (`utils.py:519`).
  The dispatcher calls `get_resample_filter_torch(up, down, x.dtype, ...)` on every
  call, so complex subbands cause a *complex* copy of the filter to be designed and
  cached per rate — used only for `filter_length` and `n_pre_remove`, which only the FFT
  fallback needs. The cache holds 132 entries for 64 distinct rates; the 1229 band's
  filter is 24581 taps, ~393 KB complex. Costs memory rather than time, plus 198
  pointless cache lookups per decomposition. Take the reduced rates from `math.gcd` and
  fetch the filter only on the fallback branch. **Bit-identical.** Effort: low (~15 min).
- **decide deliberately: is `_fft_resample` still worth keeping?** (`utils.py:398`, its
  spectrum cache at `:96`, the size guard at `:460`.) Since 2026-08-12 nothing reaches
  it in practice — `taps` is bounded by `2*half_length_factor + 2`, so the polyphase
  intermediate is at most ~22x the output and cannot blow up the way `in_len * up` does,
  and no realistic case was found that trips the guard. Deleting buys ~90 lines and one
  `lru_cache`. Two reasons not to: `next_fast_fft_length` (`utils.py:29`) must stay
  regardless, since `auditory_model.py:134` uses it; and the mixed-rate tests use
  `_fft_resample` as their independent oracle, which is the strongest check in that
  file. This is a "decide and write down why" item, not a perf win.
- **low confidence, unexamined**: `decomposition.py:177` and `:250` do
  `.unfold(...).flip(-1)`, and a flip on an unfolded view materializes the whole
  expanded window tensor — 35 calls, 0.026 s, inside
  `perform_time_varying_least_squares_projection_torch` (0.346 s cumulative).
  `_get_polyphase_kernel` folds its own index reversal into the filter at design time
  and the same trick may apply, but nobody has looked at the surrounding algebra.

### correctness, not perf

- **Backend-dependent onset transient on short synthetic inputs** — found 2026-08-15
  while tightening thresholds, documented in place at
  `tests/regression/test_differential_numpy_vs_torch.py:213`. On the synthetic 2-source
  pair in `test_differential_decomposition_pipeline`, torch's `target_distortion` and
  `interference` carry a transient in the **first ~60 samples peaking at ~1e3, against a
  numpy RMS of ~1e-1**. Their individual correlations against numpy are **0.127 and
  0.110**.

  Why it has stayed invisible: the transient *cancels* between those two components, so
  both backends still reconstruct the estimate to the same 1.1 max error and every
  algebraic-reconstruction invariant passes. And that test only ever asserted
  `true_target`, which is unaffected. The same two components measure 1e-11 parity on the
  MATLAB reference WAVs, so it is specific to short synthetic input rather than a general
  divergence — most likely a filter/solver warm-up at the very start of the signal that
  the 5 s reference clip dilutes to nothing.

  This matters beyond tidiness: short clips are exactly the regime the reflection-padding
  item is about, and exactly where `reference/` is meant to start generating ground
  truth. Root-cause before either. Do not widen that test to the other components until
  it is understood — it will fail, and correctly.

- **Three fragile absolute tolerances, 1.2-1.6x of margin** (the narrowest in the suite),
  all in `test_differential_numpy_vs_torch.py` and marked `FRAGILE -- do not tighten`:
  mixer gains `:353` (measured 8.31e-7 vs `atol=1e-6`), FFT-vs-IIR analyzer `:327`
  (7.38e-7 vs 1e-6), synthesizer delay/phase `:382` (3.15e-6 vs `atol=5e-6`). Also
  `test_numpy_gammatone.py:381` DC rejection at 5.05e-3 against 1e-2.

  These are the most likely source of an ubuntu-latest-only CI failure in the whole
  suite, and the root cause is a **units mismatch, not a too-tight number**: each is a
  genuine ~1e-6 *relative* algorithmic difference being checked against an *absolute*
  tolerance, so the margin depends on the signal's peak. The principled fix is to express
  them as `rtol` rather than raise `atol` — that is a correctness fix in how the
  comparison is written, not a weakening. Raising `atol` to ~1e-5 is the quick
  alternative. Deliberately left alone for now because both directions widen a currently
  passing bar, which is a judgement call rather than a measurement.

- The torch gammatone's accuracy floor is its **wrap guard, not its FFT length**:
  against a 4x-padded oracle the current transform is 5.4e-6 relative, driven by a
  0.2 s guard against a designed `pad_len` of 4800. Raising `pad_len` matters far more
  than any FFT sizing change (see the dropped sizing item in `ARCHIVE.md`).
