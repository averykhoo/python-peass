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
3. **Measure, don't estimate** — and be careful what you claim from one machine. Two
   entries in `ARCHIVE.md` exist because a platform-specific observation was written
   down as a universal invariant.

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

## perf ideas not yet taken

From the decomposition-focused profiles of 2026-08-09 and 2026-08-10. All prototyped
and measured on `tests/resources/database/exp01_*` unless marked hypothesis, and all
subject to rule 1 — efficiency and SIMD, no fanning out.

What landed, and what was tried and rejected, is in `ARCHIVE.md` rather than here —
check it before reviving anything, particularly FIR symmetry folding, Levinson, and
the batching experiment that the polyphase GEMM obsoleted. The P-numbers below start
at P4 because P1-P3 of the 2026-08-09 list are archived there too.

**Re-measure before trusting the numbers below.** They were sized against a
decomposition in which resampling was 60% of runtime; it is 33% now, so the
end-to-end value of P4 and P5 especially has shrunk.

### torch

- **P4, was 1.08x — fuse the synthesis modulation/phase/delay/gain chain**
  (`decomposition.py:459` + `gammatone.py:238-256`). Currently four full passes over a
  246 MB block plus a gathered index tensor. Cache `mod_matrix * phase_factors`, use
  `x.real*c.real - x.imag*c.imag` instead of a complex multiply then `.real`, and
  replace `gather`/`where`/`einsum` with a 32-iteration shift-accumulate. Measured
  298.6 ms -> 135.3 ms (2.21x) in isolation, deviation 4.3e-16, but allocator-bound so
  worth less end-to-end than that implies. Effort: medium.
- **P5, hypothesis, removes 109 ms — fold the modulation into the polyphase filters**
  (`decomposition.py:382` and `:459`). Complex-exponential modulation distributes
  through convolution exactly, so the two full-length modulation multiplies (60.0 ms
  analysis + 48.6 ms synthesis) disappear into the 21-tap filters, and the two cached
  modulation matrices (61 MB + 62 MB) can be dropped. Exact in real arithmetic,
  expected ~1 ULP. Effort: high — the index algebra needs careful per-band validation.
- **Mixed-rate calls still take the FFT path** (`utils.py`), new from the 2026-08-10
  pass. After the polyphase GEMM the largest single resample entries are the 2/3 and
  3/2 conversions either side of the filterbank (~12-27 ms each, ~90 ms total, ~7% of
  the decomposition). A general polyphase form covers them too, but the index algebra
  is genuinely mixed-rate rather than the clean `up == 1` / `down == 1` cases, so it
  was left alone. Effort: medium.

### numpy

- **Synthesis scatter in place** (`decomposition.py:706-721`), ~93 ms of ~246 MB of
  copies that `fast_resample_poly` could write directly into. Effort: small.

### correctness, not perf

- The torch gammatone's accuracy floor is its **wrap guard, not its FFT length**:
  against a 4x-padded oracle the current transform is 5.4e-6 relative, driven by a
  0.2 s guard against a designed `pad_len` of 4800. Raising `pad_len` matters far more
  than any FFT sizing change (see the dropped sizing item in `ARCHIVE.md`).
