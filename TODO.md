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
2. **Bit-identical changes land; near-exact ones get documented instead.** That is why
   the torch decomposition stack below is measured, prototyped, and still unlanded at
   2.10-2.45x. If you want to change that bar, it is a decision to raise explicitly,
   not to assume.
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
  recurrence (the decomposition's pinv backward is cheap by comparison). The gradient
  path cannot use the fast forward paths since it needs the straight-through max, so
  it runs far slower than the no-grad path. A custom `autograd.Function` with a
  hand-derived analytic backward for the 5-stage cascade would make training-scale
  backprop practical; it's substantial and needs careful gradient validation. Note
  the 2026-08-08 Numba kernel does **not** help here — it is forward-only by design.
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

## perf ideas not yet taken (decomposition-focused profile, 2026-08-09)

All prototyped and measured on `tests/resources/database/exp01_*` unless marked
hypothesis. Ideas that were tried and *failed* are in `ARCHIVE.md` rather than here;
check there before reviving anything, particularly FIR symmetry folding and Levinson.

**Updated 2026-08-10.** The bulk of this list has landed — torch P1 and P2 and both
numpy kernel items, together worth 2.25x mono / 2.06x stereo on torch and 1.47x on
numpy. See `ARCHIVE.md` "Decomposition: torch ~2.2x, numpy ~1.47x". P3 was implemented,
measured, and reverted; it is in `ARCHIVE.md` too. What remains below is what is still
genuinely open, with numbers that now need re-measuring against the faster baseline —
in particular P4 and P5 were sized as fractions of a decomposition in which resampling
was 60% of runtime and is now 33%, so their end-to-end value has shrunk.

Everything here must respect the no-threads/no-subprocess constraint (see
`ARCHIVE.md`) — efficiency and SIMD only.

### torch

- **P4, was 1.08x — fuse the synthesis modulation/phase/delay/gain chain**
  (`decomposition.py:412` + `gammatone.py:204-222`). Currently four full passes over a
  246 MB block plus a gathered index tensor. Cache `mod_matrix * phase_factors`, use
  `x.real*c.real - x.imag*c.imag` instead of a complex multiply then `.real`, and
  replace `gather`/`where`/`einsum` with a 32-iteration shift-accumulate. Measured
  298.6 ms -> 135.3 ms (2.21x) in isolation, deviation 4.3e-16, but allocator-bound so
  worth less end-to-end than that implies. Effort: medium.
- **P5, hypothesis, removes 109 ms — fold the modulation into the polyphase filters**
  (`decomposition.py:335` and `:412`). Complex-exponential modulation distributes
  through convolution exactly, so the two full-length modulation multiplies (60.0 ms
  analysis + 48.6 ms synthesis) disappear into the 21-tap filters, and the two cached
  modulation matrices (61 MB + 62 MB) can be dropped. Exact in real arithmetic,
  expected ~1 ULP. Effort: high — the index algebra needs careful per-band validation.

### numpy

- **Synthesis scatter in place** (`decomposition.py:494-505`), ~93 ms of ~246 MB of
  copies that `fast_resample_poly` could write directly into. Effort: small. This is
  the only numpy item from the 2026-08-09 pass that has not landed.

### torch, new from the 2026-08-10 pass

- **Mixed-rate calls still take the FFT path** (`utils.py`). After the polyphase GEMM
  the largest single resample entries are the 2/3 and 3/2 conversions either side of
  the filterbank (~12-27 ms each, ~90 ms total, ~7% of the decomposition). A general
  polyphase form covers them too, but the index algebra is genuinely mixed-rate rather
  than the clean `up == 1` / `down == 1` cases, so it was left alone. Effort: medium.

### correctness, not perf

- The torch gammatone's accuracy floor is its **wrap guard, not its FFT length**:
  against a 4x-padded oracle the current transform is 5.4e-6 relative, driven by a
  0.2 s guard against a designed `pad_len` of 4800. Raising `pad_len` matters far more
  than any FFT sizing change (see the dropped sizing item in `ARCHIVE.md`).
