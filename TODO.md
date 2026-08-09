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
hypothesis. None of these are bit-identical, which is the only reason they did not
land — the bit-identical findings from the same pass are already in. Ideas that were
tried and *failed* are in `ARCHIVE.md` rather than here; check there before reviving
anything, particularly FIR symmetry folding and Levinson.

Everything here must respect the no-threads/no-subprocess constraint (see
`ARCHIVE.md`) — efficiency and SIMD only.

### torch (stacked: 2.10x mono / 2.45x stereo end-to-end, worst score deviation 3.4e-9)

Resampling is 55.6% of torch decomposition, and structurally so: there are 32 bands
with 32 *distinct* decimation factors, so the band grouping produces 32 groups of one
band each and every one of the 198 FFT convolutions runs on a batch of 1 or 2 rows.
torch's FFT parallelises over the batch dimension, so most threads idle (measured
3.9 ms/row at batch 2 vs 1.0 ms/row at batch >= 32).

- **P1, 1.86x e2e alone — polyphase GEMM instead of FFT convolution**
  (`backend_torch/utils.py:154-175`). 196 of the 198 calls have `up == 1` or
  `down == 1`, and with `half_length_factor = 10` the filter is 21 taps per polyphase
  phase *regardless of rate* — a small dense GEMM, not an FFT. Measured **6.67x**
  upsampling and **3.54x** decimating, worst relative deviation 1.09e-15, with no large
  intermediate (the FFT path materialises a 121500-point complex spectrum per band).
  Autograd verified: gradients match to 1.1e-15 and backward is not slower. Effort:
  medium, ~80 lines plus a dispatch; the two mixed-rate calls stay on the FFT path.
- **P2, 1.11-1.23x e2e — `cholesky_ex`/`cholesky_solve` instead of `torch.linalg.pinv`**
  (`backend_torch/decomposition.py:188`, and `:118` in the single-frame twin). The Gram
  is Hermitian PSD with min eig 3.4e-10 and pinv's rcond cutoff is ~7.6e-15, so
  **nothing is ever truncated** — it is an expensive way to solve `Gx = R`. Measured
  over the real 2338-matrix stack: pinv 409 ms, `pinv(hermitian=True)` 318 ms, `lstsq`
  92 ms, `ldl` 28 ms, **cholesky 21 ms (19.4x)**. `cholesky_ex` returns per-matrix
  `info`, giving the same rank-deficient fallback the numpy backend already has at
  `backend_numpy/decomposition.py:190-198` (`info != 0` fired 0 times in 2337 frames).
  Do **not** copy numpy's 1e-15 diagonal regularisation: against min eig ~3e-10 that is
  a 3e-6 relative perturbation, measured at 1e-5 relative deviation on the solution.
  Effort: low, ~8 lines. This is the cheapest real win on the list.
- **P3, 1.09x on top of P1 — merge the two analysis passes, batch the four synthesis
  components** (`decomposition.py:563-570` and `:621-628`). Sources and estimate go
  through two independent passes over an identical filterbank; `cat` them into one.
  Effort: low-medium.
- **P4, 1.08x on top of P1-P3 — fuse the synthesis modulation/phase/delay/gain chain**
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

### numpy (stacked with the landed changes: 1.54x end-to-end)

- **AXPY interpolate kernel, 4.4x on the kernel / 1.17x e2e**
  (`backend_numpy/gammatone.py:137-163`). The current `for output_sample: for tap in
  range(21)` inner reduction cannot be vectorised by LLVM — measured ~0.42 GMAC/s
  against a ~2.3 GMAC/s plain-loop ceiling. Invert the loops so `p` (length `up`,
  contiguous) is innermost, accumulating into a small L1 buffer: same FLOPs, fully
  vectorisable. Measured 209 ms -> 47 ms over the 32 real band shapes, max relative
  error 5.9e-16. Needs `rb.T` cached contiguous next to `get_polyphase_branches`
  (`gammatone.py:720-734`) and real/imag split buffers. Effort: medium, ~60 lines.
- **`ddot`-based decimate kernel, 2.1x on the kernel** (`gammatone.py:103-134`).
  Replace the hand-rolled tap loop with `np.dot(rf, X[b:b+num_taps])` on split
  real/imag buffers, which numba lowers to `ddot`. Measured 80.8 ms -> 37.9 ms at
  `MKL_NUM_THREADS=1`, so the gain is AVX2 vectorisation and not BLAS fan-out; max
  relative error 2.4e-15. Effort: small once the real/imag padding path above exists.
- **Write complex output directly from the numba interpolate kernel.** After the AXPY
  rewrite the kernel is only ~190 ms of the 417 ms stage; ~228 ms is the real/imag
  deinterleave and recombine in the wrapper. Effort: small, folds into the above.
- **Synthesis scatter in place** (`decomposition.py:494-505`), ~93 ms of ~246 MB of
  copies that `fast_resample_poly` could write directly into. Effort: small.

### correctness, not perf

- The torch gammatone's accuracy floor is its **wrap guard, not its FFT length**:
  against a 4x-padded oracle the current transform is 5.4e-6 relative, driven by a
  0.2 s guard against a designed `pad_len` of 4800. Raising `pad_len` matters far more
  than any FFT sizing change (see the dropped sizing item in `ARCHIVE.md`).
