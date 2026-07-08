# PEASS Port — Comprehensive Code Review (2026-07)

Correctness-first review of the whole `peass/` package, then performance. Ground
truth: the Emiya 2011 paper, the official MATLAB PEASS v2.0.1 source (verified via
the CVSSP/peass-software GitHub mirror), and the MATLAB `.wav` test vectors in
`tests/resources/matlab_reference/`. The port is not expected to be bitwise
identical to MATLAB, but the algorithm must be correct and the output close.

Method note: findings were verified empirically (running numpy code, replicating
`torch` conv/resample semantics in numpy since torch is not installed in the dev
env) and by cross-checking the MATLAB source. Correlations below are per-component
against the MATLAB gold `.wav` files for the stereo reference clip.

---

## Summary

| Severity | Finding | Status |
|---|---|---|
| CRITICAL | Torch haircell lowpass applied time-reversed (missing `conv1d` kernel flip) | **FIXED this session** (`c2c7e76`) |
| MEDIUM | Reduced-order (3×) resample filter was the dominant deviation from MATLAB (subband resamples) | **RESOLVED this session**: default now full-order 10× (corr → 0.99999), configurable via `DecompositionConfiguration.resample_filter_half_length_factor` |
| LOW | Sequential independent loops leave ~2× wall-clock on the table | Follow-up (perf) |
| LOW | numpy predictor re-decodes the estimate file just for its sample rate | Safe fix |
| LOW | Unbounded `lru_cache` on the synthesis modulation matrix (memory growth) | Safe fix |
| LOW | Misc robustness/naming (see below) | Note only |
| — | numpy DSP core, auditory model, metrics, predictor; torch parity (post-fix) | **Verified correct** |

No correctness bugs were found in the numpy backend. The one critical bug was in
the torch backend and is fixed. The remaining substantive item is a
speed-vs-MATLAB-accuracy tradeoff, which is a design decision.

---

## CRITICAL (fixed)

### C1. Torch haircell lowpass ran time-reversed — `backend_torch/auditory_model.py`
`F.conv1d` is cross-correlation (no kernel flip), so the decaying FIR kernel must
be flipped to implement the causal one-pole lowpass that matches numpy's
`lfilter([b0], [1, -gain])`. Without the flip the haircell transduction ran
backwards, collapsing auditory-representation parity with numpy to ~0 (corr
0.0015) and corrupting all four perceptual scores (OPS/TPS/IPS/APS) on the torch
backend. The BSS energy ratios were unaffected (they don't use the ear model).
Verified in numpy by replicating conv1d semantics: unflipped vs reference IIR
correlates at 0.0015, flipped matches to 4e-16. **Fixed** by adding `.flip(-1)` to
the kernel (the resampler in `utils.py` already used this convention). The
differential auditory-representation parity test (added this session) guards it in CI.

---

## MEDIUM (decision needed)

### M1. Reduced-order resample filter dominates the deviation from MATLAB
`backend_numpy/gammatone.py` `get_resample_filter` uses `half_len = 3*max_len`
(vs scipy's `10*max_len`) for anti-aliasing. Measured effect on the stereo
reference clip, sweeping the filter half-length factor:

| Factor | true corr | artif corr | RMS ratio vs MATLAB | decompose time (5s stereo) |
|---|---|---|---|---|
| 3 (current) | 0.99804 | 0.98985 | ~0.938 (−6%) | 4.50 s |
| 5 | 0.99937 | 0.99655 | ~0.97 (−3%) | 5.09 s (+13%) |
| 7 | 0.99984 | 0.99913 | ~0.987 (−1.3%) | ~5.5 s |
| 10 | 1.00000 | 1.00000 | 1.0026 (~exact) | 6.01 s (+33%) |

Key correction to an earlier hypothesis: the deviation comes from the **subband**
resamples (many, inside the analysis/synthesis filterbank), **not** the two
broadband full-band resamples — verified by a controlled experiment (full-order on
broadband only left corr unchanged at 0.99; full-order everywhere reached 1.0).

Interpretation: the current 3× filter is an aggressive speed optimization that
introduces a systematic ~6% energy deficit and is why correlation sits at ~0.99
rather than ~1.0. Because the deficit is roughly uniform across components, the
BSS *ratios* (SDR/ISR/SIR/SAR) are largely unaffected (scaling cancels), but the
decomposed *waveforms* and the perceptual scores that consume them can shift.

**RESOLVED:** made configurable via `DecompositionConfiguration.resample_filter_half_length_factor`,
defaulting to full-order `10` (near bit-exact MATLAB match: min component
correlation on the reference clip went 0.98985 → 0.999997). Users who need the
old speed can set it to `3` (corr ~0.99, ~25% faster decompose). Threaded through
both the numpy and torch decomposition paths; the auditory-model resampling uses
the same full-order default. The perceptual scores shifted with the more accurate
decomposition (e.g. OPS 15.38 → 17.67); the characterization test values were
updated accordingly, and the numpy regression correlation gate tightened to 0.999.

---

## Verified CORRECT (suspicions investigated and cleared)

- **Multichannel `np.min` aggregation** (`metrics.py:207-210`): confirmed against
  the MATLAB source (`audioQualityFeatures.m`), which uses `min(qTarget)` etc. —
  "the worst value over all channels." Correct.
- **Reference combinations** for the four features (`true+interf+artif` → target
  quality, etc.): match the MATLAB source exactly.
- **Haircell `2000.0` constant**: `exp(-π·2000/fs) = exp(-2π·1000/fs)` — a correct
  1 kHz one-pole (measured −3 dB at 1013 Hz). Not a bug.
- **numpy DSP core** (`decomposition.py`, `gammatone.py`): the strided Toeplitz +
  `[:, ::-1]` (bit-exact vs `scipy.linalg.toeplitz`), the Gram/RHS reformulation,
  the posv-solve + pinv fallback, the overlap-add windowing/normalization and the
  `[:-(window_length-1)]` crop, the gammatone coefficient math, the numba 4th-order
  kernel (bit-identical to the lfilter cascade), delay/phase alignment, and mixer
  gain optimization — all bit-exact or provably equivalent to MATLAB/Hohmann. No
  off-by-one errors.
- **numpy auditory model / metrics / predictor**: adaptation-loop time constants
  and threshold cascade, the similarity framing/reshape (order-invariant under the
  downstream reductions), PEMO-Q mean-subtraction and energy weighting, the
  assimilation rule (0.25/0.75), log-mapping + MLP + feature→score ordering — all
  correct.
- **torch parity** (post-C1): FFT gammatone == numpy IIR transfer function; the
  polyphase resampler (`conv1d`/`conv_transpose1d`) matches `scipy.upfirdn`
  bit-for-bit on the pipeline ratios; NumFrames closed-form == numpy while-loop
  count; Toeplitz/reshape ordering matches numpy's `order='F'` (safe because
  `num_estimates==1` is genuinely guaranteed); adaptation loop, modulation
  filterbank, metrics, and MLP all equivalent within accepted surrogate tolerances.
- **`lru_cache` keying**: all caches keyed on the complete set of determining
  inputs; no staleness or mutable-array hazards (returned arrays used read-only).

---

## LOW (note / optional)

- **Dead multichannel-flatten path** (`backend_numpy/auditory_model.py:273-276`):
  if ever reached with a stereo array, `.ravel()` interleaves samples (garbage
  representation). Currently unreachable (single channels always passed). Latent.
- **`root_mean_square_energy` is mean-square, not RMS** (`metrics.py:145`): no
  `sqrt`. Almost certainly the intended energy-based salience weighting (matches
  the reference), so likely just a misleading name — worth a one-line confirmation.
- **Div-by-zero for `fs < 10`** (`metrics.py:98`): `frame_length = int(0.1*fs) = 0`.
  Not reachable in the real pipeline (fs is 100/800 there). Low.
- **`int()` vs `round()` for `frame_length`** (`metrics.py:98`): harmless at
  pipeline rates.
- **Torch softplus floor overshoot** near the absolute-threshold noise floor
  (`backend_torch/auditory_model.py:82`): ~7e-4 error at `val≈abs_thresh`; within
  accepted surrogate tolerance (energy-weighted downstream).
- **Torch `1e-15` eps** in gammatone slope/gain normalization
  (`backend_torch/gammatone.py:61,79`) vs numpy's `finfo(float).eps`: negligible
  (dwarfed by the 1e-6 parity tolerance). Optional to align for consistency.

---

## Performance (measured: 5s 16kHz stereo, warm numba)

Total `predict_perceptual_evaluation_scores` ≈ 3.6 s (decompose 71%, metrics 23%).
No single 80% hotspot; time is spread across `upfirdn` resampling (24%), the
least-squares solve cluster (21%), and two GIL-holding numba kernels (17%).

1. **Parallelize the independent outer loops** (highest impact-to-risk):
   analysis ×(sources·channels), synthesis ×channels, and metrics ×(5·channels)
   all run sequentially on one core though each iteration is independent (the code
   even flags it). To make it pay off, add `nogil=True` (or `parallel=True` +
   `prange` over the band axis) to the numba kernels so the GIL stops serializing
   them, then dispatch the loops over a thread pool. Estimated ~1.8–2.4× wall-clock
   on 4 cores. **Correctness risk: low** (iterations are fully independent) — but
   must validate bit-identity vs the serial path and keep the numba cache warm.
2. **`predictor.py` re-decodes the estimate file** just for its sample rate
   (`sf.read` after decomposition already read it) — use `sf.info(...).samplerate`.
   Correctness risk: none. (Safe fix — applied this session.)
3. ~~**`scipy.linalg.solve(..., check_finite=False)`**~~ — tried; benefit is below
   the measurement floor on 2337 tiny matrices. Not worth it (see investigation
   below).
4. **Unbounded `lru_cache`** on `_get_synthesis_modulation_matrix_cached` (keyed on
   signal length): memory growth across varying-length batches. Cap `maxsize`.
   Correctness risk: none. (Safe fix — applied this session.)
5. **complex64/float32** through the gammatone/modulation path: ~10–20% but
   **medium/high** correctness risk on a precision-sensitive reference port. Not
   recommended without careful tolerance validation.
6. **Torch** (profiled with a CPU runtime; 5 s stereo clip):
   - **FIXED — haircell OOM.** The FIR haircell used `F.conv1d`, whose im2col is
     `O(B*K*L)`; on the real clip the batched metrics make a `(10,27,120000)`
     tensor → a **62 GB** allocation that aborted. The torch predictor could not
     run on any realistic-length signal (tests only used 0.5 s clips). Replaced
     with FFT convolution (`O(B*L)`, bit-identical to conv1d at 1e-15). This was a
     robustness bug, not just perf.
   - **DONE — adaptation loop unroll.** The nonlinear 5-stage adaptation recurrence
     is the dominant torch cost (~93% of the auditory model, ~34 s). It's
     inherently sequential (can't parallelize the time axis), but unrolling the
     5-stage inner loop to drop a per-timestep `torch.stack` gave ~1.35×
     (40.4 s → 30.0 s), bit-identical.
   - **Still slow**: even after the above, a 5 s clip takes ~30–50 s in torch
     (sequential recurrence at the 24 kHz upsampled rate). The torch backend is
     for differentiable *training* on short segments; batch scoring of long clips
     should use the numpy backend. `torch.compile` on the loop is a possible
     future lever but speculative (deprecation-migration territory, uncertain
     bit-parity). `.item()`/`.tolist()` device syncs still matter on GPU.
   - **FINDING — score divergence on long clips.** On the 5 s reference clip the
     torch scores differ from numpy by up to ~5 points (OPS 14.2 vs 17.7), the
     accumulated effect of the differentiable surrogates (softplus max, FIR
     truncation) — only observable now that the predictor runs at length. Within
     the "matches by correlation, not precision" contract, but larger than the
     short-clip tests imply; worth a dedicated parity study on realistic lengths.

### Hands-on performance investigation (2026-07, torch installed)

Re-profiled `predict_perceptual_evaluation_scores` on the stereo reference clip
(~8 s warm, 8-core box) after the full-order (10×) resampling became the default.
The profile shifted substantially:

| Stage | Share | Note |
|---|---|---|
| `upfirdn` resampling | **~43% (3.6s)** | now dominant — the cost of full-order (10×) MATLAB-fidelity resampling |
| LS solve cluster (`perform_least_squares_projection` ×2337 + `solve` + `hstack`) | ~15% | |
| `_numba_gfb_analyze` | ~7% | |
| `_numba_fused_auditory_kernel` | ~6% | |

Tried and **rejected** (measured against a within-process bit-parity anchor):
- **numba `parallel=True` + `prange` over bands** on the four kernels
  (`_numba_gfb_analyze`, `_numba_delay_process`, fused auditory, adaptation): the
  kernels are called many times on small band counts (~30), so thread-spawn
  overhead plus contention with MKL/`upfirdn` made the whole pipeline **slower**
  (8.8s vs 8.0s), and introduced a ~5e-11 drift. Reverted.
- **`scipy.linalg.solve(check_finite=False)`**: safe (inputs provably finite) but
  the benefit on 2337 tiny matrices is below the measurement floor. Reverted —
  no evidence of a win. Note: within a process the pipeline is bit-deterministic
  (spread 0.0 over repeated runs); the ~5e-11 variation is only *cross-process*
  MKL BLAS thread-scheduling, which swamps sub-5% optimizations.

Conclusion: the pipeline is already well-optimized at the micro level, and the
dominant cost is now the fidelity-driven resampling, which cannot be reduced
without either lowering `resample_filter_half_length_factor` (trades MATLAB
correlation) or a risky bit-inexact FFT-resampler rewrite. The remaining
substantial levers all trade something:
- **Batch the per-frame LS solves** (like the torch backend does with `unfold` +
  batched `pinv`) — ~15% of runtime — but batching forfeits the per-frame
  posv→pinv ill-conditioning fallback (49/2337 frames use it), so it changes the
  numerics by ~1e-6 (still ~0.9999 vs MATLAB). Small correctness trade.
- **Batch the analysis/synthesis across sources+channels** (like torch) — reduces
  the 548 `upfirdn` call overheads but not the dominant filter *compute*, so a
  limited win for a medium-risk refactor.
- **Thread the analysis/synthesis loops** — `upfirdn` releases the GIL and is now
  43%, so threading the independent per-(source,channel) passes could give
  ~1.3–1.5× (the numba parts stay GIL-serialized). This is the "parallelization"
  item deferred earlier; medium risk on the Windows MKL/OpenMP stack.

---

## Recommendation: this session vs. a new one

- **Done:** C1 (critical torch fix); M1 (full-order resampling as the
  configurable default → MATLAB correlation ~0.99999); the two zero-risk
  perf/hygiene fixes (#2 predictor sample-rate, #4 cache cap); torch backend
  validated end-to-end with a CPU runtime (decomposition torch-vs-numpy-vs-MATLAB
  all ~1.0; auditory parity confirms the haircell fix); torch test cleanup.
- **Investigated & rejected:** intra-kernel `prange` (slower here) and
  `check_finite=False` (unmeasurable). No safe *micro*-optimization moves the
  needle; the dominant cost is the fidelity-driven resampling.
- **Available but trades something (your call):** the three structural levers in
  the investigation above — batched per-frame solve (~15%, ~1e-6 numeric change),
  batched analysis (limited win, medium risk), or threading the GIL-free
  analysis/synthesis loops (~1.3–1.5×, medium risk on Windows MKL/OpenMP). Best as
  a deliberate, benchmarked follow-up.
- **Note-only / optional:** the LOW items; address opportunistically.
