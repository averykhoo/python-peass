# archive

Settled work: investigations that are root-caused, fixes that have landed, and options
deliberately declined. Nothing here is open — it lives outside `TODO.md` so the handoff
list stays short, and it is kept rather than deleted because the reasoning is what stops
the same ground being re-covered. Entries are append-only; if one is genuinely reopened,
move it back to `TODO.md`.

---

## Declined

### Numba `parallel=True`/`prange` on the NumPy kernels (2026-08-08)

Profiled and prototyped: annotating the five Numba kernels
(`backend_numpy/gammatone.py:20, 64, 103, 137` and `backend_numpy/auditory_model.py:27`)
with `parallel=True` and parallelising over bands — or over flattened `row x out_len`
for the two polyphase kernels — measured **1.63x at 1s, 1.87x at 5s mono, 1.83x at 5s
stereo**, and was **bit-identical** (`max|diff| = 0.0` on all four waveforms and all
four features, every case). The parallel axis is over genuinely independent work
items, so there are no cross-thread reductions to reassociate.

Declined anyway, on a design constraint rather than a numerical one: this backend
promises that **no threads or subprocesses are spawned** (README, "NumPy backend
performance and numerical reproducibility"), so that the library stays predictable
inside a caller that is already parallelising and remains trivially thread-safe.
Performance here is to come from efficiency — SIMD, better memory traffic, fewer
per-call overheads — not from fanning out. To be explicit, since it is a natural
objection: Numba's `prange` is *not* Python threading (it releases the GIL and uses a
native threading layer), but it still spawns OS threads, which is exactly what the
constraint excludes.

Reopen only if the no-threads promise is deliberately retired. For whoever does: Numba
selected **TBB** on this machine, so no second `libiomp5md.dll` and no clash with
torch (verified: import torch, run the parallel kernels, then a torch matmul). With
`NUMBA_THREADING_LAYER=workqueue` — the OpenMP-free fallback — the speedup was still
1.80x, so pinning the layer costs almost nothing and removes the OpenMP risk.

Amended 2026-08-09: the promise this was declined under is narrower than it read. It
covers *this package's own code*; BLAS threading is inherited from the user's NumPy,
and on a stock MKL build the backend already runs multi-threaded inside BLAS (cpu/wall
3.86-4.05, dropping to ~0.9 under `MKL_NUM_THREADS=1`). The README now says so
explicitly. This does not reverse the decision — the distinction between "the library
fans out" and "the library calls a BLAS the user configured" is exactly the one that
keeps behaviour predictable and controllable from outside — but anyone reopening this
should argue against the real invariant rather than the overstated one.

### `segmentation_factor` — fail loudly instead of porting (2026-08-05)

`DecompositionConfiguration.segmentation_factor` now raises `NotImplementedError` in
`__post_init__` for any value other than `1` (`peass/config.py`, guarded by
`tests/unit/test_config.py`). The MATLAB path was **not** ported, by decision: it is not a
quality knob — MATLAB documents it purely as a peak-memory relief valve ("increase this
integer if you experienced out of memory problems", `example.m:26`) — and implementing it
would mean validating decomposition accuracy across the whole reasonable range of the
parameter for no accuracy gain. Failing loudly costs nothing and gives a MATLAB user
chasing an OOM the full story from the traceback.

Reopen only if someone actually hits an out-of-memory decomposition. The port spec, for
whoever does:

MATLAB does the work in `extractDistortionComponents.m`. The `options.segmentationFactor > 1`
branch at ~lines 107-110 short-circuits into `aux_segmentAndDecompose` (~lines 270-386),
which cuts every source and the estimate via `aux_cutWav` into windows of
`N = 2*round(TCut*fs/2)` samples at a 50% hop, where `TCut = ceil(nSamples/segmentationFactor)/fs`
— so the actual segment count is ~`2*segmentationFactor - 1` overlapping windows, not
`segmentationFactor` disjoint ones. It recurses with `segmentationFactor = 1` and with
`shadeInMs`/`shadeOutMs` forced to 0 except on the first/last segment respectively, then
`aux_mergeWav` overlap-adds each of the four components under `flipud(hann(N,'periodic'))`
and divides by the accumulated window where it is non-zero. The config field already
exists, so the public API surface would not change — it is purely a decomposition-path
implementation, plus dropping the `__post_init__` guard and its test.

---

## Closed investigations (do not reopen)

### Decomposition optimisations that measured worse, or not at all (2026-08-09)

From a decomposition-focused profile of both backends. Each of these is the kind of
idea that looks obviously right on paper, which is exactly why the measurement is
worth keeping:

- **FIR symmetry folding in the NumPy polyphase kernels.** The resampling FIR is
  linear-phase and the taps *are* exactly symmetric for the decimating case
  (`np.array_equal(rf, rf[::-1])` holds for every band). Folding the symmetric pairs
  to halve the multiply count measured **38.6 ms vs 38.4 ms — no gain at all**. On FMA
  hardware two folded taps cost one add plus one FMA, which is the same two operations
  as two plain FMAs; folding only halves *coefficient* loads, and those are not the
  bottleneck. It would have cost 2.9e-16..1.3e-15 of deviation for nothing. For the
  interpolator it does not even apply: branches `p` and `up-1-p` are reverses of each
  other, which is coefficient sharing, not an arithmetic reduction.
- **Levinson-Durbin on the per-frame systems.** Dead: the `w^2` analysis window
  destroys the Toeplitz structure. Relative spread along the diagonals of the Gram's
  (0,0) block over all 2338 real frames measured median 7.8e-2, min 6.7e-3, max 6.4e-1.
- **Sliding / rank-update Gram across overlapping frames.** Frames overlap 75%
  (`hop = window/4`), but `w^2` is frame-anchored, so `Gram_{f+1}` is not a rank update
  of `Gram_f`. There *is* a real route -- `hann(N)[t] = 0.5 - 0.25e^{i2*pi*t/N} -
  0.25e^{-i2*pi*t/N}`, and at `hop = N/4` the frame-dependent phase collapses to a
  constant `(-i)^f`, making each of three sums a sliding window updatable in O(hop).
  Not worth it: it touches only Gram+RHS, 52 us of the 280 us frame, so the ceiling is
  ~4.5% end-to-end, bought with prefix-sum cancellation on an 8572-sample running total.
- **Removing the redundant fancy-index copy** at `backend_numpy/decomposition.py:425-433`.
  It copies a whole 1.9 MB row per band (82 ms total) where a zero-copy slice would do.
  Replacing it is bit-identical and **consistently 10% slower** (2.75/2.85/2.75 s vs
  2.49/2.50/2.54 across three independent passes): the "wasted" copy acts as a
  sequential prefetch of the subband row ahead of the strided padded fill. Leave it.
- **Short-FFT spectrum tiling in the torch resampler** (previously an open TODO item).
  Superseded and its mechanism was misdiagnosed. Tiling is exact only when `up` divides
  the transform length, which holds for **3 of 32 bands**; forcing `N = up*P` makes `N`
  inherit `up`'s prime factors, measured 2.19x (u=27) down to **0.77x (u=139)**. The
  earlier note blamed the prime 107 specifically -- it is general. It also cannot
  coexist with batching, and the polyphase GEMM route beats it by ~3x uniformly while
  being exact rather than near-exact.
- **`next_fast_fft_length` in the torch gammatone analysis** (previously an open TODO
  item). Dropped on three counts. The quoted number was wrong for this clip
  (`T + pad = 124800`, so `next_fast_fft_length` gives 125000, not 124416); it measured
  inside noise and slightly *slower* in a full ablation; and it **costs accuracy**,
  because shrinking the transform shrinks the IIR wrap guard from 11072 to 5000 samples
  against a designed `pad_len` of 4800, moving worst score deviation from 2.3e-9 to
  **6.1e-4**.

### Micro-optimisations measured and found to be noise (2026-08-08)

Each of these is real but too small to justify the code it costs. Measured, not
estimated, so they do not need measuring again:

- **Real-valued modulation filter in the LOWPASS path**
  (`backend_numpy/auditory_model.py:338-351`). In LOWPASS the only modulation centre
  is 0 Hz, so `exp(2j*pi*0/fs)` is exactly `1+0j` and the `lfilter` denominator is
  purely real; running it real instead of complex is bit-identical (verified). But the
  modulation stage runs at 100 Hz, so the arrays are tiny: it saves **0.19 ms on a 5 s
  clip** against a ~4.4 s total, i.e. 0.004%. Not worth the branch.
- **TensorExpr/NNC CPU fuser** force-enabled on the scripted adaptation loop
  (`_jit_override_can_fuse_on_cpu(True)` + `_jit_set_texpr_fuser_enabled(True)`):
  **1.02x**. Not a substitute for the Numba kernel.
- **torch thread count**: 2/4/8 threads at 5 s measured 5.31/4.31/4.29 s — inside
  run-to-run noise. The default is fine; do not add a tuning knob.
- **Feeding the fused NumPy auditory kernel a contiguous array** instead of the
  `np.real()` stride-16 view: **1.03x**, and the copy itself costs 0.011 s. The kernel
  is latency-bound on its 5 serial divisions, not memory-bound.

### +0.257% level offset against the MATLAB gold WAVs

Root-caused and deliberately **not** fixed. Our decomposition output is a flat factor
1.0025651 larger in amplitude than `tests/resources/matlab_reference/` on all four
components, because `scipy.signal.firwin` defaults to `scale=True` (unit DC gain) while
MATLAB's `resample` filter is not DC-normalized (raw DC gain 0.9993253); four resamples per
path give 0.999394194^2 * 0.999325320^2 = 0.997441484, whose reciprocal is 1.00256514. It
is frequency-flat because the tap set is scale-invariant in `pqmax`. A line-by-line MATLAB
transcription confirms flipping only that normalization takes the ratio to 1.0000001, with
the residual being PCM16 quantization noise. The gold WAVs are also stale — byte-identical
to the v2.0 shipped outputs, never regenerated for 2.0.1.

The offset is now locked rather than tolerated: `test_matlab_regression.py` asserts the
measured ratio *equals* `_MATLAB_RESAMPLER_GAIN_OFFSET = 1.0025651` to within 1e-3, so a new
gain regression in either direction breaks the test. Full write-up in README under "Known
deviations from the MATLAB reference".

---

## Resolved

### Torch adaptation recurrence ~2.2-2.3x via a Numba kernel (2026-08-08)

1.97s -> 0.85s for 1s audio, 9.25s -> 4.24s for 5s, with all eight reported scores
comparing equal under `==`. The recurrence was already collapsed to ~7 dispatches per
timestep (see below), which made it *dispatch*-bound: the per-step tensors hold one
element per band, so wall clock tracked kernel launches, not arithmetic — 61.9% of
torch runtime at 5 s. Running it as a Numba kernel removes the dispatch entirely
(43-280x on the kernel in isolation, depending on shape).

Bit-identical rather than merely close, because the kernel is an operation-for-operation
transcription: running product over the previous step's state, divide, then
`c*(1-g) + s*g` as two multiplies and an add, under `fastmath=False` so LLVM cannot
contract the multiply-add into an FMA. The in-place state update is equivalent to
torch's cumprod-then-update because each stage is folded into the running product
*before* that stage is overwritten. `torch.equal` holds at every shape measured.

Gated on CPU + `float64` + no-grad + Numba present; everything else keeps the
TorchScript loop. The differentiable path was not touched, so training is unaffected —
and correspondingly this does nothing for the backprop cost still listed in `TODO.md`.

### NumPy decomposition ~1.3x, bit-identical (2026-08-09)

Two changes to `backend_numpy/decomposition.py`, both verified byte-level (`uint8`
view, so signed zeros count) at `max|diff| = 0.0` on all four components across seven
configurations, and end-to-end scores identical to HEAD:

- **Batched per-band Gram/RHS build** (`:241-432`). Every frame in a band shares
  `window_length` and `filter_length`, so the band's Toeplitz stack is one strided
  gather and Gram/RHS become two batched `matmul` calls; only the small Cholesky solves
  stay in a Python loop, capped at `LEAST_SQUARES_FRAME_BATCH = 256` so the footprint is
  independent of clip length. The win is *not* the matmul — measured per frame, ~64% of
  the cost was per-frame Python/NumPy call overhead, the `as_strided`/`hstack` Toeplitz
  build alone being ~24%.
- **Analysis modulation matrix cached and shared with synthesis** (`:580-671`). The
  analysis matrix was rebuilt on every call (347 ms, 11.5% of the decomposition) while
  synthesis was already cached. They are exact conjugates, so synthesis now conjugates
  the analysis matrix and calls `np.exp` only on its overhang.

`TODO.md` had carried the batching as a hypothesis estimated at 1.05-1.08x and "~1 ULP";
both were wrong in the project's favour — it is ~1.15x and exactly bit-identical.

Three notes for anyone touching this code, since each looks like dead weight and is not:
the silence bypass writes explicit zeros because `toeplitz @ 0` is numerically equal but
can yield `-0.0`; the analysis and synthesis window multiplies stay separate because
folding them into one factor rounds differently; and the synthesis modulation matrix
recomputes column `t = 0` with a direct `np.exp` instead of conjugating, because at
`t = 0` the exponent's sign is multiplied away so both directions give `1+0j` and
conjugation returns `1-0j` — equal in value, different bytes. That last one was missed
on the first pass precisely because the check used `np.array_equal` on a float64 view,
where `-0.0 == +0.0`; the byte-level test now asserts both halves of it.

The single-frame `perform_least_squares_projection` is deliberately kept as the readable
reference and as the oracle the torch backend is diffed against, so the numerical core
now exists in two places; `test_batched_projection_matches_single_frame_bitwise` pins
them together.

### `torch.jit.script` deprecation reduced to one justified site (2026-08-08)

Three functions in `backend_torch/auditory_model.py` were scripted; measured
individually, only one was earning it — `_adaptation_loop_forward` at **2.03x** on the
hot loop (1.31x end-to-end). `_straight_through_max` and `_raw_adaptation_loop` were at
1.07x and 1.05x, i.e. inside noise, and their decorators were dropped. The survivor is
scripted via a call rather than a decorator so the `DeprecationWarning` filter can be
scoped to that one line instead of silencing the category process-wide, and it is
wrapped in `try`/`except` because torch documents `jit.script` as unsupported on Python
3.14+ and this runs at import time — an unguarded failure would take the backend down
rather than merely cost speed. The suite now runs clean under `-W always`.

One finding worth keeping: the deprecated code was the *less* deterministic one.
Scripted `_raw_adaptation_loop` returned warmup-dependent gradients — call #1 bit-identical
to eager, calls #2+ differing by 6.7e-12 (8.3e-16 relative) as TorchScript's profiling
executor specialised and re-associated the autodiff graph. Dropping that decorator
removed a nondeterminism rather than introducing one. Forward scores were never
affected: scoring runs under the no-grad branch.

### Gammatone ERB form (MATLAB parity)

The gammatone filter constructor now uses MATLAB's ERB form `24.7 + fc/9.265`
(`Gfb_Filter_new.m` / `Gfb_set_constants.m`, Hohmann 2002 eq. 17) instead of the
algebraically-equivalent-in-intent `erbBW` form `24.7*(0.00437*fc + 1)`. Numerical impact on
typical clips is negligible.

### Shade-in/shade-out window shape and length (MATLAB parity)

The shade window now matches MATLAB's `hann(2*round(ms/1000*fs + 1), 'periodic')(2:end/2)`
strict-interior ramp instead of a full 0→1 ramp. The shape difference is confined to the
first/last few milliseconds and is negligible on typical clips. The same fix also corrected
the window *length*: MATLAB's `round` breaks ties away from zero while Python/NumPy round
half to even, so `round(ms/1000*fs)` was one sample short wherever the product lands exactly
on `.5` — which includes the common cases 44100 Hz @ 5 ms and @ 25 ms, and 22050 Hz @ 10 ms.
Length is now `floor(x + 0.5)`.

### Torch backend ~5.7x faster end-to-end (2026-07-30)

10.8s -> 1.9s for 1s audio, 54.2s -> 9.7s for 5s; scores unchanged to 6 decimals. The
adaptation-loop "cannot be exactly parallelized" note was true about the *time* axis but
missed that the 5-stage *cascade* collapses exactly: every division at step t reads step t-1
state, so it is one `cumprod` + one divide rather than five interleaved pairs (~75 dispatches
per timestep -> ~7). Also: FFT convolutions were running at arbitrary (often near-prime)
lengths, and the resampler re-transformed its FIR filter on all ~200 calls per decomposition.

### Torch decomposition tolerance was not a torch gap (2026-07)

With a torch runtime available locally (`KMP_DUPLICATE_LIB_OK=TRUE`): the
`test_torch_decomposition` relaxed tolerance is the shared Gammatone reconstruction floor
(numpy hits the same ~1.9e-3), not a torch gap — comment corrected and real numpy-vs-torch
parity assertions added. The redundant `test_linalg_solve_fallback_parity` (it tested library
internals, not our code) was removed.
