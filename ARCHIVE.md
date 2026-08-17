# archive

Settled work: investigations that are root-caused, fixes that have landed, and options
deliberately declined. Nothing here is open — it lives outside `TODO.md` so the handoff
list stays short, and it is kept rather than deleted because the reasoning is what stops
the same ground being re-covered. Entries are append-only; if one is genuinely reopened,
move it back to `TODO.md`.

---

## Declined

### Numba `parallel=True`/`prange` on the NumPy kernels (2026-08-08)

Profiled and prototyped: annotating the five Numba kernels then present (the
`@numba.njit` kernels in `backend_numpy/gammatone.py`, plus
`_numba_fused_auditory_kernel` in `backend_numpy/auditory_model.py`) with
`parallel=True` and parallelising over bands — or over flattened `row x out_len`
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

Amended 2026-08-10: there are six kernels in `backend_numpy/gammatone.py` now, not
four — the two polyphase kernels were each split into a real and a complex twin (see
"Decomposition: torch ~2.2x, numpy ~1.47x"). Nothing about the decision changes; the
parallel axis is still over independent work items, and the constraint is still a
design one. But the measured 1.63-1.87x was against the *old* scalar kernels, and
those are now 1.9-2.2x faster single-threaded, so the headroom `prange` would buy is
smaller than the number above suggests. Re-measure before arguing from it.

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

### Five decomposition items settled by measurement, all declined (2026-08-15)

From the second 2026-08-15 pass, the one that worked the **decomposition** path. Four of
these were queued in `TODO.md`; one of them was that file's own "cheapest decisive next
step". None of them survived contact with a measurement.

- **Routing torch's no-grad CPU resampler at the numpy Numba kernels — DEAD, 0.461x.**
  This was the most-anticipated item on the list, precedented on the adaptation
  recurrence (2026-08-08, ~2.2-2.3x by exactly this trick), and `TODO.md` carried it for
  three passes as a cheap measurement nobody had run. It has now been run, with
  `benchmarks/ab.py` at the real filterbank rates. All 10 configurations `AGREE` under
  swapping:

  | route | mean ratio (>1 would mean Numba faster) | share of MACs |
  | --- | --- | --- |
  | interpolate | **0.348** | 56.4% |
  | decimate | **0.801** | 42.3% |
  | mixed (scipy) | 0.768 | 1.25% |

  Weighted, routing torch at the Numba kernels projects to **0.461x — about 2.2x
  slower**. Numerical agreement was verified *first* (worst 1.255e-15 relative), so this
  is a pure speed verdict and not a correctness one. torch's polyphase GEMM is simply the
  better kernel at these shapes; the one discouraging data point already on file (the
  `ab.py` demo showing the Numba path at 0.57x against SciPy `upfirdn` at `up=2`) turned
  out to be representative rather than an artefact of an unused rate. The adaptation-loop
  precedent does not transfer, because that win was about removing *dispatch* from a
  per-timestep recurrence, and the resampler has no dispatch problem.

  Incidentally confirmed while tracing this: **zero calls reach `_fft_resample`** in the
  real workload.

- **Deleting `_fft_resample` — declined; the decision is KEEP AND DOCUMENT.** The premise
  for deleting it was that nothing reaches it, and that premise is now independently
  confirmed (above). It stays anyway, for two reasons that outweigh ~90 lines and one
  `lru_cache`: it is the mixed-rate tests' only *independent* oracle, which is the
  strongest check in that file, and `next_fast_fft_length` has to stay regardless because
  `auditory_model.py` uses it. This was always a "decide and write down why" item rather
  than a perf win; this is the decision.

- **`.flip(-1)` → `.index_select(-1, ...)` in the least-squares assembly — attempted,
  REFUTED, dropped.** It was claimed bit-identical. It is not. `src_unfold` has a
  **second consumer** — `T_s_all = src_unfold.permute(...)` feeding `torch.matmul` — and
  the two routes hand it different strides (`(2926, 22, 1, 2)` under `flip` against
  `(2926, 22, 11, 1)` under `index_select`). `matmul` is not stride-invariant there, and
  `proj_all` comes out different. The lesson is the reusable part: a change that is
  bit-identical *at the site you changed* can still move the output through an aliasing
  consumer you did not look at. Check every consumer of a view before calling a view
  substitution exact. Note this also removes the dependency the queued LS-assembly 2(c)
  item was written against — see `TODO.md`.

- **LS assembly 2(a), flipping the solved weights instead of the Toeplitz block —
  refuted.** The idea was that flipping the far smaller solved weights is a pure
  permutation and therefore free and exact. It is not exact, because the flip does not
  commute with the factorization: `chol(J G J) ≠ J chol(G) J`. Measured 4.475e-16 on the
  weights, and 4.25e-15 / 2.64 on the projection.

- **LS assembly 2(b), a batched Hermitian rank-k Gram — declined.** The Gram is Hermitian
  PSD built with a full batched GEMM, so it does about 2x the necessary flops, and a
  `herk` would halve them. There is no batched Hermitian rank-k in torch or in BLAS, so
  taking it means a per-frame Python loop over up to 251 frames x 32 bands — reinstating
  exactly the overhead the 2026-08-09 batching removed. Reasoned, not prototyped, and
  judged a near-certain loss. This is the same conclusion the 2026-08-12 rejected list
  reached for the identical opportunity in the numpy backend.

### Auditory-path optimisations that measured worse, and four method traps (2026-08-15)

From the same pass as the wins in "The 2026-08-15 auditory-path pass". The two rejections
were built and reverted; the four method findings each invalidated somebody's first
answer, which is why they are here rather than in a commit message.

- **Interleaving N independent bands through the numpy fused auditory kernel's sample
  loop.** The reasoning is sound and is the archive's own: the 5-stage cascade is five
  *dependent* divisions at ~14 cycles of latency each, and the 2026-08-08 note below
  already says this kernel is "latency-bound on its 5 serial divisions". Processing a
  block of independent bands in lockstep should overlap those chains. An isolated harness
  read **1.53x**. In situ it is worth nothing, at every block width:

  | block width | vs width 1 (`ab.py`, 20 samples/phase, AGREE) |
  | --- | --- |
  | 2 | 0.9921x CI [0.9754, 1.0047] |
  | 4 | 0.9753x CI [0.9527, 0.9871] |
  | 8 | 1.0042x CI [0.9794, 1.0177] |
  | 16 | 0.9818x CI [0.9498, 1.0008] |

  Every width is bit-identical to every other (reordering independent recurrences
  reassociates nothing), so this was purely a timing decision and the timing says no.
  Built, measured, reverted; there is a comment at the site in
  `backend_numpy/auditory_model.py` so it is not re-prototyped. **The torch twin was
  measured independently the same day and also came out flat-to-slower.**

- **Sharing the linear front end across the 5 auditory runs per channel.** The metric path
  runs the auditory model five times per channel, and the four references are
  leave-one-out sums of the same four components. The front end — scale, resample,
  gammatone — is linear, so four filterbank passes could in principle serve all five.
  Measured, and it is a loss: it saves one gammatone pass (~44 ms) and costs forming the
  combinations on `(27, N)` subbands instead of on `(N,)` waveforms, at ~11 ms per
  full-array op. **Summing in the time domain, before the filterbank, is the correct
  place.** It is also only a reassociation, not bit-identical, so it would have to clear
  the near-exact bar to buy a negative. Recorded in place at the call site in
  `backend_numpy/metrics.py`, which is also where a `ThreadPoolExecutor` suggestion that
  directly violated ground rule 1 used to live.

Four method findings, each of which broke a first attempt:

- **`np.longdouble` is float64 under MSVC on this platform.** `np.finfo(np.longdouble)`
  reports `eps == 2.220446049250313e-16` and `machep == -52`. An "80-bit oracle" built on
  it therefore silently compares float64 against itself and will report exactly **0.0**
  deviation for any candidate whatsoever — including a badly wrong one. One agent's first
  accuracy claim in this pass was built on exactly that and was wrong. Use a double-double
  instead (Dekker split, `two_prod`, `two_sum`; ~106 bits); two independent agents
  reproduced the same construction from scratch, and it is what the haircell accuracy
  numbers above rest on.

- **torch's batched FFT is not bit-row-invariant, and never was.** Measured on a
  `(9, 20000)` float64 input at fs=24000, one call against nine single-row calls:
  `process` 7.11e-16, `process_real` 5.55e-16. Through the public auditory entry with one
  row per chunk forced, the internal representation moves 4.55e-12 on the real path and
  3.64e-12 on the complex one. `_FFT_CHUNK_BUDGET_BYTES` claimed "rows are filtered
  independently, so any value gives identical output"; that was false **before** this
  session's patches as well as after, so it is a property of torch's batched FFT and not
  of anything landed here. The comment, the `process`/`process_real` docstrings and README
  are now correct. The consequence for anyone touching that constant: it is a memory knob,
  not a tuning knob, and changing it changes the last digits of a score.

- **A complex `torch.exp` is not length-invariant either, and CI caught it where a
  32-case sweep did not.** `_get_gammatone_H_real_torch` builds `z_inv` on the half grid
  while `_get_gammatone_H_torch` builds it on the full one. The half filter is therefore
  *not* bit-identical to `(H[k] + conj(H[-k]))/2` sliced from the full one, even though it
  is exactly that on Windows across 32 config x size combinations — which is how it was
  asserted with `torch.equal` when it landed. ubuntu-latest failed it at `N_fft=4096` on
  the first CI run: different grid length, different libm vectorization, last-ULP
  difference, then a 4th power with a resonant denominator. The test now bounds at 1e-13
  of peak, which is still ~200x tighter than the quietest thing it exists to catch (a
  ~2e-11 Nyquist-bin error). Same family as the batched-FFT entry above.

  Worth stating as a rule, since this is the third entry in this file with the same shape:
  **a sweep proves a property on the machine you swept.** Exactness across many shapes on
  one platform is evidence about that platform's libm, not about the identity. If a test
  asserts bit-equality between two *differently-shaped* computations of the same quantity,
  it will eventually fail on another platform — bound it, and size the bound from the
  error the test is meant to detect.

- **The "old path's own noise" control — treat this as the standard method for any
  near-exact claim.** Before attributing a deviation to a candidate, reproduce it on the
  **unchanged** path via a mathematically exact identity: permute the rows, change a chunk
  boundary, run full-batch against row-by-row. In this pass a seed-1006 row permutation on
  the *unpatched* path reproduced the rfft candidate's OPS deviation of **2.117e-09 to the
  digit**. Without that control, the number you report as "what my change moved" may be
  characterizing the existing filterbank instead. The 2026-08-12 mixed-rate entry used the
  same trick with a different valid padding length and reached the same kind of
  conclusion; this generalizes it.

- **Isolated harnesses overstate on this codebase — now three data points, and the rule is
  general.** Band-interleaving read 1.53x isolated against 1.0042x in situ; its torch twin
  behaved the same way; and the gammatone loop order read 1.16-1.53x in a kernel harness
  against 1.024x in situ on the decomposition. A kernel harness feeds warm cache, omits
  the allocator, and — most of all — omits Amdahl. **No timing claim in this repo is valid
  unless it came from `benchmarks/ab.py` driving a real public entry point.** Note the
  converse is also on record: the same rule caught the gammatone loop order being
  *under*-sized by 2.2x once measured on the entry point that actually calls it 18 times.

### GNU Octave as a reference generator — does not work, and why (2026-08-14)

**The idea.** We have the complete MATLAB PEASS v2.0.1 source in `.scratch/`. If Octave
could execute it faithfully, it would be a *reference generator*: gold WAVs for any input
we like, instead of the single clip in `tests/resources/matlab_reference/`. That would
have made the whole clean-reference question mostly moot.

**It does not work.** Octave, running the genuine MATLAB source with both
decomposition-path MEX files compiled, reproduces the gold WAVs only to:

| component | correlation | peak dev | RMS ratio |
| --- | --- | --- | --- |
| true_target | 0.99937 | 3.8% | +1.65% |
| target_distortion | 0.99896 | 5.6% | +2.39% |
| interference | 0.99937 | 4.1% | +1.42% |
| artifacts | 0.99668 | 6.2% | +1.36% |

Our Python reaches correlation 0.999996 and gain error ~1e-5 on the same clip — roughly
three orders of magnitude closer. **The Python port is a far better MATLAB reproduction
than actual Octave running the actual original code.** That is worth stating plainly,
because it is the opposite of what you would assume.

**Cause, isolated.** `resample`. Octave's signal-package version differs from ours by
9.05e-2 relative on a single 3/2 call (rms ratio 1.0116) and 6.83e-2 on 2/3 (1.0129),
while ours is *bit-identical* to `scipy.signal.resample_poly`. MATLAB designs the
anti-aliasing filter with `firls`; Octave uses a kaiser-windowed ideal sinc. The
magnitudes reconcile: 1.0116 compounded over the four resamples per signal path, with the
up and down conversions partially cancelling, lands on the observed 1.014-1.024 — so the
resampler accounts for essentially all of the deviation.

**The obvious fix makes it worse.** Reimplementing `resample.m` from MathWorks' documented
algorithm (order `2*n*max(p,q)` with `n=10`, cutoff `pi/max(p,q)`, `firls` windowed by
`kaiser(beta=5)`, normalized `b = p*b/sum(b)`, then `upfirdn`) and shadowing the signal
package with it measured **+3.1% RMS against gold, versus stock Octave's +1.2%**. At the
filter-tap level against our scipy `firwin` design:

| design | max tap difference |
| --- | --- |
| Octave `firls` (MATLAB's documented method) | 2.90e-2 |
| Octave `fir1` (window method, same family as ours) | 2.93e-3 |

Even Octave's `fir1` — nominally the same windowed-Kaiser algorithm scipy's `firwin`
implements — differs from ours by ~0.9% on the taps. **So the problem is not that Octave
picked the wrong filter design; it is that Octave's DSP primitives and scipy's do not
agree at the precision this project cares about.** Closing that would mean reimplementing
the filter design in `.m` to mirror scipy, at which point the resampler leg is circular,
the result is still hostage to Octave's other primitives, and it is more work than the
Python transcription it was meant to avoid.

**The consequence for the reference design, which is the durable part.** There is no route
to an *independent* resampler except real MATLAB. A Python transcription needs a
`resample` too, and we would write the same reverse-engineered one. So transcription buys
no resampler independence either — which means the honest scope of any clean reference is
"independent everywhere except the resampler", and the cheapest way to get that is a
Python port using stock `scipy.signal.resample_poly`. That is why `reference/` carries a
declared `# !!! DEVIATION` at its resample call rather than attempting a transcription.

**Practical notes, so a future attempt does not rediscover them.** Octave ships a plain
`.zip` for Windows (`octave-9.4.0-w64.zip`, ~765 MB) alongside the installer, so it needs
no admin rights and no GUI. `mkoctfile --mex` builds the PEASS MEX files without trouble
once `mingw64/bin` is on `PATH`. Three findings about the MATLAB source itself are worth
keeping:

- `haircell` and `adapt` are **metrics-only** (`pemo_internal.m`); the decomposition never
  touches them.
- `toeplitzC` and `Gfb_Analyzer_fprocess` are in the decomposition path but both have pure
  MATLAB fallbacks — a `try/catch` onto built-in `toeplitz`, and the `analyzer.fast` else
  branch respectively. **No compilation is needed to run the decomposition.**
- `myPemoAnalysisFilterBank.m:44` sets `analyzer.fast = true` *unconditionally*, without
  the `exist(...)` guard its counterpart at `pemo_internal.m:65` uses. In MATLAB you would
  have run `compile.m` first so it never shows; on a bare interpreter it is a hard error.

### Two CI failures from over-claimed test invariants (2026-08-09)

Both were defects in the *tests*, not the code, and both came from writing down a
measurement taken on one machine as though it were a guarantee. Recorded because the
failure mode is easy to repeat and the second one is genuinely subtle.

**Bit-equality across toolchains.** The Numba adaptation kernel is exactly
bit-identical to the torch loop on the reference platform, and the test asserted
`torch.equal`. CI (Linux, CPython 3.14) failed at 1.8e-14: whether a toolchain
contracts `a*b + c` into a single FMA varies by LLVM and torch build. The local kernel
compiles to separate `vmulsd`/`vaddsd`; a newer LLVM need not. The fix is a tolerance
chosen by measurement rather than taste — perturbations fall into two cleanly separated
bands, roundoff (FMA contraction 2.6e-16 relative, algebraically-equal EMA
reassociation 1.5e-14) and real transcription errors (updated state in the running
product 4.4e+00, running product reset per stage 1.0e+00). Fourteen orders apart.

**Relative tolerance across a cancellation.** The follow-up fix was still wrong for the
*public entry point*, which failed at 7.9e-11 relative while the raw kernel passed.
Not a second divergence — the same roundoff, amplified.
`simulate_auditory_nerve_adaptation` ends with
`100/(1 - final_thresh) * (adapted - final_thresh)` where `final_thresh` is
0.6978305849: a 330.94x scale-up wrapped around a subtraction. Outputs span four orders
(median 161, minimum 0.024), and for the ~0.3% of samples near `final_thresh` the
cancellation shrinks the denominator until relative error is meaningless. The worst
relative point had an output value of 0.036 and back-solved to an absolute error of
8.7e-15 in `adapted` — the same 1-ULP roundoff the kernel test already tolerated.

The lesson for this codebase specifically: **relative tolerance is the wrong instrument
downstream of a subtraction of comparable quantities**, which this pipeline does in at
least two places (here, and the `artifacts` component the README already documents).
Bound those on absolute error at the output scale instead.

### Batching the torch analysis and synthesis passes — obsoleted by the GEMM (2026-08-10)

`TODO.md` carried this as **P3, 1.09x**: sources and estimate go through two independent
passes over an identical filterbank, and the four distortion components go through four
independent synthesis passes, so `cat` them into one wider batch each. Implemented both,
measured all four on/off combinations twice on a quiet machine:

| analysis merged | synthesis merged | mono | stereo |
| --- | --- | --- | --- |
| no | no | 1.211 / 1.348 s | 3.553 / 3.461 s |
| no | yes | 1.174 / 1.234 s | 3.466 / 3.695 s |
| yes | no | 1.231 / 1.298 s | 3.634 / 3.848 s |
| yes | yes | 1.205 / 1.201 s | 3.708 / 3.787 s |

Within noise on mono and mildly *negative* on stereo, so both were reverted. The
original 1.09x was real when it was measured — but it was measured against the FFT
resampler, whose whole problem was having no batch dimension to parallelise over. The
polyphase GEMM removed that, and with it the reason to widen the batch. Merging also
raises the peak frequency-domain footprint, which is the likely source of the stereo
regression.

Two lessons worth keeping: a prototyped speedup is only valid against the code it was
prototyped on, and a first attempt at this was measured while a dozen unrelated agents
were saturating the machine, producing a 5.5-11.1 s spread on a 3.5 s workload and two
mutually contradictory conclusions. Check the machine is idle before A/B-ing anything
at this scale.

The related **memory** finding did land: `GammatoneAnalyzerTorch.process` now chunks its
batch (`_FFT_CHUNK_BUDGET_BYTES`) and copies out the valid region instead of returning a
slice of the `N_fft`-wide buffer. Measured speed-neutral at 256 MB (mono 1.313 vs
1.306 s, stereo 3.724 vs 3.739 s) and kept purely so the two
`(rows, num_bands, N_fft)` intermediates stop scaling with batch width — 8 rows would
otherwise allocate ~1 GB twice to produce a few hundred MB of subbands. Tighter budgets
do start to cost (134 MB: 1.342 / 3.868 s).

### Decomposition optimisations that measured worse, or not at all (2026-08-12)

A second sweep of the same kind, from the profiles taken while landing the 2026-08-12
pass. Same principle as the 2026-08-09 list below — these all look right on paper.

- **Polyphase GEMM for the *numpy* interpolator**, mirroring what won 2.2x in torch.
  Interpolation with `down == 1` is exactly one dgemm on a strided `(in_len, 21)` view.
  Prototyped across the 20 real `(up, in_len)` pairs: **0.30x, i.e. 3.3x slower**, at
  2e-16 agreement. The torch GEMM won against an *FFT convolution*; numpy's baseline is
  already a vectorized AXPY kernel. This is the most tempting-looking idea in the file
  and it is dead — do not port torch wins to numpy without re-measuring the baseline.
- **`np.zeros` -> `np.empty` + zero-only-the-tail for the synthesis buffer.** 64 MB per
  call, and after the scatter change only ~4% of each row still needs zeroing. Measured
  **21.4 -> 23.2 ms, marginally worse**. `np.zeros` is calloc: it gets lazily-zeroed
  pages from the OS and the cost is the first touch, which the scatter pays either way.
  Note this is the *opposite* result to the same trick inside the resampler padding
  buffers, which took decimate from 1.27x to 1.90x (see the 2026-08-10 entry) — the
  difference is that those buffers are immediately overwritten in full.
- **Removing the per-band temporary in `_numba_delay_process`.** It allocates a
  `(delay + num_samples)` buffer per band, fills it, copies `num_samples` back out —
  ~128 MB of avoidable traffic per mono decomposition. Rewritten to shift against the
  output row directly: bitwise identical on both output and state, and **0.99x**. The
  kernel is bound on streaming the complex input, not on the temporary.
- **Skipping the zero blocks in `toeplitz_stack @ block_diagonal_weights`.** The weights
  are block-diagonal, so the GEMM does `num_sources`x the necessary flops. Per-source
  GEMMs measured **0.86x** (3.26 -> 3.79 ms per 256-frame batch) and were not
  bit-identical (7.4e-15 — BLAS picks different kernels). The per-frame matrices are
  tiny, so flops are not the cost.
- **Hermitian `herk` Gram, and fusing the conjugate transpose via `zgemm trans_a=2`.**
  Would halve flops and kill a 12 MB `.conj()` temporary (6.85 ms per batch, 2.3%), but
  neither is batched in BLAS, so both force a Python loop over 256 frames — reinstating
  exactly the overhead the 2026-08-09 batching removed. Reasoned, not prototyped, and
  judged a near-certain loss.
- **Batching the four identical `2/3` synthesis downsamples into one 4-row call.**
  23.94 -> 21.90 ms, **1.09x**, about 2 ms of a 1.2 s decomposition. This is the narrow
  remaining slice of the archived P3 batching experiment and it confirms that entry's
  conclusion at finer granularity. Likewise batching the 4 components into 4-row blocks
  in the band loop: 128 calls collapse to 32, but the Python dispatch is only ~48 us per
  call, so ~5 ms total, and the padding volume is unchanged.

- **The torch gammatone forward transform is not where the time is.** In
  `gammatone.py:176` the forward `fft` is 0.009 s against the 32-band inverse at
  0.131 s. An `rfft` forward would halve ~7% of the transform work and is not worth the
  Hermitian reconstruction; everything else there routes back to transform sizing, which
  the dropped sizing item already closed. The `_FFT_CHUNK_BUDGET_BYTES` chunking probed
  as correctly sized.

  **Amended 2026-08-15 — do not read this as a blanket "no rfft in the gammatone"; a
  sweep that does will re-reject a win that has landed.** This bullet is about the
  *forward* transform on the **decomposition** path, where the output must stay analytic
  so the inverse is complex-to-complex and the Hermitian reconstruction has to be paid
  for. The **auditory** path discards the output to `.real`, which makes the inverse
  complex-to-**real** — so it attacks the 0.131 s side this bullet itself identifies as
  the cost, not the 0.009 s side, and the Hermitian symmetry is folded once into an
  `lru_cache`d half filter rather than reconstructed per call. Measured **1.1306x** on
  the metric path; see `GammatoneAnalyzerTorch.process_real` and "The 2026-08-15
  auditory-path pass" under Resolved. The bullet as written remains correct for
  `process`, which is untouched.
- **Collapsing the synthesis band-group unpack loop** (`decomposition.py:460` and the
  32-iteration write loop at `:465-478`). Within a decimation group every band
  necessarily has the same length — `torch.stack` would fail otherwise — so the loop
  collapses to one advanced-index assignment per group. Bit-identical and trivial, but
  it moves the same bytes and `torch.stack` did not reach the top 22 of the profile.
  A readability item that might incidentally be free, not a perf item.

Two observations from the same sweep that are not opportunities but bear on sizing:

- **The numpy resampler kernels are at their achieved ceiling.** Interpolation 337.7 ms,
  decimation 237.6 ms, mixed 36.8 ms — 612 ms, 42% of the numpy decomposition, still the
  largest block. That is ~1.9-2.0 real GMAC/s against the ~2.3 GMAC/s this archive
  already documents for these kernels. Further gains need *fewer MACs*, not better SIMD,
  and the tap count is pinned at 21 by MATLAB parity (`half_length_factor = 10`). The
  only route to fewer MACs is folding the modulation into the taps — TODO P5, which is
  flagged as probably fatal for the same real-FIR reason in numpy as in torch.

  **Amended 2026-08-18: the "probably fatal" flag was refuted in torch, and that is not an
  endorsement of the numpy twin.** P5 landed at 1.1673x mono by making the folded kernel a
  true complex `zgemm`; the widened real `dgemm` that would have preserved the real-FIR split
  measured 0.717x, i.e. the caveat's mechanism is real and the way past it is a genuine
  complex GEMM. numpy has no complex GEMM here to win with — its kernels are `ddot`/AXPY on
  real planes, and this file's own "a numpy polyphase GEMM measured 3.3x slower" result is the
  direct precedent. Do not port P5 to numpy without re-measuring the baseline first.
- **The "VECTORIZED 2D BLOCK" comments oversell what happens.** All 32 bands have
  distinct decimation factors, so every "block" is literally one row — `shape=(1, 8572)`
  on every synthesis resample. The comments describe an intent, not a realized benefit.

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
- **Removing the redundant fancy-index copy** at `backend_numpy/decomposition.py:571`
  (`block = subbands_output[band_indices, :]` in the analysis band loop).
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

  **Amended 2026-08-15: this result is still correct, and it is a *component* of a win
  that landed.** What was rejected here is buying contiguity at the consumer, where the
  copy costs more than the 1.03x it buys. The producer can supply it for free — the
  gammatone kernel was writing that memory anyway, so storing float64 rather than
  complex128 hands the same contiguous array over at negative cost. See "The 2026-08-15
  auditory-path pass" under Resolved. Generalizable: a micro-optimisation rejected at the
  call site that has to pay for it may still be free one stage upstream.

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

### P5 — the modulation folded into the polyphase filters: 1.167x mono, −117.9 MB (2026-08-18)

The last survivor of the original P-numbered list, and the largest known remaining
decomposition win. **Both halves are implemented and both gates have been run and passed.**
`_get_analysis_mod_matrix_torch` (61.44 MB, `(32, 120000)` complex128) and
`_get_synthesis_alignment_matrix_torch` (61.61 MB, `(32, 120336)`) are **gone**, replaced by
`fast_demodulate_decimate_torch` and `fast_interpolate_modulate_torch` in
`backend_torch/utils.py`: a true complex `zgemm` on both sides, range-reduced phase on both,
folded on **both** the grad and no-grad decimation routes (interpolation has only one route).
`decomposition.py`'s two `torch.unique` group loops are per-band loops now, and the synthesis
half takes `GammatoneSynthesizerTorch.process`'s `alignment=None` default, so the surviving
full-rate factor is a per-band complex scalar.

Complex-exponential modulation distributes through convolution exactly, so the full-length
modulation multiplies fold into the 21-tap filters. What survives the fold is a residual at
the *decimated* rate — `m[(hf+i)·D]` on the analysis output, `m[U·i]` on the synthesis input
— applied as its own elementwise pass, deliberately: summed over the 32 bands those are
~0.35% of one full-rate pass, and fusing a factor into a shifted-diagonal *sum* is not
expressible anyway.

**Accuracy — parity, per band and end to end.** Every reference is built from an
**independent** exact-`Fraction` modulation that shares no code with the library:

| check | bar | measured |
| --- | --- | --- |
| analysis, per-band, 4 batch geometries | 5e-15 | **1.219e-15** of peak |
| synthesis, per-band, mono + stereo | 5e-15 | **2.224e-15** of peak |
| synthesis, whole-function, mono/stereo/unbatched | 1e-13 | **6.237e-16** of peak |
| synthesis gradients, all 32 subband tensors | 1e-13 | **1.207e-15** of peak |
| end-to-end, four components, folded vs pre-change | 1e-13 | **6.394e-16** of peak |

The folds themselves therefore move nothing above ~1e-15. **P5's whole accuracy movement
against unmodified HEAD is the range reduction**, which is a separate, strictly-improving
change: measured end to end on the stereo 3-source reference, relative to each component's
own peak, true_target 2.296e-12 / target_distortion 2.348e-12 / interference 2.607e-12 /
artifacts 3.700e-12, toward truth.

#### 1. Ground rule 2 — passed, and torch has converged onto numpy

`measure.py p5` + `compare.py baseline p5` against the frozen `e960c5e` capture, verified
genuine before use (`git_dirty = false`). Idle machine, run by the lead.

- **numpy: exactly zero movement** on every correlation, gain error, score and waveform.
- **All 8 torch correlations improved.** Worst correlation drop `+0.000e+00`.
- Worst |gain error| growth `+6.814e-07` (torch/`target_distortion` ch0) — and, exactly as on
  2026-08-17, it is **convergence onto numpy**, not regression: torch moved `-6.03e-07` →
  `-1.28e-06`, and numpy's value *is* `-1.28e-06`. Same headline number, same explanation. A
  tool that only knows "moved away from the reference" cannot tell the two apart.
- **`overall_perceptual_score` is now `17.665620658898` on BOTH backends — identical to all
  twelve digits** — and several torch gain errors now equal numpy's exactly. Largest torch
  score move `-3.643e-04` (`interference_perceptual_score`) against a 1.0 bar.
- Worst waveform deviation **6.144e-05** of peak (`matlabref_torch__artifacts`) — *unchanged*
  from the pre-P5 series figure. P5 added nothing measurable to it.

#### 2. Ground rule 3 — passed, and the two directions AGREE

`benchmarks/ab.py`, driver a full torch decomposition of `tests/resources/database/exp01_*`
through `benchmarks/measure.py`'s own `decompose()`, idle machine, with the old code loaded
from a `git worktree` at HEAD **alongside** the new in ONE process. **All six A/Bs returned
AGREE on the first attempt; none was retried.**

| workload | ratio (>1 = P5 faster) | 95% CI | phases |
| --- | --- | --- | --- |
| mono, 96 calls/phase | **1.1673x** | [1.1631, 1.1751] | 1.1612 / 1.1759 |
| stereo, 64 calls/phase | **1.1571x** | [1.1486, 1.1643] | 1.1561 / 1.1579 |

Attribution, mono 96 calls/phase, measured in **both** directions as the 2026-08-15 lesson
demands, all AGREE:

| half | ON (from all-old) | OFF (from current tree) | ON saving | OFF saving |
| --- | --- | --- | --- | --- |
| analysis fold | **1.1047x** | **1.1062x** | 102.2 ms | 107.5 ms |
| synthesis fold | **1.0514x** | **1.0735x** | 57.7 ms | 64.9 ms |
| product / sum | 1.1615x | 1.1875x | 159.9 ms | 172.4 ms |

**This is the notable methodological result, and it is the OPPOSITE of 2026-08-15.** There,
leave-one-out read three real items as null because two of them overlapped on the same copy.
Here the halves are genuinely independent — the analysis fold reads the same to ~5 ms in both
directions, the synthesis fold's two ratios differ only because they are taken against
denominators ~200 ms apart, and the absolute savings sum to 160-172 ms against 161 ms
measured together. The ON-direction product 1.1615x reproduces the 1.1673x headline to 0.5%.
Analysis is worth ~1.7x synthesis; neither half is a dud and neither can be dropped because
the other one "already got it".

**The 2026-08-15 lesson is vindicated by a case where the two directions AGREE, not only by
the case where they disagreed.** Measuring both directions is what made that checkable at
all; had only one been taken, this result would have been indistinguishable from a lucky
guess.

**Ground rule 3's "harnesses overstate" warning is CONFIRMED, less brutally than the three
recorded 1.5x+ cases.** The kernel harness predicted 1.351x analysis / 1.426x synthesis *in
isolation*; in situ the halves are 1.105x and 1.05-1.07x **of the whole decomposition**. The
warning stands as written — 1.167x is well short of 1.35x/1.43x — but the in-situ result is
nowhere near noise, and P5 does not have to fall back on its memory argument.

**A new method fact worth having: absolute medians drift 11.4% BETWEEN processes.** The
identical `new` code read 995 ms in one run and 882 ms minutes later on the same idle
machine. Only *within-run* ratios are quotable; nobody may cross-quote the millisecond
columns between two `ab.py` runs. This independently re-confirms ground rule 3's premise.

#### 3. Memory — a MEMORY result, not an `ab.py` one

Not governed by ground rule 3. Method: one clean process per state, one full decomposition,
`gc.collect()` then a `gc.get_objects()` sweep for live complex tensors.

Resident complex tensors fall **181.352 MiB → 68.926 MiB, −112.4 MiB**, against a predicted
−117.9 MB ledger. **Both matrices are measurably absent, not merely unreferenced** — the
sweep walks `gc.get_objects()`, so a live cache entry would show. Only the `(32, 131072)`
gammatone `H` remains above 8 MiB. What replaces them is 128 small tensors — 64 modulated
`(21, rate)` kernels and 64 residual/pre-multiply vectors — totalling 4.93 MiB.

#### 4. Two anti-double-counting warnings — do not add these to anything

- **P4's fusion win and P5's synthesis win are the SAME win re-banked.** The synthesis fold
  deletes the very matrix P4 built (`_get_synthesis_alignment_matrix_torch`). P4's ~1.15-1.18x
  and P5's synthesis 1.05-1.07x must not be added.
- **P5 overlaps the standalone "complex real/imag round trip" item almost completely** (which
  `TODO.md` carried at 4-8%). The folded path skips `_split_real_imag`/`_merge_real_imag`
  entirely on both sides. Do not add that either.

#### 5. The best lesson of the pass — the two matrices were bitwise exact conjugates

The analysis and synthesis modulation matrices were **bitwise exact conjugates**
(`max|a - conj(s)| == 0.0`), so their unreduced-exponential phase errors **cancelled end to
end**. That is why a ~4e-11 phase error was invisible to every existing test — and, far more
importantly, **why range-reducing only ONE half would have made the pipeline worse**: it
breaks the cancellation and leaves the full 4.158e-11 on the output. A locally-correct change
that degrades the whole. That is exactly what the first attempt did, and it was caught and
fixed by reducing both sides; the fold then inherits the property, because the taps on both
sides are built by the same `_reduced_phase_exp`.

Generalise it: **before improving one side of a conjugate/inverse pair, check whether the two
errors were cancelling.** An error that only exists in an intermediate is not an error in the
output, and removing half of it is not half a fix.

#### 6. A live defect that passed a fully green suite, because the reference shared code

During the analysis half, `_modulation_phase` returned `limit_denominator(10**12)` of
`fc/fs`, on the reasoning that capping the *ratio* (rather than `fc` and `fs` separately)
bounds the error at ~1e-24 turns. **It does not: `limit_denominator` is a *best rational
approximation* and can return a denominator far below its limit.** The analyzer's
base-frequency band sits one ULP above 1000 Hz, and `1000.0000000000001/24000` collapsed to
exactly `1/24` — 1.137e-16 relative, 3.572e-12 rad at `n = T`, and a measured **3.4373e-12**
of that band's own peak, i.e. ~10x *worse* than unmodified HEAD on that one band while every
other band improved by four orders. It happens at every supported rate, always on exactly one
band.

**It survived a green 630-test suite and two parity scripts because every reference was built
by calling `_modulation_phase` itself, so the error cancelled on both sides** — precisely the
trap that function's own docstring accused a prototype of. It was caught only by an
*independent* exact-`Fraction` reference, and confirmed against a 50-digit `mpmath` gold
value. The phase is now the **exact** ratio, split inside `_reduced_phase_exp` into a capped
rational plus a float64 remainder.

**Never approximate the ratio, and never build the reference for a phase check out of the
code under test.** The new tests were then mutation-proved: with the fix monkeypatched back
out, **5 of 13 cases fail**.

#### 7. Two hard constraints, both re-confirmed by measurement

1. **A true complex `zgemm`, never a widened real `dgemm`.** The variant keeping the folded
   kernel on the real split path measures **0.717x** — four real GEMMs lose to one complex one
   at these shapes.
2. **Range-reduce the phase mod one turn before the exponential.** Naive folding measures
   3.7e-11 (analysis) / 4.4e-11 (synthesis), because `2*pi*fc*n/fs` reaches **2.3465e5 rad**
   at `fc = 7468.977`, `fs = 24000`, `n = T = 120000`, so `eps*theta = 5.2e-11 rad`. Reduced,
   the same fold measures ~1e-15. (An older figure of "~2.2e8 rad" appeared in three places
   and was wrong by three orders; it predicts ~5e-8, which nobody ever measured.)

Also implemented and worth knowing: the folded path is no longer complex128-only — the
kernel, residual vector and signal are brought to the signal's own complex precision, and the
reduction still runs in float64 and is rounded once at the end. And the index split at `2**21`
deleted a guard whose fallback was a per-sample pure-Python `Fraction` loop, which would have
tripped at ~23.1M upsampled samples (~321 s at 48 kHz) with no warning.

**Considered and not taken**, filed as a micro-item in `TODO.md`: reducing `turns` into
`(-0.5, 0.5]` rather than `[0, 1)` halves the polar argument and measures **5.16e-16 against
9.14e-16** on a 50-digit `mpmath` reference — a real 1.8x, orthogonal to the defect above,
both forms already on the float64 floor.

Evidence, all under the gitignored `.scratch/p5-2026-08-17/`: `AB_RESULTS.md`,
`CHECKPOINT-analysis.md` (read its §8 revision first — §§1-7 describe a state that no longer
exists), `CHECKPOINT-synthesis.md`, `parity_analysis.py`, `parity_synthesis.py`,
`endtoend_synthesis.py`, `ab_p5_insitu.py`, `mem_p5_insitu.py`.

### `resample_filter_half_length_factor` — error and halt, not tolerate (2026-08-18)

The latent defect §5 of the 2026-08-17 verification pass found, decided and applied.
`DecompositionConfiguration.__post_init__` (`peass/config.py`) now raises `ValueError` for
any `resample_filter_half_length_factor < 1`, mirroring the `segmentation_factor` check
beside it.

**The decision, and the reasoning behind it: if the output was going to be wrong anyway,
failing loudly beats failing silently.** Below 1 the fast `_polyphase_decimate` does not
raise — it returns finite numbers that disagree with its own `_polyphase_decimate_padded`
reference by **O(1)** (deviations 0.39 to 2.13 across 33 swept `(down, in_len, rows, dtype)`
combinations, against 2.22e-16 for `hf >= 1`), and the gradient path, which routes to the
padded reference, disagrees with the no-grad path there. The root cause is that the fast
path's grid algebra needs `right_pad >= (hf-1)*down + 1 > 0`. This is a public behaviour
change — a loud error where callers previously got silently wrong output — and it was taken
deliberately rather than by default.

Four new tests in `tests/unit/test_config.py`, **including three ACCEPTANCE cases pinning
that 1 is the first valid value**, so a later edit cannot quietly move the boundary while the
rejection tests still pass. The `_polyphase_decimate_padded` docstring in
`backend_torch/utils.py` was corrected in the same pass: it had gone stale in exactly one
respect, still claiming `hf = 0` "is accepted through the public API".

**The reusable point, worth keeping independently of the fix:** a `__post_init__` that
validates one field of several **reads as though it validates all of them**, and "nothing in
the library passes that" is a statement about the library, not about the API. This dataclass
is public; the field went unvalidated for as long as it did precisely because no internal
caller ever exercised it.

### `hypothesis` live-oracle property suite — applied, both stages (2026-08-18)

`tests/regression/test_reference_oracle_property.py` fuzzes `peass/` against `reference/` as
a live oracle: hypothesis draws a 9-scalar description of an input, a deterministic builder
turns it into audio, `reference/` decomposes it once, and both backends are asserted to
reproduce all four components within per-component, per-backend relative-L2 bars. This closes
the gap the existing tests structurally cannot — every invariant we had (algebraic
reconstruction, gain invariance, gammatone round trip) constrains the *form* of the answer
and is satisfied by a **wrong** one.

Opt-in via `pytest -m oracle` (`addopts = ["-m", "not oracle"]` deselects it; nothing in
`.github/workflows/` reselects it, **deliberately** — decision 2026-08-17, a local opt-in run
is practical and a CI leg is not worth buying). Declared as a `test` extra in `pyproject.toml`
plus line 1 of `requirements.txt`; `[tool.flit.sdist]` already excludes `tests/*`, so nothing
shipped to PyPI can import it. Sized by `PEASS_ORACLE_EXAMPLES`.

Gates: `pytest -q` → **655 passed, 24 skipped, 4 deselected**; `pytest -m oracle -q` →
**4 passed, 679 deselected in 37.48s**. (The deselected count is **4, not 3** — the Stage B
liveness test is `oracle`-marked too.) The 37 s figure is rough wall clock, a **budget
estimate**, explicitly outside ground rule 3.

**The flat torch `1e-3` bar was replaced by per-component bars, and torch has essentially
converged on numpy.** The flat bar dated from a torch that sat uniformly ~2-4e-6 off the
reference, correctly blamed at the time on the float32 `torch.fft.fftfreq` grid. After the
float64 grid (`6e0162d`) and P5's range reduction, torch improved by 1.6e4x to 1.2e7x per
component and now measures 1.985e-13 / 5.072e-11 / 1.448e-10 / 1.266e-10 (worst of 120
draws) — the *same magnitude as numpy*. Against 1e-3 that was seven to ten decades of slack,
which is the vacuous-coverage failure mode this suite exists to prevent. The bars are now
`{true_target 1e-10, target_distortion 1e-8, interference 1e-8, artifacts 1e-7}` — **numpy's
own bars, unchanged, on three of the four components**. Only `true_target` still needs a
looser bar, and only by ~100x (torch 1.985e-13 against numpy 2.060e-15, machine epsilon on
the one component that is not an output of the ill-conditioned solve). That one factor, plus
the fact that the two backends can regress independently, is why the tables stay separate.
`true_target` is bounded at 1e-10 rather than 1e-11 on purpose: 1e-11 clears the observation
by only 50.4x, the floor of this repo's band, against a 2.1x run-to-run spread.

**The numpy bars never moved and no bar has ever been breached** — `{1e-13, 1e-8, 1e-8, 1e-7}`
from the day the file landed, holding with 48.5x-1017.7x margin over 120 draws. Nothing was
quietly loosened to fit.

**The recurring defect is quoting a max-of-N or min-of-N observation as a bound, and this
file has now done it THREE times.** Three runs of 40 show a *different* component is worst in
each run, and that the worst swings by 6.3x (numpy `interference`) to 9.0x (torch
`interference`) between runs of the identical strategy. `_DEGENERACY_FLOOR`'s documented
per-component minima went **1.8x lower on 120 draws than on 14** (`interference` 1.11e-1 →
6.104e-2), and the strategy-**corner** sweep (1.71e-1) is *higher* than the fuzzed minimum, so
corners are not a bound either. **In every case the bar itself was fine and only the comment
was wrong — fix the column, not the bars.** All tables now say "worst over N draws in M runs,
dated" and carry an explicitly stated margin. The 1e-6 floor is unchanged and correct; it
rests on a ~4.8-decade separation from the worst healthy component and ~5.9 decades from the
linear-mix trap (1.24e-12 / 1.21e-12), not on any single figure.

**All eight bars are mutation-proved live**, by a shipped test (`test_every_tolerance_bar_is_live`)
and by bisection. Scaling one component by `1 + eps` drives its relative L2 to ~`eps`, so a bar
`B` resolves a relative gain error of ~`B` on that component; measured, all eight trip at
**1.0000x-1.0026x** their own value. The shipped test brackets each at 0.5x/2.0x and perturbs
one component at a time, which additionally proves the four comparisons are independent. It
calls the *shipped* `_compare_against_reference`, extracted for the purpose — a liveness proof
against a re-implementation proves the re-implementation. Proved through the real pytest path
too: `torch/artifacts` scaled by `1+2e-7` fails with `relative L2 2.000e-07 exceeds the 1e-07
bar`; `1+5e-8` passes. And the liveness test is itself proved live — stubbing the comparison
to report nothing makes it fail in 3.42 s.

**The liveness proof's own limitation, stated honestly and recorded in the file:** because the
perturbation is scaled *by* the bar, the test is invariant to bar size and **would have passed
just as happily against the vacuous flat 1e-3 it now guards**. It proves reachability and
independence, not tightness. The measured tables are what prove tightness. Anyone who reads
the liveness test as sufficient will re-introduce a vacuous bar.

**Partial vacuity is now detectable — and the prescribed fix did not work, which was measured
rather than assumed.** `HealthCheck.filter_too_much` is suppressed, so total filtering raises
`Unsatisfiable` but *partial* filtering passes silently. The plan's fix — a floor on the
**count** of valid examples — was implemented first and measured **nearly blind**: with 95% of
draws forced to filter, hypothesis simply compensated by drawing more and still delivered the
full 40 valid examples, green. The diagnostic quantity is the **acceptance rate** (examples
past both `assume`s over examples entered; `assume` raises from inside the property body, so
both are countable). The same 95% filtering reads 5.0% — **40 valid out of 799 entered** — and
fails loudly. Both ship: rate as the detector, count as the backstop for the case where
hypothesis gives up early, and the rate check skipped below 10 draws so
`PEASS_ORACLE_EXAMPLES=1` stays a legal smoke run. Healthy today: 3 x 40 draws accepted 40 of
40 every time.

**Cost of that check, disclosed in the file:** `@given` is driven by a plain wrapper so the
counts can be checked after the run, and hypothesis' pytest plugin gates statistics on
`is_hypothesis_test`, so `--hypothesis-show-statistics` now prints an empty block for this
file. Three references to that flag were corrected. It is a net win — the one thing those
statistics were used for here, the invalid-example count, is now *asserted* rather than left
to be read — and falsifying examples, shrinking and `@reproduce_failure` blobs are unaffected.

From Stage A, still worth keeping:

- **The shipped cost claim was ~10x too high**, and the reason generalises: it applied the
  **gold-clip** cost (5 s / 16 kHz / stereo / 3 sources, ~12-15 s) as the **per-example** cost
  of a strategy box 10-45x structurally smaller. A real in-box reference call is ~0.3-1.2 s.
- **`PEASS_ORACLE_EXAMPLES` needed a clamp, not just a defensive parse.** It was read at
  module scope, so a bad value was a *collection* error that took down the whole session; and
  `int()` alone is insufficient, because `max_examples=0` parses fine and then raises
  `InvalidArgument`. Parsed lazily, clamped to >= 1.
- **Four "nightly" claims, not three,** all deleted. **PyCharm wart, verified:** a command-line
  `-m` *replaces* the `addopts` one, so running a single oracle test by node ID alone reports
  `0 collected, 1 deselected` — put `-m oracle` in the run configuration's Additional
  Arguments.
- **Both documented traps are genuinely excluded and pinned as characterization tests.** A
  linear estimate in the span of the sources collapses `target_distortion` **and** `artifacts`
  (not `artifacts` alone) to ~1.2e-12 of the estimate RMS — an earlier "~1e-14" was wrong. And
  FIR-related sources reproduce **in mono between two sources** (`[x, 0.5*x]`, 3.6e-1 / 3.9e-1
  relative L2), so "constrain to mono" is *not* a sufficient exclusion, contrary to the earlier
  wording.

Full working under the gitignored `.scratch/hypothesis-2026-08-17/`: `APPLY_PLAN.md`,
`STAGE_B_RESULTS.md`, `prove_oracle_bars.py`, `bars_run{1,2,3}.{txt,json}`,
`bisect_bar_liveness.py`, `mutate_plugin.py`.

### The 2026-08-17 verification pass — the gate run, the A/Bs run, two defects closed (2026-08-17)

A pass that added no optimisation. Everything the 2026-08-15 decomposition pass left owing
got measured or fixed: the ground-rule-2 gate, in-situ A/Bs for the three `utils.py` perf
items, an adversarial re-verification of the three bit-identity claims, and two of the three
outstanding test defects. Green at **606 passed, 24 skipped** — up from 603, because the
gammatone fix adds three tests. `.scratch/` and `benchmarks/results/` are gitignored, so this
entry is the durable copy of all of it.

**All of the code is now committed**, on branch `perf/2026-08-12-decomposition` above base
`e960c5e`, in five commits — which is why the 2026-08-15 entry below, written while the same
work sat in the working tree, reads as though it is uncommitted:

| commit | contents |
| --- | --- |
| `f847817` | the `1e-15` solver ridge + the analysis-modulation fold (`decomposition.py`, and its unit test) |
| `6e0162d` | the float64 gammatone frequency grid |
| `72992d9` | the three `utils.py` items (Kaiser dedup, padless polyphase, fused split/pad) |
| `12f32ea` | §3(a) — the structural half-spectrum negative tests |
| `cd80f01` | §3(b) — the four re-derived parity bars |

Note the split: the perf items and the correctness fixes are in separate
commits, and the two test commits are separate again, so the near-exact `72992d9` can be
reverted without taking the ridge with it.

**Corrected 2026-08-18: this entry said "Nothing is pushed", and that is no longer true.**
All five, plus the docs commit `9f78864` that recorded this entry, are pushed —
`git ls-remote origin perf/2026-08-12-decomposition` returns `9f78864`. Nothing from
2026-08-18 is committed at all; see `TODO.md`'s process-state section for the live picture.

**The single most useful result is a method lesson about leave-one-out A/B, in §2. Read that
one even if nothing else here is relevant.**

#### 1. The ground-rule-2 gate — run, and the tree moved torch onto numpy

`measure.py current` (80.4 s) then `compare.py baseline current`, against the frozen
`e960c5e` capture. The capture was verified genuine before trusting it: `git_commit =
e960c5e4fa…`, `git_dirty = False`.

- **numpy: exactly zero movement** on every correlation, gain error and score. No numpy file
  was touched, and the gate says so rather than being assumed to.
- **torch: every one of the 8 component/channel correlations improved.** `compare.py`
  reports worst correlation drop `+0.000e+00` — none.
- Correlation deficits (`1-corr`) in `current`, against the per-component floors in
  `test_matlab_regression.py`:

  | component | ch0 / ch1 deficit | bar | margin |
  | --- | --- | --- | --- |
  | `true_target` | 4.3804e-08 / 4.3508e-08 | 1e-5 | 228x / 230x |
  | `interference` | 6.9628e-08 / 7.3593e-08 | 1e-5 | 144x / 136x |
  | `target_distortion` | 2.4818e-07 / 2.4307e-07 | 1e-5 | 40x / 41x |
  | `artifacts` | 3.3107e-06 / 3.2732e-06 | 1e-4 | 30x / 31x |

- Worst `|gain_error|` anywhere in `current`: **1.1717e-05** (numpy/`artifacts` ch0) against
  the 1e-4 bar — 8.5x, exactly the value `_GAIN_TOLERANCE`'s comment was derived from.
- **`compare.py`'s one flagged growth is convergence, not regression, and must be read that
  way.** It reports "worst |gain error| growth +6.814e-07 (torch/`target_distortion` ch0)".
  torch went -6.03e-07 → -1.2846e-06, and numpy's value *is* -1.2846e-06. The same holds for
  all eight. A comparison tool that only knows "moved away from the reference" cannot tell
  the two apart; a reader has to.
- **The headline: the tree moved torch onto numpy.** Per-component torch↔numpy agreement:

  | quantity | before | after |
  | --- | --- | --- |
  | correlation | 7.6e-12 … 4.6e-10 | 4.4e-16 … 1.7e-14 |
  | gain error | 9.4e-08 … 6.8e-07 | 8.9e-16 … 7.9e-12 |
  | scores | 5.9e-07 … 3.6e-04 | 3.1e-15 … 8.5e-10 |

- Scores: numpy exactly zero movement; largest torch move **3.643e-04**
  (`interference_perceptual_score`) against a 1.0 bar. Tightest score margin 687x
  (`target_perceptual_score`, 0.0015 from its locked value).
- Worst waveform deviation against the pre-edit capture: **6.144e-05** relative to own peak
  (`matlabref_torch__artifacts`) — the component whose LS Gram min-eigenvalue is ~3.4e-10, so
  a 1-ULP upstream perturbation arrives ~1e6 larger. It moved *toward* MATLAB.

**This corrects a claim in the 2026-08-15 entry below, and the correction is the argument for
the frozen-capture rule.** That entry says the ridge's blast radius is "all 8 scores move
≤2.2e-7". The measured *cumulative* score move is **3.643e-04**, ~1600x larger. That is not
a refutation of the ridge figure itself — the ridge was measured alone, and the float64
grid's own score blast radius was never quoted — but **no per-change number in that entry
predicted the observed cumulative move**. Per-change agent measurements do not compose into
a cumulative one, which is exactly why ground rule 2 asks for a capture frozen before the
whole series rather than a stack of per-change deltas. The entry below now carries the
correction inline.

**`compare.py`'s section-1 timings remain unquotable, and this run illustrates why**:
`numpy_mono` read **0.936x** — an *untouched* backend appearing to slow by 6.4% — with
19.9%/9.4% spreads within the run. Use `compare.py` for accuracy; use `ab.py` for speed.

#### 2. In-situ A/B of the three `utils.py` perf items — and the leave-one-out trap

`benchmarks/ab.py`, driver = a full torch decomposition of
`tests/resources/database/exp01_*` through `benchmarks/measure.py`'s own `decompose()`.
Machine idle. This closes the 2026-08-15 gap "no per-change in-situ A/B exists".

The tree as a whole, current code against all three items reverted:

| workload | ratio (>1 = current faster) | 95% CI | phases | verdict |
| --- | --- | --- | --- | --- |
| mono, 96 calls/phase | **1.0394x** | [1.0310, 1.0467] | 1.0368 / 1.0390 | AGREE |
| stereo, 64 calls/phase | **1.0334x** | [1.0198, 1.0446] | 1.0387 / 1.0284 | AGREE |

Attribution, each item turned on individually from an all-old baseline (mono, 96
calls/phase):

| item | ratio | 95% CI | verdict |
| --- | --- | --- | --- |
| fused split/pad | **1.0327x** | [1.0267, 1.0386] | resolved |
| padless polyphase interior | **1.0230x** | [1.0153, 1.0365] | resolved |
| Kaiser double-design dedup | 1.0053x | [0.9966, 1.0098] | **null** |

**The method lesson, and it is the most reusable thing in this pass. Measured the other way
round — removing ONE item from the otherwise-current tree — all three come back null.**
Kaiser 1.0040x (48 calls/phase), padless 0.9912x (48), fused split/pad 1.0035x (96); every
CI spans 1.0. Those three point estimates multiply to ~0.999, which flatly contradicts the
1.0394x measured for the three together.

The cause is overlap, not measurement error. Items 2 and 3 attack the **same copy**: the
fused split/pad removes one full-signal copy per complex call inside the padded variants,
and the padless interior removes the pad those variants were doing at all. In the current
tree the fast decimate/mixed paths never call `_split_pad_real_imag`, so once either item is
present the other has almost nothing left to save. **Each is sufficient; neither is
individually necessary.** A leave-one-out A/B alone would have reported three duds and
thrown away a real 3.9%.

Generalise it: for overlapping optimisations, leave-one-out measures *marginal* value
against an already-optimised baseline and can read ~0 for a change that matters. Measure
from the un-optimised baseline too, and report both directions. Neither number is wrong —
they answer different questions ("what would I lose by dropping this?" versus "what does
this buy?") and the answers genuinely differ when two changes fight over the same cost.

**The Kaiser dedup is a null on speed in both directions**, and it stands on its memory
result — the 2.5 MB / 64 duplicate complex cache entries its own comment describes — not on
speed. The same is true of the analysis-modulation fold, whose comment already says so. Two
of the six 2026-08-15 changes are therefore memory items with no speed claim, and the
archive should not be read as crediting them with any part of the 1.0394x.

Numerical agreement was re-verified end to end before any timing: Kaiser **bit-identical**
mono and stereo; fused split/pad **bit-identical** mono and stereo; padless 1.674e-16 mono /
4.4801e-14 stereo (~1 ULP at the resampler, arriving amplified through the LS stage).

#### 3. Two of the three outstanding defects, closed

**(a) The blinded gammatone test — fixed, and the recorded diagnosis was half wrong.**

`test_torch_gammatone_half_spectrum_filter_is_the_exact_reflection`. The `1e-13 * peak` bar
did go blind to both traps it exists to catch (trap A `rfftfreq`'s +0.5 Nyquist instead of
`fftfreq`'s first half: 2.446e-07 → 1.110e-16; trap B omitting
`combined[:, -1] = forward[:, -1].real`: 1.486e-07 → 1.304e-16).

**But `TODO.md` concluded the test "catches neither", and that was wrong.** Injecting each
trap into the source and running the *unmodified* test: trap A passed — genuinely blind —
and trap B **failed**, caught by the structural `assert torch.equal(half[:, -1].imag, 0)` a
few lines below the bar, on 1.1796e-16 of residual Nyquist imaginary part. The *bar* was
blind to trap B; the *test* never was. Only trap A was a real hole. The reusable half: when
a tolerance is found to be blind, check the rest of the test before declaring the test
blind — a structural assert next door may already be carrying it, and the review that
measured the bar had measured only the bar.

**And trap A could never have been closed by re-tuning the bar**, which is why the fix is
structural. Under the Nyquist fixup the output at that bin is `Re(H(f_nyq))`, and
`e^{i*pi} = e^{-i*pi} = -1` **exactly**. The two conventions are the same number in exact
arithmetic: the grids differ by exactly 1.0 at that one bin, `torch.exp` of them differs by
~2.4e-16 in the imaginary part, and `.real` then discards even that. **No scalar bar can
separate a difference whose true value is zero.**

Fixed with explicit negative tests: a monkeypatch spy asserting the grid comes from
`fftfreq` at float64 with `rfftfreq` forbidden (the only thing that catches trap A at live
precision), a Nyquist-fixup negative test, and a float32-grid convention test whose
separation is 2.4e6x the bar. Mutation-proved in both directions. Suite 603 → 606 passed.
The general lesson holds: prefer negative tests to a scalar bar, because a bar blinded once
by an accuracy improvement will be blinded again by the next one.

**(b) The four fragile tolerances — re-derived, tightened, and their units fixed.**

| bar | old | measured 2026-08-17 | new bar | margin |
| --- | --- | --- | --- | --- |
| FFT-vs-IIR analyzer | atol 1e-6 | 6.3503e-15 | `1e-12 * peak(numpy)` | 114x |
| mixer gains | atol 1e-6 | 4.6412e-12 | `1e-9 * peak(numpy)` | 247x |
| synthesizer delay/phase | atol 5e-6 | 9.6811e-14 | `1e-12 * peak(numpy)` | 50x |
| DC rejection | 1e-2 absolute | 5.0460e-3 | `dc_bias * 4e-3` (== 1e-2 exactly) | 1.98x, deliberately unmoved |
| `_MAX_AUDITORY_REPRESENTATION_DEFICIT` | 1e-5 | 0.0 exactly | 1e-12 | resolution-limited |
| auditory max-deviation (**new bar**) | never asserted | 6.0163e-10 | `1e-10 * peak(numpy)` | 257x |

**The "re-measure after the solver ridge" worry recorded in `TODO.md` was unfounded, and
that is worth keeping.** The three filterbank numbers reproduced the pre-ridge measurements
exactly, because **none of those three sites runs the decomposition solver** — the ridge
could never reach them. Trace the call path before scheduling a re-measurement; "this change
moves tonal input and that fixture is tonal" is not enough to establish that a site is
downstream of it. DC rejection did not move either: it is pure NumPy, untouched by this
tree, and its change is an exact numerical no-op that only fixes the units. Liveness was
proven by re-injecting the float32 grid: all four re-derived sites fail, by 725x to 3e6x,
and the auditory deficit by 4258x.

**The principle question raised in review is settled: never `rtol`.** One rule, stated once
in the file header and applied at all four sites: `atol = k * max|reference|` with
`rtol = 0.0`, where `reference` is always the **NumPy** array. The reason is mechanical —
`assert_allclose` scales `rtol` by `desired`, its second positional argument, which at every
call site in that file is the **torch** output. A literal `rtol` therefore lets the value
under test set its own bar, exactly what the analyzer comment ~50 lines above argues
against. Applied uniformly even at sites where an `rtol` would have been numerically safe,
so the file carries one rule rather than three exceptions.

The stale pre-ridge patch in the handoff was rejected outright in its **widening** direction
(it loosened all three bars by 1.85-2.92x), as was its `_FILTERBANK_PARITY_RTOL = 3e-6`:
with the float64 grid the noise floor fell 8 decades while the bug signature stayed ~1e-6,
so 3e-6 would leave the bars vacuous *and* still blind to a float32 regression at the gains
site.

Found and refreshed while in the file: all seven correlation-deficit floors have collapsed
to 0..6.7e-16 and its two measured tables were stale by 4-6 decades. The measured columns
were refreshed; the seven *allowed* values were deliberately left alone as a separate
change. A comment describing the pre-ridge onset transient as present was corrected —
post-ridge those components measure 9.1e-6 / 5.5e-6 / 1.3e-3.

**One open risk came out of (b) and it is recorded in `TODO.md` rather than closed here.**
`_MAX_AUDITORY_REPRESENTATION_DEFICIT` was tightened 1e-5 → **1e-12** against a deviation
that measures **exactly zero** on this machine. Headroom that cannot be measured is not
headroom, and ground rule 3 exists because this repo has twice written a one-machine
observation down as a universal invariant. It is mitigated — the new companion `1e-10 *
peak` bar has a real 257x margin, and 1e-12 still catches the float32 regression by 4258x —
but it runs on ubuntu-latest 3.10/3.12/3.14 and windows-latest 3.14 in CI, and it is the one
number in this pass that is not margin-justified.

**(c) The `hypothesis` oracle property suite is still not done** and stays in `TODO.md`.
`hypothesis` is approved as a dev/test extra but is not installed, and the ~700-line patch
is not applied; the five reviewed defects still need fixing as part of applying it. One
decision was taken on 2026-08-17: the suite should be **opt-in only, with the three
"nightly" claims deleted**, rather than wired into CI. Nothing reselects the marker today
and a reference call is only ~1.1-1.5 s, so a local opt-in run is practical and a CI leg is
not worth buying.

#### 4. Adversarial verification of the three bit-identity claims

Each claim was isolated by patching only the changed function, so the gammatone grid change
and the solver ridge were present on both sides and cancel. Negative controls first: a 1-ULP
perturbation injected inside each old shim is detected end to end, so no A/B here was
vacuous.

- **Kaiser double-design dedup: confirmed bit-identical.** 5962 cases (86 rates x 12 lengths
  x 3 rows x 2 dtypes), plus 608 with the FFT fallback force-enabled, plus cold/warm/
  post-eviction cache states, plus 56 degenerate cases. All four internal consumers of
  `get_resample_filter_torch` re-derive their own filter, and there is **no
  `cache_info`/`cache_clear` anywhere in the repo**, so cache-population order is
  unobservable and the function is pure. One non-numerical deviation, unreachable:
  `down == 0` now raises `ZeroDivisionError` from `math.ceil` instead of `ValueError` from
  `firwin`.
- **Analysis-modulation fold: confirmed bit-identical**, including `dL/dest` and every
  `dL/dsrc`. `subbands_output` has **exactly three** consumers and the two besides the gather
  read only `.shape`, *before* the removed multiply — so there is no second value consumer.
  That was checked head-on precisely because the earlier `.flip(-1)` → `index_select` claim
  was refuted for exactly that reason: a second consumer whose strides changed. One refuted
  claim is worth a permanent checklist item.
- **Fused split/merge: values confirmed bit-identical, the docstring's layout claim
  refuted.** No value deviation across ~3200 micro + 2472 consumer + 96 negative-pad cases,
  three full end-to-end decompositions and two end-to-end gradients. But the docstring's
  absolute claim that "`F.pad`'s output is contiguous, hence the final `reshape` is a view
  and every consumer sees byte-for-byte the tensor it saw before" is **false when every pad
  width is ≤ 0**: `F.pad` narrows, keeps the `(1, 2n, 2)` plane strides, and the reshape is
  then a non-contiguous stride-`(1,2)` view at `batch == 1` and a **silent copy** at
  `batch >= 2` (30 of 360 geometries). Values are unaffected, verified through the
  stride-sensitive `unfold`/`matmul` consumers. Unreachable in practice: 132 traced pad pairs
  on a real decomposition with grad on give `min left = min right = 10`, and zero calls at
  `left == right == 0`. The comment is now scoped rather than absolute.
- **The padless ~1 ULP bounds hold, but the two docstrings stated them in different units.**
  `_polyphase_decimate` says 4.44e-16 **absolute** (reproduced; 2.22e-16 on the rates the
  filterbank actually drives). `_polyphase_mixed` says 4.4e-16 **relative to the output
  peak** (holds — worst relative 3.39e-16) while its **absolute** deviation reaches 1.33e-15
  at rate 147/160, a rate the original sweep did not cover. Two adjacent functions with
  numerically similar bounds in different units is an easy way to carry the wrong number
  across; both are now labelled.
- Confirmed incidentally: **`_fft_resample` takes zero calls on the real workload.** Branch
  counts on a 1 s decomposition — interpolate 512, decimate 257, mixed 25, FFT fallback 0.
  Independent support for the existing "keep and document" decision.
- Not covered: CPU only, no CUDA/MPS. Multi-band decimation groups never occur at any
  reachable sampling rate (the groups are singletons in practice), so the fused gather's
  multi-band path is only covered synthetically.

#### 5. A new latent defect, found while correcting a comment

**`resample_filter_half_length_factor = 0` is reachable and silently produces wrong
output.** Carried into `TODO.md` as open work, with the fix recommended but deliberately not
applied.

`DecompositionConfiguration.__post_init__` (`peass/config.py`) validates **only**
`segmentation_factor`, so `resample_filter_half_length_factor=0` is accepted through the
public API. At `hf = 0` the fast `_polyphase_decimate` does **not** raise: it returns finite
numbers that disagree with its own `_polyphase_decimate_padded` reference by **O(1)** —
deviations **0.39 to 2.13** across 33 of the swept `(down, in_len, rows, dtype)`
combinations, against 2.22e-16 for `hf >= 1`. The gradient path is unaffected because it
routes to the padded reference, so **the no-grad and grad paths disagree at `hf = 0`**. Root
cause: the fast path's grid algebra needs `right_pad >= (hf-1)*down + 1 > 0`, i.e.
`hf >= 1`.

**The docstring asserted `hf = 0` was "not reachable", and that was false on both counts** —
it is reachable through the public API, and it is not merely unreachable-but-guarded, it is
unguarded. The comment now states the measured truth. Nothing in the library passes 0
(default 10; the config comment contemplates lowering only to ~3), so it has never been hit,
which is why an unvalidated public field went unnoticed for as long as it did. Reusable:
"nothing in the library passes that" is a statement about the library, not about the API, and
a `__post_init__` that validates one field of several reads as though it validates all of
them.

#### 6. Comment corrections applied to `peass/backend_torch/utils.py`

Applied to the source in this pass, listed here so the archive matches the code:
`_split_pad_real_imag`'s contiguity claim is now scoped and the ≤0-pad corner documented;
`_polyphase_decimate`'s bound is labelled **absolute** and its false "`hf = 0` is not
reachable" replaced with the measured truth above; `_polyphase_mixed`'s bound is labelled
**relative** with the 1.33e-15 absolute figure added. `TODO.md`'s trap-B claim was corrected
in place.

#### What this pass did not do

**P5 was not started.** Three implementation attempts died on transient **API 529
Overloaded** errors — a server-side capacity issue, unrelated to the work. The tree was
verified byte-for-byte untouched after each failure (`peass/` at 86/29/330 changed lines, no
untracked files) and no checkpoint was written, so P5 is exactly where 2026-08-15 left it,
with both hard constraints intact (a true complex `zgemm`, not a widened real `dgemm`; exact
`Fraction` phase range reduction — naive is 3.7e-11/4.4e-11, reduced ~7e-16).

One thing changed that P5 planning must account for: the resampler path P5 touches has now
been measured **1.0394x faster in situ** by §2, and ground rule 3's "kernel harnesses
overstate" warning applies to P5's 1.351x/1.426x isolated-harness numbers. Expect less in
situ.

### The 2026-08-15 correctness and `utils.py` pass — a missing solver ridge, and five exact changes (2026-08-15)

The second pass of 2026-08-15, on the **decomposition** path rather than the metric one.
Six changes across `backend_torch/{decomposition,gammatone,utils}.py` and one test file.

**Recorded here while still uncommitted**, which was a deliberate exception: the work was
green (`603 passed, 24 skipped`, verified twice) but sat in the working tree at branch
`perf/2026-08-12-decomposition` on base `e960c5e`. **It was committed on 2026-08-17** as
`f847817` / `6e0162d` / `72992d9`; see the 2026-08-17 entry above for the mapping. The
accuracy numbers below are
per-change agent measurements, not capture comparisons. The blow-by-blow, the patches and
the harnesses are in `.scratch/handoff-2026-08-15/`, which is gitignored, so this entry is
the durable copy.

**Amended 2026-08-17: the ground-rule-2 gate has now been run, and it passed.** When this
entry was written the gate (`measure.py current` + `compare.py baseline current` against the
frozen `e960c5e` capture) was outstanding, and the entry said so. It was run on 2026-08-17:
numpy exactly zero movement, **every one of torch's 8 component/channel correlations
improved**, worst correlation drop `+0.000e+00`, every correlation deficit and gain error 30x
to 230x inside its bar, and torch↔numpy agreement tightened from 7.6e-12…4.6e-10 to
4.4e-16…1.7e-14 on correlation. See "The 2026-08-17 verification pass" above for the full
tables — including why the one growth `compare.py` flags is convergence onto numpy rather
than a regression.

**The headline is a correctness fix, and the thing it fixed had been misdiagnosed here
for a pass.** torch's `_solve_hermitian_batch` omitted the `1e-15` diagonal ridge that
MATLAB PEASS v2.0.1's `LSDecompose.m` applies ("useful when a sequence of zeroes occurs in
sources s"), which `reference/LSDecompose.py` transcribes and `backend_numpy` mirrors.
torch was the only backend without it. Against the independent oracle
`reference/LSDecompose_tv.py`, numpy was 1.3e-11..2.4e-3 off and torch was off by up to
**2.5e3** — torch was simply wrong on rank-deficient input. On the tonal
`baseline_signals` fixture, torch-vs-numpy correlation:

| component | before | after |
| --- | --- | --- |
| `target_distortion` | 0.016490 | 0.999991 |
| `interference` | 0.016452 | 0.999995 |
| `artifacts` | 0.966525 | 0.998739 |

torch's peak drops from 3.90e2 to 1.559, against numpy's 1.556. Blast radius on the
MATLAB reference clip is small: `true_target` bit-identical, the other three components
move ≤5.9e-8 of peak mono and 3.5e-7 stereo, gold-WAV correlation moves ≤3.0e-12, and all
8 scores move ≤2.2e-7 against 1.0/0.5 tolerances.

> **Corrected 2026-08-17: that ≤2.2e-7 figure does not describe the tree.** The gate
> measures the *cumulative* largest torch score move at **3.643e-04**
> (`interference_perceptual_score`) against a 1.0 bar — ~1600x larger, and still 2700x
> inside the bar. The ridge figure itself is not refuted; it was measured with the ridge
> alone, and the float64 grid's own score blast radius was never quoted anywhere. What is
> refuted is the implicit arithmetic: **no per-change number in this entry predicted the
> observed cumulative move.** Per-change deltas do not compose, which is the whole reason
> ground rule 2 asks for one capture frozen before the series instead of a stack of
> per-change comparisons. Also worth having: the worst waveform move against the capture is
> 6.144e-05 of peak on `artifacts`, whose Gram min-eigenvalue is ~3.4e-10 — a 1-ULP upstream
> perturbation arrives ~1e6 larger there — and it moved *toward* MATLAB.

This reverses the "deliberately not
copied" decision recorded under "Decomposition: torch ~2.2x, numpy ~1.47x"; that entry
now carries a dated correction explaining why its eigenvalue argument did not reach the
output.

**`TODO.md`'s diagnosis of this was wrong in a specific and instructive way, and that is
worth keeping.** It was carried as a *filter/solver warm-up on short input* — "most likely
a filter/solver warm-up at the very start of the signal that the 5 s reference clip
dilutes to nothing" — on the evidence that the transient sat in the first ~60 samples of a
synthetic clip and that the same components measured 1e-11 parity on the MATLAB gold WAVs.
Both observations were real; the inference was not. The divergence covered the whole clip
and **grew with length**; the reference clip hid it because it is not rank-deficient, not
because it is long. The reusable form: "it only shows up on short input" and "it is a
transient at the start" are the same observation twice, and neither is evidence of a
warm-up. Reaching for the independent oracle — which existed by then — settled in one run
what two passes of reasoning from symptoms did not.

Its unit test was rebuilt alongside, because the ridge makes a zero Gram factorize and the
old fixture asserted something the assembly cannot produce: it paired a zero `gram` with an
**independent** random `rhs`. The fixture is now coupled the way the real assembly is
(`gram = T^H diag(w²) T`, `rhs = T^H diag(w²) est`), so `T == 0` forces `rhs == 0`. A
fixture that constructs a state its caller cannot reach tests the solver against a
contract nobody has.

**The other five changes, all on the torch decomposition path:**

- **float32 → float64 gammatone frequency grid** (`gammatone.py`, three sites). An
  accuracy improvement: `torch.fft.fftfreq(N_fft, device=device)` inherited torch's
  *default* dtype, so the cached filter was built on a float32 grid. Pre-existing, not
  introduced by any recent pass.
- **Kaiser double-design dedup** (`utils.py`) — **bit-identical**, `torch.equal` over 46
  rate sweeps. Complex input was causing a complex copy of the filter to be designed and
  cached per rate, used only for two scalars that the FFT fallback needs.
- **Padless polyphase interior** (`utils.py`) — **~1 ULP**, worst 1.94e-16 over a wide
  sweep. `F.pad` was copying the whole signal to give `unfold` a small margin at each end;
  the interior region needs no pad and only the edge samples are handled separately.
- **Fused complex real/imag round trip** (`_split_real_imag` / `_merge_real_imag`,
  `utils.py`) — **bit-identical**, `torch.equal` over 96 cases.
- **The analysis modulation folded into the band gather** (`decomposition.py`) —
  **bit-identical**, `torch.equal` on all four components plus `dL/dinput`, mono and
  stereo. Taken **only** for the ~61 MB mono / ~123 MB stereo temporary it removes; it
  remains a measured speed dud and the comment at the site says so. *(Superseded 2026-08-18:
  P5 folds the modulation into the polyphase taps instead and deletes the matrix outright, so
  the gather has nothing left to fuse. Its "speed dud, do not re-prototype" ledger was
  rewritten rather than deleted and is still at the site.)*

**`TODO.md` predicted the real/imag round trip would not pay, and named a specific
blocker; both were wrong.** It was carried at "confidence it pays: low" with the reasoning
that `view_as_real` puts the real/imag axis last, which is exactly where `unfold` needs the
time axis, so the permute forces the copy straight back. That reasoning about
`view_as_real` is still correct — and irrelevant, because the route that landed does not
use `view_as_real` at all. A confidence rating attached to *one* implementation of an idea
is not a confidence rating on the idea. This is the second wrong prediction in this pass
and the archive keeps both on purpose: the entries that record what the list *believed* are
the ones that stop the same reasoning being trusted twice.

**Two things this pass created rather than closed** — both were carried into `TODO.md` under
"outstanding defects", and **both were fixed on 2026-08-17**; see that entry above for the
re-derived bars, the structural negative tests that replaced the blinded one, and the two
ways this paragraph's own diagnosis turned out to be wrong. The float64 grid made the correct path orders of magnitude more
accurate, and in doing so it **blinded**
`test_torch_gammatone_half_spectrum_filter_is_the_exact_reflection` — the two traps that
test exists to catch went from 2.446e-07 and 1.486e-07 CAUGHT to 1.110e-16 and 1.304e-16
MISSED, both now ~800x *under* its `1e-13 * peak` bar. And it slackened **three** of the
four "fragile" tolerances by five to eight orders of magnitude, which means they now need
re-deriving in the **tightening** direction, the opposite of what `TODO.md` predicted.
(The fourth, the DC-rejection assert in `test_numpy_gammatone.py`, is a NumPy-backend test
that a change to the *torch* frequency grid cannot reach; it is unmoved at 5.05e-3 against
1e-2. Worth stating precisely, because "all four moved" would imply a coupling that does
not exist.) Generalizable, and
this is the third entry in this file to say something adjacent: **an accuracy improvement
can invalidate a test without failing it.** A scalar tolerance separates right from wrong
only while the distance between them is what it was when the bar was set.

**No in-situ speedup number exists for any of this** — at the time of writing. The three
`utils.py` items are perf items and none of them had been through `benchmarks/ab.py` driving
a real entry point, so ground rule 3 forbade quoting a figure for them.

**Amended 2026-08-17: it does now, and the attribution is not what a leave-one-out A/B would
have said.** The three together measure **1.0394x** mono [1.0310, 1.0467] and **1.0334x**
stereo [1.0198, 1.0446], both AGREE. Turned on individually from an all-old baseline: fused
split/pad 1.0327x, padless interior 1.0230x, Kaiser dedup 1.0053x (null). Removed
individually from the current tree, **all three read null** — the two real items overlap on
the same copy, so each is sufficient and neither is necessary. Full numbers and the method
lesson are in "The 2026-08-17 verification pass" above.

### The 2026-08-15 auditory-path pass — metric 1.75x, pipeline 1.24x (2026-08-15)

Four changes, all on the *metric*/auditory path rather than the decomposition, all timed
with `benchmarks/ab.py` driving a real public entry point and all compared for accuracy
against a capture frozen before the series (`perf_20260815_final`) rather than against
each other. Two are bit-identical; two are near-exact, and one of those two is a strict
accuracy *improvement*.

| change | class | metric path | full pipeline |
| --- | --- | --- | --- |
| numpy `_numba_gfb_analyze` band-outer loop + real store | bit-identical | 1.0796x [1.0583, 1.0912] | 1.0405x [1.0306, 1.0499] |
| numpy dB affine fused into the kernel's store | bit-identical | 1.04-1.11x, see below | — |
| torch haircell + 5-stage adaptation fused into one kernel | near-exact | 1.4918x [1.4501, 1.5215] | 1.1761x [1.1604, 1.1850] |
| torch rfft/irfft real-output gammatone | near-exact, ~1 ULP | 1.1306x [1.1061, 1.1498] | 1.0680x [1.0539, 1.0872] |

Each row is that change measured on its own. The two torch rows were then measured
*together*, which had never been done — see "The pair" below, because neither the speed
nor the accuracy composes the way you would guess.

**numpy: band-outer loop order and a real-valued store** (`a13fed2`, bit-identical).
`_numba_gfb_analyze` ran `for sample: for band:`, so every input sample wrote 32 output
cells `num_samples * 16` bytes apart — 32 distinct cache lines per sample into a C-order
`(bands, samples)` array. The bands are independent 4th-order recurrences, so the order
is free; `for band: for sample:` writes each row contiguously and keeps the four states
in registers across the row. State load/store traffic per complex call at n=120336 falls
631.5 MB -> 5.2 KB. Separately, the auditory model discards the imaginary part
immediately, so `_numba_gfb_analyze_real`/`GammatoneAnalyzer.process_real` store float64
where the *state* stays complex — halving the output write (518.4 -> 259.2 MB per mono
`predict`) and handing the next kernel a contiguous array rather than a stride-16 view.
`process` is untouched, so the decomposition still gets its analytic subbands. Measured
1.0239x [1.0161, 1.0376] on the decomposition, 1.0796x [1.0583, 1.0912] on the metric
path, 1.0405x [1.0306, 1.0499] end to end, all three AGREE — plus an unclaimed secondary
win, independently measured: **peak working set on `predict` falls 31.9%, 606.3 ->
412.9 MB**. Bit-identity was attacked rather than assumed: 173 kernel comparisons
including a SIMD-width sweep over 1..128 bands (the hypothesis being that the old
band-inner loop could vectorize where the new one cannot — it does not), denormals, +-0.0,
Inf/NaN payloads, strided and F-order inputs, and 50 chained calls carrying state. Output
and mutated state byte-equal every time; `compare.py` reports 0.000e+00 on every waveform,
score and gain error.

**That item's sizing in `TODO.md` was wrong by ~2.2x, and both causes are reusable
mistakes.** It was carried as "~29 ms, ~1.8% end-to-end" against a delivered ~4% on the
pipeline. First, it counted **3** kernel calls per run, from a decomposition profile;
instrumenting the real entry point shows **18 per `predict`**, because the metric path
runs the filterbank five times per channel. Second, it sized the *decomposition*, which
turned out to be the weaker of the two halves (1.024x there against 1.080x on the metric
path). Profile the entry point you actually intend to speed up, and count the calls
rather than inheriting a count from a different profile.

Note also what this says about the 2026-08-08 rejection below, "feeding the fused NumPy
auditory kernel a contiguous array instead of the `np.real()` stride-16 view: **1.03x**,
and the copy itself costs 0.011 s". That rejected result is a *component* of this win,
obtained without paying for the copy that sank it: the consumer cannot afford to make its
input contiguous, but the producer can store it contiguously for free, because it was
writing that memory anyway. A rejected micro-optimisation is sometimes only rejected at
the call site that would have to pay for it.

**numpy: the dB affine fused into the auditory kernel's store** (`5e52075`,
bit-identical). The global dB scaling ran as two NumPy ufunc passes over a 26 MB array
`_numba_fused_auditory_kernel` had just finished writing. It is now applied at the store
as `scale * (value - offset)` — the same two IEEE operations, in the same order, on the
same value, so bit-identical, but folded into a write that was happening anyway. Verified
three ways: `np.array_equal` against a verbatim copy of the old kernel on real pipeline
subbands, `compare.py` against a frozen pre-change capture (24 waveforms at 0.000e+00, 16
scores identical to 12 decimals), and the full suite.

Measured **1.037x CI [1.014, 1.049]** and **1.105x CI [1.085, 1.124]** on the metric path,
on two separate runs, both AGREE. **Those intervals do not overlap, and that is recorded
rather than averaged away**: `ab.py`'s interval covers within-run variance only, and this
machine drifts more between runs than the interval is wide (ground rule 3 in `TODO.md`).
The honest statement is "1.04-1.11x on the metric path", not either endpoint. Anyone
quoting a single `ab.py` CI as the accuracy of a speedup should read this entry first.

**torch: haircell + 5-stage adaptation fused into one row-major Numba kernel**
(`c482740`). The auditory path ran `F.relu`, an FFT convolution against the haircell
impulse response, a `clamp`, a transpose into the adaptation kernel, a transpose back,
then two more full-size passes for the closing dB affine — each writing a buffer the size
of the subband block. `_numba_fused_haircell_adaptation` writes one. The win is memory
traffic; the FLOPs are unchanged. 1.4918x [1.4501, 1.5215] on
`calculate_auditory_quality_features`, 1.1761x [1.1604, 1.1850] end to end.

**Its class is near-exact but it is specifically *not* a reassociation, and calling it one
understates it.** Nothing is reassociated: the five adaptation stages are transcribed
operation-for-operation from `_numba_adaptation_loop` under `fastmath=False`, same order,
same result. What changes is the *evaluation method* for the haircell filter — the torch
function convolves the one-pole impulse response truncated at 10 ms by FFT, the kernel
runs the recurrence that impulse response comes from. Ground rule 2 permits exactly this:
it computes the same quantity. And it is a strict accuracy improvement, not a trade. The
truncation it removes sits at `g**(0.01*fs) == exp(-20*pi) == 5.2e-28` relative, and the
exponent does not depend on `fs`, so that is twelve orders below float64 eps at every
rate. Against a ~106-bit double-double oracle the recurrence is the *more* accurate of the
two: 1.96e-16 worst absolute against the FFT path's 4.31e-16, and an independent
double-double referee scored it **982x closer to the correctly-rounded result**, with
100% of the worst manufactured divergence being the *old* path being wrong.

**Two numbers, and they must not be conflated** — the README table reports one of them and
a reader can easily take it for the other:

- *Build-to-build delta*: up to **~7.3e-11 of peak**, on two scores. Against the frozen
  capture, `target_perceptual_score` moves +4.328e-11 (7.1e-13 relative) and
  `artifact_perceptual_score` -8.214e-12 (1.1e-13). Every correlation and gain error
  against the MATLAB gold WAVs is identical to the digit, all 24 component waveforms are
  0.000e+00 (the decomposition never enters this path), and 14 of the 16 scores/ratios are
  unchanged.
- *Distance from exact*: **<= 5.1 ULP**, and the new path is the closer of the two.

The first is "the output moved"; the second is "the output is more correct than it was".
A change can have a large first number and a favourable second, and this one does.

Three contract defects surfaced in adversarial verification and were fixed with the change
rather than carried: the transcription test asserted `torch.equal` across a
numpy-vs-numba boundary at `s = g*s + w*x`, exactly the `a*b + c` shape a toolchain may
contract into one FMA (the same failure mode that broke CI on CPython 3.14 for the
adaptation kernel next door — see the 2026-08-09 entry above); README claimed all eight
scores compare equal with `==` between Numba-present and Numba-absent runs, which was true
before this change and false after; and the `float64` condition in
`_can_fuse_haircell_adaptation` was documented as a float32 opt-out and is not one, since
`GammatoneAnalyzerTorch` promotes unconditionally, so a float32 caller does run the fused
kernel. The last was documented as-is and pinned by a test rather than "fixed" with a
float32 branch, because the promotion is intentional and a branch would be a behaviour
change.

**torch: rfft/irfft real-output gammatone on the auditory path** (`a7f2d62`, near-exact,
~1 ULP). The auditory model takes `.real` of the analyzer's output and never touches the
imaginary part, but `process` built the full analytic subbands anyway: a complex `fft`, a
complex `ifft` per band per row, and a full-length complex filter. With
`conj(X[-k]) == X[k]`,

    Re(ifft(fft(x) * H)) == irfft(rfft(x) * Hmod),   Hmod[k] = (H[k] + conj(H[-k]))/2

so `process_real` halves the **inverse** transform — the dominant cost — halves the cached
filter, and hands the next stage a contiguous real block instead of a stride-2 view of a
complex buffer that stays alive behind it. `Hmod` is built straight on the half grid, so
the full-length `H` is never materialised. Measured 1.1306x [1.1061, 1.1498] on the metric
path, 1.0680x [1.0539, 1.0872] end to end. `process` is deliberately left alone — the
decomposition genuinely needs the analytic subbands — so this is a second entry point, not
a flag on the first, which would have put a branch in the hot path of the one caller that
cannot use it.

Two details in `Hmod` are invisible under `allclose` and would each cost ~2e-11 in the
Nyquist bin, so the test asserts bit-for-bit equality against the fold of the full filter
rather than a tolerance: the frequency grid inherits torch's default float32 (see
`TODO.md` — that is a real pre-existing accuracy item), so `exp(-+i*pi)` carries an
~8.7e-8 imaginary residue and `fftfreq`'s Nyquist convention (-0.5) differs from
`rfftfreq`'s (+0.5) by ~1e-7 relative; and the Nyquist bin reflects onto itself, so the
conjugate-grid formula does not hold there and it is set to `Re(H)` directly.

**The pair, measured together — and neither speed nor accuracy composes.** The two torch
changes had only ever been measured individually. Driving the real public entry points on
the reference clip with both candidates in one process, 32 samples per phase:

    calculate_auditory_quality_features   1.7466x  95% CI [1.6608, 1.8574]  AGREE
                                          phases 1.8245 / 1.7990
    predict_perceptual_evaluation_scores  1.2412x  95% CI [1.2243, 1.2597]  AGREE
                                          phases 1.2430 / 1.2336

*Speed.* Disjoint stages in series save additively rather than multiplying. On the
pipeline the additive prediction is 1.27x and the multiplicative one 1.256x; the measured
**1.2412x sits below the additive prediction and covers the multiplicative one at its top
edge**. On the metric path the interval covers both models with the phase estimates
sitting at the additive one. Use the additive model as an upper bound when stacking
independent stage wins, and do not assume the ratios multiply.

*Accuracy, which is the non-obvious half.* **The deviations are not additive — they are
not even independent.** Alone, the rfft change moved four fields, including
`overall_perceptual_score` by -2.117e-09. With the fusion also present those contributions
vanish at score resolution: **all eight torch scores in the pair are bit-identical to the
fusion-only state**, which reports the same two moved fields and the same two magnitudes.
And `target_perceptual_score` lands on exactly **60.99057681753397** under either change
alone *and* under both — 6091 ULP from the baseline. That is not a
representable-resolution floor; it is a discrete landing point that any ULP-scale upstream
perturbation reaches, because the score path's mapping stage quantizes small input moves
onto the same output. Do not read "same value under A, under B and under A+B" as evidence
that a change is inert, and do not read a per-change deviation as a term you can sum.

Pre/post agreed to 2.2e-16 on the metric features and 4.3e-11 on the scores before either
was timed. For the pair: all 16 correlations and gain errors against the MATLAB gold WAVs
identical to the digit, all 24 component waveforms 0.000e+00. Full suite green, 603 passed
/ 24 skipped.

One cost, recorded in README and carried forward in `TODO.md`: Numba-neutrality is gone.
3 of the 8 reported scores now differ between Numba-present and Numba-absent runs, where
before this pass all 8 compared equal with `==` (the fusion alone made it 2 of 8).

### `reference/` — interlinear MATLAB transcription, and what it proved (2026-08-14)

A frozen, deliberately unoptimized transcription of the MATLAB PEASS v2.0.1
**decomposition path** — 25 modules, the 7 PEASS-layer files plus the 18 gammatone
toolbox files. It exists to be an independent second opinion, so it **imports nothing
from `peass`**: numpy, scipy and the stdlib only, and stock `scipy.signal.resample_poly`
rather than this project's resampler, so the two share no code at all.

**The format is the point.** Each module carries its `.m` file's complete source as
`# `-prefixed comments, in order, interleaved with the Python implementing it, fenced by
`# >>> MATLAB` / `# <<< MATLAB`. That makes three checks separable:

- *faithfulness of the copy* — mechanical. `python -m reference.verify_transcription`
  concatenates each module's embedded MATLAB and diffs it against the real `.m`, byte for
  byte including blank lines, licence headers and the presence or absence of a trailing
  newline. **25 passed, 0 failed.** It earned its keep immediately: all 18 gammatone
  modules initially dropped each file's final newline, a one-character-per-file error that
  no amount of reading would have caught.
- *faithfulness of the port* — by eye, each Python block sitting under the MATLAB it
  implements, with MATLAB's variable names kept.
- *faithfulness of the output* — against the gold WAVs.

**The headline: the ~1e-5 residual gap to MATLAB is inherent, not our error.** The
transcription reproduces the gold WAVs at correlation 0.999999956 / 0.999999752 /
0.999999930 / 0.999996689 — the same digits the optimised backends produce. Two
implementations sharing no code land in the same place, so the gap belongs to the
algorithm as specified, and the fast port is exonerated. This is the question
`ARCHIVE.md` previously recorded as unanswerable without a second implementation.

**It also de-circularizes the gain constant.** `_MATLAB_RESAMPLER_GAIN_OFFSET = 1.0025651`
was asserted by `peass`'s own test against `peass`'s own output. The reference
independently lands on 1.002565… — six significant figures — from the `.m` files alone.

Two findings about the *original* code, which is the other thing a transcription buys:

- **The segmentation path in shipped v2.0.1 cannot run.** `aux_mergeWav` sets
  `siz0 = infos_est.TotalSamples`, a scalar, where the `wavread` it replaced returned
  `[nSamples nChannels]` — so `zeros(siz0)` allocates `siz0`x`siz0` (~441000² for 10 s)
  and `siz0(2)` indexes out of bounds. It fails in MATLAB too. The port raises
  `NotImplementedError` rather than emit Python faithful to code that does not work, and
  this independently corroborates commit `07ba346`'s decision to reject
  `segmentation_factor != 1` instead of porting it.
- **Latent bugs in the gammatone toolbox**: `Gfb_Mixer_new` tests `nargin < 4` inside a
  three-argument function, so a caller-supplied `iterations` is always clobbered; and
  `Gfb_Delay_new` can index off the front of `impulse_response`. Neither is reachable with
  PEASS's parameters. Both transcribed as written, with the second raising rather than
  silently wrapping, because a loud difference beats a silent one.

An extra validation worth knowing about: the toolbox ships `README_examples.txt`
containing **real MATLAB console output**, and the port reproduces every printed digit —
`Gfb_Filter_new(10000,1000,100,3,4)` coefficient `0.7526+0.5468i`, normalization factor
`4.7434e-05`, and the filter state after a 200-sample impulse. That is a second gold
source for the gammatone layer specifically, independent of the WAVs.

Nine declared deviations, each marked `# !!! DEVIATION` and pinned by a test so a *new*
silent one fails the build: four `resample` call sites (the sole numerical deviation, and
the entire source of the gain offset), the file-I/O inversion so the primary API returns
arrays, `audiowrite` quantization, the segmentation `NotImplementedError`, and the
gammatone MEX-branch and colon-operator notes.

The single highest-risk spot in the port is `reshape(..., order='F')` in `extractTSIA` —
MATLAB is column-major, and C order would silently transpose the source/channel grouping
so every `(nSource-1)*NChan+(1:NChan)` slice addressed the wrong columns. Also load
bearing: MATLAB's `round` is half-away-from-zero where Python's is half-to-even, which
decides the shade-window length and the synthesis trim offset.

Scope is the decomposition only. The auditory model and the OPS/TPS/IPS/APS score path
are not transcribed — see `TODO.md`.

### History sweep: accuracy and speed across 14 commits (2026-08-13)

`benchmarks/history_sweep.py`, tidy output in `benchmarks/history.csv`, full detail in
the gitignored `benchmarks/results/history_raw.json`. Every commit is judged by **one
fixed yardstick**: the library code comes from the checkout, but the gold WAVs, the
gain-offset constant, the estimate/stereo conventions and the measuring code all come
from HEAD and never vary. That matters — `_MATLAB_RESAMPLER_GAIN_OFFSET` was itself
introduced partway through the swept range, so using each commit's own test constants
would have moved the ruler along with the thing being measured.

Two integrity checks, both clean: `5ed534a` is documentation-only against `c51f76a` and
came out **byte-identical** on every accuracy metric, and accuracy matched exactly
between the forward and reverse passes on all 1344 values (compared as floats, not with
a tolerance).

**The question that prompted it — was some earlier version closer to the MATLAB gold
WAVs? — answers no.** Accuracy is monotone non-worsening toward HEAD. Nothing regressed,
and there is no better starting point buried in the history.

What the sweep did turn up:

- **The accuracy cliff is at `3079de1`**, "default to full-order resampling". Its parent
  `e281b56` sits at correlation 0.9898 on `artifacts` with a **-6.4% gain error on every
  component** — precisely the "~-6% component energy, correlation ~0.99 at 3x" the config
  docstring predicts for `resample_filter_half_length_factor = 3`. Restoring full-order
  resampling bought back four to five orders of magnitude of accuracy and cost a great
  deal of speed: 0.75x numpy, **0.41x torch mono, 0.48x torch stereo**. Everything since
  has been paying that back.
- **`4ab223a` traded gain error between components rather than reducing it.** The
  MATLAB-parity ERB form improved `true_target` gain error ~10x (5.5e-6 -> 5.3e-7) while
  making `artifacts` ~3x worse (-4.2e-6 -> -1.2e-5); RMS gain error across all components
  is essentially unchanged (4.62e-6 -> 4.29e-6). So HEAD is closer to gold on
  *correlation* but not uniformly on *gain*. Worth knowing before anyone treats the ERB
  change as a pure accuracy win.
- **numpy and torch have been within 5e-10 correlation of each other for the entire
  swept history**, including at `e281b56` where both are equally wrong. The backends have
  never been the accuracy story; the resampler always has been.
- `c2c7e76` ("torch haircell lowpass applied time-reversed") shows zero effect here — it
  lives in the auditory/scores path, not the decomposition path. Torch *scores* OOM at
  the three oldest commits (62 GB allocation; the haircell FFT-convolution fix postdates
  them), recorded as `status=partial`; their numpy numbers and torch decompositions are
  fine.

Speed, as realtime factor on a 5 s clip (audio seconds per compute second), swept forward
and again in reverse. All 56 commit x backend x layout pairs agreed between passes within
10% — zero flags, worst 9.4%, typical 1-3%. Outside the noise: `3079de1` the slowdown
above; `d866075` 1.31x numpy mono; `76e53f3` 1.84x torch mono (note that is 1.8x on the
*decomposition*, not the 5.7x pipeline figure in its subject line); `cd061f3` 1.36x numpy
mono; `c51f76a` 1.49x numpy / **2.28x torch mono, 2.87x torch stereo**; `04de76a` 1.17x
torch mono.

**A caution about what this method can and cannot resolve.** The sweep reads `fda3030` at
1.08x numpy and `ae0bdc2` at 1.04-1.06x torch, and calls both "inside noise". That is a
statement about the *sweep's* resolution, not a contradiction of the paired A/B numbers
in the entries below — cross-commit wall clock cannot certify a 4-8% effect, which is
exactly why `benchmarks/ab.py` exists. The magnitudes agree: the A/B put the scatter at
1.07-1.08x and the mixed-rate change at ~1.04x mono. Do not read "inside noise" here as
"no effect"; read it as "this instrument cannot see it, use the other one".

### The 2026-08-12 pass, measured cumulatively against the fixed reference

The three entries below landed together. Measured end to end against a capture frozen
*before* any of them (commit 5ed534a), rather than each against the one before it —
which is the point of rule 2 in `TODO.md`:

- **numpy is exactly bit-identical.** 0.000e+00 on all twelve waveforms, all eight
  scores equal to every printed digit, all gain errors unchanged.
- **torch moves 1.18e-9** on `artifacts` relative to its own peak, essentially all of it
  from the mixed-rate resampler and via least-squares conditioning rather than lost
  precision (see that entry). Correlation against the MATLAB gold WAVs falls at worst
  1.08e-14, gain errors grow at worst 2.33e-12 against the test's 1e-3 bound, and the
  scores move at most 9.0e-11 against a +-1.0 bound. Distance from MATLAB is ~1e-5 and
  was not measurably changed by any of this.
- **Speed**: numpy ~1.07-1.08x, torch ~1.21x mono and ~1.19x stereo, composing the
  paired in-process A/B numbers. Do not quote `measure.py`-style before/after wall clock
  for these — it drifted 6-8% between runs on this machine, enough that untouched
  backends appeared to move more than touched ones, and one final run reported a 41.9%
  spread within six repeats of a single configuration.

The gate went from 291 passed / 21 skipped to 546 / 21; every one of the 255 new tests
came with the changes, and no existing test was weakened, skipped or edited.

### P4 — torch synthesis chain fused, ~1.15-1.18x, and one third of it declined (2026-08-12)

`GammatoneSynthesizerTorch.process` was four full passes over the band block plus a
gathered index tensor: modulate, phase-align and take `.real`, `gather` the delay shift,
`where` the pre-onset mask, `einsum` the mixer gains. It is now one 32-iteration
shift-accumulate,
`out.narrow(-1, d, n).add_((subbands[..., b, :n] * alignment[b, :n]).real, alpha=gain)`,
with the `where` mask falling out for free as the untouched `out[:d]`.
`_get_synthesis_mod_matrix_torch` is replaced by `_get_synthesis_alignment_matrix_torch`,
which caches `modulation * phase_factors` as one tensor keyed on the phase factors' full
dependency set (`delay_sec`, `fs`, cfs, `norms`, `coefs`) as well as the modulation's —
it builds its own `exp` rather than reading the old bare-matrix cache, so resident cache
memory is unchanged at one 62 MB tensor.

**Amended 2026-08-18: `_get_synthesis_alignment_matrix_torch` is deleted, and P5's synthesis
win is THIS win re-banked, not a second one.** P5 folds the modulation into the polyphase
taps, so the 61.61 MB matrix this entry introduced is gone and the decomposition takes
`GammatoneSynthesizerTorch.process`'s `alignment=None` default. The shift-accumulate below is
untouched and its 2.6x on the chain still stands. **Do not add P4's ~1.15-1.18x to P5's
synthesis 1.05-1.07x** — the second deletes the object the first built.

**The third sub-change from the original P4 sketch was dropped.** Writing
`x.real*c.real - x.imag*c.imag` by hand instead of a complex multiply then `.real` is
exactly bit-identical (measured 0.0e+00, as predicted — torch computes the real
component as precisely that expression), but it is **0.85x mono / 0.84x stereo**: the
`.real`/`.imag` views are stride-2 and give up torch's vectorized complex multiply.
Worse, combined with the fused alignment matrix it cancelled that change's entire win —
the full a+b+c combination the TODO proposed measures 1.76x where a+c measures 2.99x on
the same data. There is a warning comment at the site so nobody re-adds it.

Deviation, isolated per sub-change on the synthesizer's own output: the alignment cache
2.84e-16 (reassociating `(x*m)*p` to `x*(m*p)`), the shift-accumulate 4.25e-16 (a
different summation order over the 32 bands than `einsum`), combined 4.25e-16 mono /
4.96e-16 stereo. End to end it is 7.99e-16 of each component's peak, correlations
1.000000000000000, worst MATLAB correlation drop 2.22e-16, and gain error growth exactly
zero on all 16 rows. It adds nothing measurable on top of the mixed-rate change.

Timing, order-balanced paired A/B: the isolated chain is 2.649x mono (50.6 -> 19.1 ms)
and 2.540x stereo (92.3 -> 36.3 ms), beating the 2.21x this was prototyped at; the whole
torch decomposition is 1.175x mono (+191 ms, ~9.5 sigma over 12 interleaved repeats) and
1.148x stereo (+323 ms, ~11 sigma). TODO.md had sized it at 1.08x.

The 246 MB in the original note was the allocation tally rather than the block, and it
checks out: the block is 61.6 MB mono, and the old chain allocated 251 MB mono / 437 MB
stereo across `P*M`, `(X*ph).real`, the int64 index tensor, the bool mask, the clamp, the
gather, the `where` and the `einsum` output. The new tally is 2.9 MB / 5.8 MB, and the
measured working-set rise across the chain falls from +70 MB mono / +377 MB stereo to
+0.4 MB / +3.5 MB. So it was allocator-bound, as claimed.

Two behavioural notes. Gradients are exactly identical (0.0e+00 on `dL/dsubbands`, both
layouts) and `gradcheck` passes, but `alpha=gain` needs Python floats, so `gains` and
`delays` are read out with `.tolist()` in `__init__` — that severs a gradient path to
`gains` which the old `einsum` nominally had. Nothing is lost today, since `gains` is a
constant built from `torch.ones_like` inside a cached function and never requires grad,
but it would matter if the mixer gains were ever made learnable. And negative delays are
now explicitly unhandled: `delays = target_delay - argmax(|ir|[:target_delay+1])` is
non-negative by construction, and the old `gather` would have indexed out of bounds for a
negative one, so there is no prior semantics to preserve. Zero delays *are* reachable
(fs = 16 kHz, 0.004 s) and are covered by tests.

Also measured and not taken: folding the mixer gains into the cached matrix as well. It
came out slightly slower than `alpha=gain` and would have made the cache depend on the
gains.

### torch mixed-rate polyphase — the FFT route retired to a fallback (2026-08-12)

The 3/2 upsample in front of the filterbank and the 2/3 downsample behind it were the
last resamples still taking the FFT linear convolution; six calls per decomposition
(2x 3/2, 4x 2/3). `_polyphase_mixed` now serves general `up/down` by the same collapse
`_polyphase_decimate` uses — that routine is in fact this one at `up == 1`.

The algebra, since a reviewer will want to check it: SciPy's output is
`y[n] = sum_k h[k] v[(n_pre_remove + n)*D - k]` with `v` the input zero-inserted by `U`,
so only `k = s (mod U)` survives, `s = (n_pre_remove + n)*D`. Writing `k = p + U*j`
collapses the sum onto one branch, `y[n] = sum_j h[p + U*j] x[Q - j]` with `p = s mod U`
and `Q = s div U`. Writing `n = m*U + r` then fixes both per residue, because `s` gains
exactly `m*U*D`: each residue is a decimation by `D` against its own
`ceil(len(h)/U)`-tap branch. Reversing within the branch turns it into a forward window,
a common offset makes all windows non-negative, and blocking the input gives one GEMM
plus a shifted-diagonal sum. 21 taps per branch at 3/2, against a ~120k-point transform.

**Accuracy improved where it was measured directly, and the end-to-end move is
conditioning, not error.** Against SciPy at the two rates the decomposition actually
uses, on the real audio: polyphase 4.8e-16 / 3.6e-16 versus the FFT route's 8.4e-16 /
1.07e-15 — the direct 21-tap form is 2-3x closer. Worst deviation against the FFT route
is 1.35e-15 over 264 combinations at `half_length_factor = 10`, inside the 2.3e-15 the
pure-rate GEMMs were verified at.

End to end it moves the torch `artifacts` component by 1.18e-9 relative to its own peak,
which is four orders larger than the 1.8e-13 the 2026-08-10 decomposition work reported
and is worth being precise about. It is amplification, not lost precision: `artifacts`
is the smallest-peak residual component, and the least-squares Gram's minimum eigenvalue
is ~3.4e-10 (see the note on the numpy backend's diagonal regularization above), so a
1-ULP perturbation upstream arrives ~1e6 larger. The control confirms it — holding the
FFT route but choosing a *different valid padding length*, which is exactly equivalent
arithmetic, moves the same component by 1.798e-9, i.e. more than this change does.
Correlation against the MATLAB gold WAVs is unchanged to 13 decimals (worst delta
-1.07e-14), gain errors move at most 2.3e-12 against a 1e-3 bound, and the scores move
at 1e-10 against a +-1.0 bound. numpy is untouched and exactly 0.000e+00.

Timing needed a paired in-process A/B; `measure.py` could not resolve it (torch read
1.03x/1.05x vs baseline while numpy, untouched in that run, moved 1.08x). On the
isolated calls: 3/2 mono 9.01 -> 5.38 ms (1.67x), stereo 20.64 -> 14.57 ms (1.42x);
2/3 mono 7.91 -> 3.26 ms (2.43x), stereo 20.02 -> 9.02 ms (2.22x). On the whole torch
decomposition, 14 interleaved repeats: mono +45.5 ms (sd 53.7, 13/14 positive, ~3.2
sigma), stereo +85.2 ms (sd 59.5, ~5.4 sigma, median 1.038x). That matches the ~90 ms
TODO.md sized the item at. Note the mono *min*-of-14 reads 0.98x — min is the wrong
statistic under this machine's outlier structure, and the paired difference is the
number to trust.

The pure-rate routines were deliberately **not** unified into the general form: at
`down == 1` the general form degenerates to a rank-1 update with a 21-fold intermediate
where `_polyphase_interpolate` does the same FLOPs as a dense GEMM. They are verified
byte-identical against the previous commit across 14 rates x 3 lengths x real/complex.

The FFT route survives as `_fft_resample` behind a `MIXED_POLYPHASE_MAX_ELEMENTS` guard,
and honestly it is precautionary — `taps` is bounded by `2*half_length_factor + 2`, so
the polyphase intermediate is at most ~22x the output and cannot blow up the way the
FFT's `in_len * up` does, and no realistic case was found where polyphase is worse. Its
real value now is as the independent second implementation the cross-check tests compare
against.

### numpy synthesis scatter in place — bit-identical, ~1.07-1.08x (2026-08-12)

`fast_resample_poly` gained an `out=` parameter and the four Numba polyphase kernels
now take their destination buffer as an argument instead of allocating one, so
`run_auditory_synthesis_filterbank` writes the upsampled bands straight into
`processed_subbands` rather than into a temporary it then copies row by row.

**Bit-identical, asserted as byte equality, not a tolerance.** No arithmetic moved:
every dot product and AXPY still reads only the padded input and accumulates in
registers or a local buffer, so only the store address changes. Verified against a
frozen pre-change capture of all four components on both the MATLAB reference input and
the exp01 nonlinear estimate — worst relative deviation 0.000e+00 on every waveform,
reproduced on an independent second run.

The `out=` contract is "the result goes into `out[..., :out_len]`, and nothing from
`out_len` onward is touched", strictly validated (last axis, C-contiguous, matching
leading shape, exact dtype) and raising rather than silently falling back — a silent
fallback here would look like a working optimisation that never engaged. The
decomposition-side guard additionally refuses the in-place path unless
`resample_output_length(...) <= target_length` for every band in the block, so the
truncating branch can never take it; that branch is the only one allowed to discard
samples, and an in-place write there would leak filter tail past `target_length` into a
region the old code deliberately left zero. In the real filterbank all 32 blocks pass
the guard, because decimation factors fall monotonically with band index and bands
sharing a factor have equal length, so `out_len == target_length` exactly.

Timed by a paired in-process A/B — interleaved repeats with the scatter forced on and
off in one process, because end-to-end wall clock drifted ~6% between runs that day
(torch, untouched, moved 1.02-1.07x across the same pair of runs, which is how the
drift was caught). 1.351 s -> 1.311 s mono, 3.231 s -> 3.028 s stereo, min of eight,
with the stereo distributions not overlapping at all. The naive before/after of two
separate `measure.py` runs flattered it to 1.19x; the A/B number is the honest one.

Two things this did *not* buy: the SciPy fallback cannot write in place at all
(`upfirdn` allocates internally), so it keeps the copy route and gets correctness but
no win; and the truncate/short-write branches turn out to be unreachable in practice,
since `np.array([...])` over ragged rows would raise, so all bands in a group have
equal length. Both branches are kept and tested as defensive code.

### Decomposition: torch ~2.2x, numpy ~1.47x (2026-08-10)

Measured on `tests/resources/database/exp01_*` with a *nonlinear* estimate
(`tanh(3*mix)/3` plus shaped noise) so `artifacts` is not numerically degenerate — a
plain linear mix lies exactly in the span of the sources, which makes that component
come out at ~1e-14 and turns any comparison of it into noise-vs-noise. Warmed, min of
six repeats.

| | before | after | speedup |
| --- | --- | --- | --- |
| torch mono | 2.850 s | 1.268 s | **2.25x** |
| torch stereo | 7.286 s | 3.531 s | **2.06x** |
| numpy mono | 2.436 s | 1.644 s | **1.48x** |
| numpy stereo | 4.972 s | 3.430 s | **1.45x** |

Worst deviation against the previous output, relative to each component's own peak:
numpy 4.9e-14, torch 1.8e-13, correlation 1.0 to all 15 digits on all four components.
None of these is an approximation — every one computes the same quantity, and the
differences are reassociation only.

**torch — polyphase GEMM instead of FFT convolution** (`backend_torch/utils.py`), the
dominant win. Resampling was 60.4% of the decomposition over 198 calls. Two structural
problems, not tuning ones: the filterbank has 32 bands with 32 *distinct* decimation
factors, so band grouping yields 32 groups of one or two rows and the FFT has no batch
dimension to parallelise over; and the FFT works at the *undecimated* length either
way, so a band decimated by 1229 transformed a 121500-point spectrum to produce 98
output samples, and its matching synthesis upsample transformed 121500 points to filter
327. With `half_length_factor = 10` the filter is 21 taps *per polyphase phase*
regardless of rate, so the real operation is a small dense GEMM. Interpolation becomes
`(batch*in_len, 21) @ (21, up)` whose `(q, p)` output grid is already in output order;
decimation contracts the input block against all 21 phases at once
(`M = X @ kernel^T`, then `y[n] = sum_j M[n+j, j]`), which is one GEMM plus a
shifted-diagonal sum rather than 21 matrix-vector products. Resampling fell to 32.9%.

Complex signals are additionally split into real and imaginary rows first. The FIR is
real, but torch promotes it and runs full complex arithmetic — four real multiplies per
tap where one will do. That split is exact: the discarded terms are the `a*0`/`b*0`
products of a complex multiply by a real number.

Verified against the FFT path at 2.3e-15 worst relative deviation over 100+ rate and
length combinations, cross-checked against `scipy.signal.resample_poly`, with gradients
matching. `tests/unit/backend_torch/test_torch_utils.py` is new — the torch resampler
previously had no direct test coverage at all.

**torch — `cholesky_ex`/`cholesky_solve` instead of `torch.linalg.pinv`**
(`backend_torch/decomposition.py`). The Gram is Hermitian PSD by construction, and
pinv's rcond cutoff never truncated anything, so the SVD was an expensive way to solve
`Gx = R`. `cholesky_ex` reports failure per matrix in `info`, giving the same
rank-deficient fallback the numpy backend already gets from LAPACK `?posv`. Frames it
rejects — above all silent frames, whose Gram is identically zero — still fall back to
`pinv` and still resolve to the same minimum-norm solution (verified to 1.7e-16). The
failing frames also get an identity shim before the batched refactorization, because a
failed Cholesky leaves uninitialized values that `cholesky_solve` can turn into inf/NaN
and poison the autograd graph even though the values are discarded.

The numpy backend's 1e-15 diagonal regularization was deliberately *not* copied: against
a minimum eigenvalue of ~3.4e-10 it is a ~3e-6 relative perturbation.

**Amended 2026-08-15 — that decision is REVERSED, and the reasoning behind it was aimed
at the wrong quantity.** The ridge is now applied in `_solve_hermitian_batch`, matching
MATLAB `LSDecompose.m`, `reference/LSDecompose.py` and the numpy backend. The
"~3e-6 relative perturbation against a ~3.4e-10 minimum eigenvalue" argument is about the
**weights** the solve returns, and it is correct about them — but it never reached the
**output**, which is what the caller uses, and on rank-deficient input the output without
the ridge is not perturbed, it is wrong. Against the independent oracle
`reference/LSDecompose_tv.py`, numpy was 1.3e-11..2.4e-3 off and torch was off by up to
**2.5e3**. Correlation against numpy on the tonal `baseline_signals` fixture was **0.0165**
on `target_distortion` and `interference`. MATLAB's own comment says what the ridge is for
— "useful when a sequence of zeroes occurs in sources s" — and that is exactly the case
the eigenvalue argument does not cover, because a zero Gram has no minimum eigenvalue to
argue from. The cost of omitting it was correctness on rank-deficient input; the cost of
including it, measured on the MATLAB clip, is ≤5.9e-8 of peak. See "The 2026-08-15
correctness and `utils.py` pass" under Resolved for the full blast radius. The general
lesson: an error bound on an intermediate is not an error bound on the result, and a
deviation argued in eigenvalues does not transfer to a case where the matrix is singular.

**numpy — vectorizable resampler kernels** (`backend_numpy/gammatone.py`). The decimate
kernel's tap loop became `np.dot` on contiguous float64, which numba lowers to `ddot`
(1.90x on the call; the win is AVX, not BLAS threading — measured at
`MKL_NUM_THREADS=1`). The interpolate kernel's loops were inverted so the phase index
`p` is innermost and contiguous, accumulating into an L1-resident length-`up` buffer:
an AXPY LLVM fully vectorizes, against ~0.42 GMAC/s for the original per-output-sample
reduction over strided branch taps. 2.22x on the call. Complex input is split into
real/imag planes — `ddot` needs real operands — and the complex kernels write
`complex128` directly, so the wrapper's recombine pass is gone. Padding buffers use
`np.empty` with only the pad edges zeroed rather than `np.zeros`, skipping a memset of
a region immediately overwritten; that alone took decimate from 1.27x to 1.90x.

Accuracy versus the SciPy path is unchanged to slightly better (scale-relative 8.8e-15
vs 9.1e-15 decimating, 4.0e-16 vs 5.9e-16 interpolating), so this stays inside the
accuracy class `USE_NUMBA_RESAMPLER` already documents. One behaviour change: float32
and complex64 input now promote to float64/complex128 exactly as the SciPy fallback
does, where the old numba path kept float32.

### Torch adaptation recurrence ~2.2-2.3x via a Numba kernel (2026-08-08)

1.97s -> 0.85s for 1s audio, 9.25s -> 4.24s for 5s, with all eight reported scores
comparing equal under `==`. The recurrence was already collapsed to ~7 dispatches per
timestep (see below), which made it *dispatch*-bound: the per-step tensors hold one
element per band, so wall clock tracked kernel launches, not arithmetic — 61.9% of
torch runtime at 5 s. Running it as a Numba kernel removes the dispatch entirely
(43-280x on the kernel in isolation, depending on shape).

The kernel is an operation-for-operation transcription: running product over the
previous step's state, divide, then `c*(1-g) + s*g` as two multiplies and an add, under
`fastmath=False` so LLVM does not reassociate it. The in-place state update is
equivalent to torch's cumprod-then-update because each stage is folded into the running
product *before* that stage is overwritten.

On the reference platform this is exactly bit-identical (`torch.equal` at every shape
measured). **That was over-claimed as universal and broke CI on CPython 3.14**, where
the two diverge by 1.8e-14 — cross-implementation bit-equality depends on whether the
toolchain contracts `a*b + c` into an FMA, which varies by LLVM and torch build. The
local kernel compiles to separate `vmulsd`/`vaddsd`; a newer LLVM need not.

The portable invariant, and what the test now asserts, is ~1e-12. That is not a
weakening: measured, perturbations fall into two cleanly separated bands — roundoff
(FMA contraction 2.6e-16 relative, algebraically-equal EMA reassociation 1.5e-14) and
real transcription errors (using the updated state in the running product 4.4e+00,
resetting the running product per stage 1.0e+00). Fourteen orders apart, so the
tolerance catches every real bug and survives every toolchain.

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
