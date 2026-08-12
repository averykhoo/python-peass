# python-peass

[![Build Status](https://img.shields.io/github/actions/workflow/status/averykhoo/python-peass/main-tests.yml?branch=main&label=tests)](https://github.com/averykhoo/python-peass/actions)
[![PyPI version](https://img.shields.io/pypi/v/python-peass.svg)](https://pypi.org/project/python-peass/)

> This project was ported by Gemini 3.5 Flash from
> https://gitlab.inria.fr/bass-db/peass/-/tree/22c7fc4ef670f8bb6eea9ab4abea98323006b769/v2.0.1

A Python port of the **PEASS v2.0.1** (Perceptual Evaluation methods for Audio Source Separation) toolkit [1].

## Installation

For standard execution, you can install the package directly:

```bash
pip install "python-peass[numba]"
```

If you require high-speed execution (using optimized vector libraries like Intel MKL or Apple Accelerate), it is
recommended to install NumPy and SciPy via Conda first, and then install the package:

```bash
conda install numpy scipy
pip install "python-peass[numba]"
```

### Intel OpenMP conflict on Windows

If you install PyTorch as a PyPI wheel into a conda environment whose NumPy is built
against MKL — the exact combination the section above recommends — the process can
abort on the first call:

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

```
Fatal Python error: Aborted
```

conda's MKL ships `Library/bin/libiomp5md.dll` and the torch wheel ships its own copy
in `site-packages/torch/lib`; whichever initializes second kills the interpreter. There
is no Python traceback, and the abort surfaces inside whichever MKL routine happened to
trigger the second initialization, so it reads like a numerical bug in the backend
rather than a link-time collision.

**Merely having torch installed is not enough to trigger it.** Dispatch decides whether
an input is a tensor by consulting `sys.modules`, so a process that never imports torch
never loads the second runtime and NumPy-only usage is unaffected. (Dispatch used to
import torch unconditionally, which made this abort reachable from pure-NumPy code that
never asked for it; `test_numpy_dispatch_does_not_import_torch` guards against the
regression.) You are exposed once torch is genuinely in the process — using the PyTorch
backend, or importing torch yourself alongside an MKL-backed NumPy.

The real fix is to leave only one OpenMP runtime in the environment, e.g. by taking
NumPy and PyTorch from the same toolchain instead of mixing conda MKL with a PyPI
wheel. Failing that, set one of these **before** the first import (both runtimes read
their configuration at load time, so setting it afterwards is too late):

| workaround | supported? | cost |
| --- | --- | --- |
| `KMP_DUPLICATE_LIB_OK=TRUE` | no — Intel documents it as an unsafe escape hatch | none measured; this is what the project's results are recorded under |
| `MKL_THREADING_LAYER=SEQUENTIAL` | yes | single-threads MKL; shifts the last digit of BLAS reductions (~1e-14 relative) |

The test suite sets `KMP_DUPLICATE_LIB_OK` for you in the root `conftest.py`, so
`pytest` works out of the box; an explicit setting in your environment overrides it.
Application code gets no such help — set it yourself, or fix the environment.

## Quick Start Examples

### 1. Perceptual Quality Score Evaluation

Evaluate estimated audio files saved on disk:

```python
from peass import predict_perceptual_evaluation_scores

original_files = [
    "audio/target_source.wav",
    "audio/interference_1.wav",
    "audio/interference_2.wav"
]
estimate_file = "audio/estimated_target.wav"

scores = predict_perceptual_evaluation_scores(original_files, estimate_file)

print(f"Overall Perceptual Score (OPS):  {scores.overall_perceptual_score:.1f}/100")
print(f"Target Preservation Score (TPS): {scores.target_perceptual_score:.1f}/100")
print(f"Interference Rejection (IPS):    {scores.interference_perceptual_score:.1f}/100")
print(f"Artifact-free Score (APS):       {scores.artifact_perceptual_score:.1f}/100")
```

### 2. Score Evaluation with Waveform and File Expositions

To run the full perceptual scoring pipeline and simultaneously output the physically separated WAV files (True target,
Target distortion, Interference, and Artifacts) to a specific output folder:

```python
from peass import predict_perceptual_evaluation_scores, DecompositionConfiguration

original_files = [
    "audio/target_source.wav",
    "audio/interferer.wav"
]
estimate_file = "audio/estimated_target.wav"

# 1. Configure the output directory for file-writing
config = DecompositionConfiguration(destination_directory="./output_directory/")

# 2. Set return_decomposition=True to expose the physical waveforms and file paths
scores = predict_perceptual_evaluation_scores(
    original_files,
    estimate_file,
    configuration=config,
    return_decomposition=True
)

# 3. Print overall scores
print(f"Overall Perceptual Score (OPS): {scores.overall_perceptual_score:.1f}/100")

# 4. Access the file paths of the generated WAV files on disk
print(f"True target file saved at:      {scores.decomposition_files.true_target}")
print(f"Interference file saved at:     {scores.decomposition_files.interference}")
print(f"Artifacts file saved at:        {scores.decomposition_files.artifacts}")

# 5. Read the raw NumPy arrays directly from memory
target_distortion_array = scores.decomposition_waveforms.target_distortion
```

### 3. Independent Subband Least-Squares Decomposition

You can run the auditory Gammatone/least-squares decomposition engine independently to obtain the isolated physical
sub-components:

```python
import numpy as np
from peass import decompose_distortion_components

# In-memory arrays
target_array = np.random.randn(16000, 1)
noise_array = np.random.randn(16000, 1)
estimate_array = target_array + 0.05 * noise_array

# Run subband least-squares decomposer
result = decompose_distortion_components(
    source_files=[target_array, noise_array],
    estimate_file=estimate_array,
    sampling_frequency_hz=16000.0
)

waveforms = result.waveforms
true_target, target_distortion, interference, artifacts = (
    waveforms.true_target,
    waveforms.target_distortion,
    waveforms.interference,
    waveforms.artifacts
)
```

---

## Scientific Highlights

Traditional evaluation metrics rely purely on linear energy ratios [1].
However, human hearing relies on non-linear auditory transduction, temporal masking, and cognitive thresholds [2].
This package replaces traditional energy ratio metrics (SDR, SIR, SAR) with perceptually motivated objective scores—
**OPS, TPS, IPS, and APS**—which align closely with subjective human listening evaluations [1].

`peass` executes a multi-stage cognitive simulation pipeline to assess separation quality:

1. **Subband Least-Squares Decomposition:**
   Signals are divided into subbands using a complex-valued Hohmann Gammatone Filterbank [1, 3].
   Overlapping temporal frames are projected onto estimated subspaces to isolate physical target distortion,
   interference, and artifact components [1].
2. **Inner Hair Cell Transduction:**
   Approximates the shearing limits of physical hair bundles via half-wave rectification and first-order 1 kHz
   membrane-limit lowpass filters [1, 2].
3. **Auditory Nerve Adaptation:**
   Models physiological forward masking and metabolic neural depletion via five cascaded stages of non-linear feedback
   loops [2].
4. **Perceptual Assimilation:**
   Models cognitive threshold masking where noise below a target reference threshold is partially assimilated or
   masked [2].
5. **Score Prediction:**
   Feeds weighted similarity percentiles into a multi-criteria trained sigmoidal neural network to output scores scaled
   from `0` to `100` [1].

---

## Test Suite & CI/CD

### Installation for Development & Testing

The validation suite implements rigorous numerical, physical, and integration checks.
You will need to `pip install -r requirements.txt` to set up.

You should also install the package in editable mode along with its dependencies:

```bash
pip install -e .
pytest -n auto --cov=peass --cov-report=json
```

Alternatively, if you want to run tests without installing the package, run `pytest` as a Python module:

```bash
python -m pytest -n auto --cov=peass --cov-report=json
```

### Regression Verification

We test our output waveforms directly against the original `.wav` reference waveforms generated by the official MATLAB
PEASS toolbox (located in `references/peass_master_22c7fc4e/v2.0.1/example/`).
Python's outputs must achieve a cross-correlation coefficient exceeding $0.95$ with the MATLAB reference to pass.

### NumPy vs PyTorch backends

The package dispatches to a NumPy backend for array/file inputs and a PyTorch
backend for tensor inputs (selected automatically by input type). The NumPy
backend is the numerical reference and matches the MATLAB toolbox very closely
(cross-correlation > 0.999 with the default full-order resampling; lower
`DecompositionConfiguration.resample_filter_half_length_factor` from `10` toward
`3` to trade a little fidelity for ~25% faster decomposition). The PyTorch backend is designed to be **fully
differentiable** (usable inside a training loop): it replaces the hard
non-linearities and IIR recursions of the reference with smooth, backprop-safe
surrogates (softplus, FIR-truncated filters). As a result its outputs match the
NumPy backend by high correlation rather than to floating-point precision.

### PyTorch backend performance

#### Decomposition

The decomposition is ~2.2x faster as of 2026-08-10 (mono 2.850 s -> 1.268 s, stereo
7.286 s -> 3.531 s on the reference 5 s example). Two changes account for it, neither
an approximation — both compute the same quantity, and output agrees with the previous
release to 1.8e-13 of each component's peak, correlation 1.0 to all 15 digits:

- **The resampler is a polyphase GEMM rather than an FFT convolution.** As of
  2026-08-10 this covered pure interpolation and pure decimation, 196 of its 198 calls;
  since 2026-08-12 a general mixed-rate form covers the rest, so the FFT route is now
  only a guarded fallback. Two things made the
  FFT a poor fit here, and neither was a tuning problem. The filterbank has 32 bands
  with 32 *distinct* decimation factors, so grouping bands by factor yields 32 groups of
  one or two rows and the FFT has no batch dimension left to parallelise over. And the
  FFT works at the *undecimated* length regardless of the rate: a band decimated by 1229
  transformed a 121500-point spectrum to produce 98 output samples. Because the filter
  is 21 taps per polyphase phase whatever the rate, the honest operation is a small
  dense GEMM. Resampling fell from 60.4% of the decomposition to 32.9%.

  Complex subbands are additionally split into real and imaginary rows first: the FIR is
  real, but torch promotes it and runs full complex arithmetic, four real multiplies per
  tap where one will do. The split is exact — the discarded terms are the products of a
  complex multiply by a real number's zero imaginary part.

  The mixed-rate extension (2026-08-12) covers the 3/2 and 2/3 conversions either side
  of the filterbank, six calls per decomposition, worth 1.4-2.4x on those calls and
  ~45 ms mono / ~85 ms stereo end-to-end. Against SciPy at those two rates it is
  *closer* than the FFT route it replaces (4.8e-16 / 3.6e-16 versus 8.4e-16 / 1.07e-15)
  — 21 taps beats a 120k-point transform on rounding as well as on work. It does move
  the `artifacts` component by 1.18e-9 of its own peak, which is conditioning rather
  than error: that component is the smallest-peak residual and the Gram's minimum
  eigenvalue is ~3.4e-10, so a 1-ULP change upstream lands ~1e6 larger. Merely
  re-padding the FFT route to a different valid length moves it further. See
  `ARCHIVE.md` for the index algebra and the control experiment.

- **The per-frame solve uses `cholesky_ex`/`cholesky_solve` rather than
  `torch.linalg.pinv`.** The Gram matrix is Hermitian positive-semidefinite by
  construction and pinv's cutoff never truncated anything, so the SVD was an expensive
  way to solve a linear system. `cholesky_ex` reports failure per matrix, which gives
  the same graceful handling of rank-deficient and silent frames that motivated pinv —
  those frames fall back to it, and agree with the old path to 1.7e-16.

Note that `GammatoneAnalyzerTorch.process` chunks its batch to bound the two
frequency-domain intermediates. That is a memory bound, not a speedup; it measured
neutral. See `_FFT_CHUNK_BUDGET_BYTES`.

#### Adaptation loop

The auditory-nerve adaptation recurrence is sequential in time and tiny per step
(one element per band), so in torch it is bound by kernel-launch latency rather
than by arithmetic — roughly 7 dispatches per timestep, and a 5 s clip is 120000
timesteps. On the common path (CPU, `float64`, no gradient required) the recurrence
therefore runs as a Numba kernel instead, which removes the dispatch entirely:

| clip | before | after | speedup |
| --- | --- | --- | --- |
| 1 s mono | 1.97 s | 0.85 s | 2.32x |
| 5 s mono | 9.25 s | 4.24 s | 2.18x |

The kernel is an operation-for-operation transcription of the torch loop (running
product, divide, then `c*(1-g) + s*g` as two multiplies and an add, with
`fastmath=False` keeping LLVM from reassociating it). On the reference platform —
Windows, CPython 3.10, torch 2.12.1+cpu, numba 0.65.1 — it is exactly **bit-identical**:
`torch.equal` holds at every measured shape and all eight reported scores compare equal
with `==`.

That equality is not portable, and should not be relied on as an invariant. Whether a
toolchain contracts `a*b + c` into a single FMA differs by LLVM and torch build; CPython
3.14 was measured 1.8e-14 from this kernel where 3.10 was exactly equal. What holds
everywhere is agreement to ~1e-14 relative, which is still fourteen orders below any
real transcription error — a wrong stage ordering costs O(1). `tests/unit/backend_torch/test_torch_auditory_model.py`
pins the two implementations together at that tolerance, with the measurements that set
it recorded alongside.

Any other case — CUDA/MPS, `float32`, a gradient-requiring input, or Numba not
installed — falls back to the TorchScript loop, which is unchanged. Training is
unaffected: the differentiable path was never touched.

### NumPy backend performance and numerical reproducibility

The NumPy backend carries a few single-threaded optimizations: **this package's own
code spawns no threads and no subprocesses**, and the speedup comes from SIMD and from
removing per-call overhead. That is a deliberate constraint — see `ARCHIVE.md` for the
Numba `prange` work that was declined under it despite measuring 1.8x.

To be precise about what that does and does not promise: BLAS threading is inherited
from whatever NumPy you installed, and is *not* covered by it. On a stock MKL build
this backend already runs multi-threaded inside BLAS — measured cpu/wall of 3.86 on a
900x900 dgemm and 4.05 on the small gemms the least-squares loop issues, dropping to
~0.9 under `MKL_NUM_THREADS=1`. If you need genuine single-threaded execution, set that
environment variable; the package will not do it for you. Worth knowing that the tiny
least-squares gemms measured *faster* at one thread than at four.

The 2026-08-08/09 entries below are worth ~1.2x end-to-end on the reference example;
the 2026-08-10 kernel rewrite adds a further ~1.47x on the decomposition specifically.
The two are quoted against different scopes and are deliberately not multiplied into a
single cumulative figure, because no end-to-end run was measured across both:

| change | speedup contribution | bitwise effect |
| --- | --- | --- |
| Numba polyphase resampler replacing SciPy `upfirdn` | ~1.13x | reassociated, see below |
| vectorizable rewrite of those two Numba kernels (2026-08-10) | ~1.47x | reassociated, same class |
| LAPACK `?posv` called directly instead of `scipy.linalg.solve(assume_a='pos')` | 11x on that call | **bit-identical** |
| sources pre-padded once instead of a per-frame `np.vstack` | — | **bit-identical** |
| hoisted the conjugate transpose that was built twice per frame | — | **bit-identical** |
| one block-diagonal matmul instead of one per source | — | ~1 ULP (1.9e-15) |
| batched per-band Gram/RHS build instead of per frame | ~1.15x | **bit-identical** |
| analysis modulation matrix cached and shared with synthesis | ~1.18x | **bit-identical** |
| synthesis upsampling scattered straight into the band buffer (2026-08-12) | ~1.07-1.08x | **bit-identical** |

The kernel rewrite landed 2026-08-10 and takes the decomposition from 2.436 s to
1.644 s mono (1.48x) and 4.972 s to 3.430 s stereo (1.45x) on the reference example.
It is the same two kernels doing the same arithmetic, restructured so LLVM can
vectorize it: the decimating tap loop became an `np.dot` that Numba lowers to `ddot`,
and the interpolating kernel's loops were inverted so the polyphase phase index is
innermost and contiguous — an AXPY, against the ~0.42 GMAC/s the original strided
reduction managed. Accuracy against the SciPy path is unchanged to slightly better
(8.8e-15 vs 9.1e-15 decimating, 4.0e-16 vs 5.9e-16 interpolating), so it stays inside
the accuracy class this section already describes rather than widening it. One
behaviour change: `float32`/`complex64` input now promotes to double exactly as the
SciPy fallback does, where the old kernels kept single precision.

The synthesis scatter landed 2026-08-12. `fast_resample_poly` grew an `out=` parameter
and the Numba kernels take their destination as an argument instead of allocating one,
so the synthesis filterbank's upsampled bands are written straight into the band buffer
rather than into a temporary that is then copied row by row. No arithmetic moved — every
dot product and AXPY still accumulates in registers and only the store address changes —
so it is bit-identical, verified as byte equality rather than a tolerance. Measured by a
paired in-process A/B (interleaved repeats with the scatter forced on and off, so machine
drift cancels): 1.35 s -> 1.31 s mono and 3.23 s -> 3.03 s stereo, ~1.07-1.08x. The
`out=` route is strictly validated and refuses anything it cannot serve exactly; the
SciPy fallback still allocates internally, so it gets correctness but not the win.

The batching and modulation-cache entries landed earlier (2026-08-09) and are worth a
further ~1.28-1.32x on the decomposition between them. Both are bitwise exact, verified
byte-level (a `uint8` view, so signed zeros count) across seven configurations — mono,
two-stage, 3-source, all-silent, half-silent, stereo, and a 0.1 s clip.

Batching is bitwise rather than merely close because stacked `matmul` dispatches the
same per-frame GEMM shapes and layouts, so nothing is reassociated; the win is not the
matmul at all but the ~64% of per-frame cost that was Python and NumPy call overhead.
Keeping it exact meant preserving three things that look redundant: the silence bypass
writes explicit zeros (letting `toeplitz @ 0` stand in is numerically equal but can
yield `-0.0`), the analysis and synthesis window multiplies stay separate rather than
folding into one factor, and the synthesis modulation matrix recomputes column `t = 0`
directly instead of conjugating it — at `t = 0` the exponent's sign is multiplied away,
so both directions give `1+0j` and conjugation returns `1-0j`: equal in value, wrong in
bits.

None of these is an approximation: every one computes the same quantity in exact
arithmetic. Two of them reassociate floating-point sums, and floating-point addition
is not associative, so output is not bitwise identical to older releases.

The observed difference is ~2e-11 absolute (~1e-10 of full scale). It looks larger in
relative terms for `artifacts` (~1e-9) purely because that component is a difference
of comparable quantities, so cancellation shrinks the denominator; `true_target` shows
2.5e-15. Correlation against the previous output is 1.0 to all 15 digits, and the four
quality features (and hence OPS/TPS/IPS/APS) agree to 10 significant figures.

**To remove the dominant term**, disable the Numba resampler before the first call:

```python
import peass.backend_numpy.gammatone as gammatone
gammatone.USE_NUMBA_RESAMPLER = False
```

That falls back to the SciPy path (itself bit-exact against the pre-optimization
resampler) and costs the ~1.13x. Measured against the pre-optimization output, this
takes the difference from 2.3e-11 down to 1.9e-15 — *not* to zero, because the
block-diagonal projection matmul in `perform_least_squares_projection` also
reassociates, at about 1 ULP. If you need exactly zero difference, revert that hunk
too; it is worth ~1.03x on its own.

Do *not* instead edit `fastmath=False` onto the resampler kernels: that is bit-exact
but measures ~5% **slower** than SciPy, so it loses both ways, and it requires
clearing Numba's on-disk cache to take effect.

Note that this backend was never bitwise reproducible *across machines*: any BLAS
build or CPU that reassociates differently is subject to the same amplification. Treat
~1e-10 as the backend's reproducibility floor rather than expecting exactness.

**Re-measured 2026-08-09 — the figures above are stale.** They predate the MATLAB-parity
correctness fixes (ERB form, shade window shape and length, resampler gain offset), and
they no longer reproduce. Profiled over all 2338 real frames of the mono two-source
example, the per-frame systems are **well**-conditioned: median condition number 900,
p99 2.5e5, max 1.8e6, and not a single frame above 1e8 — so the "ill-conditioned
systems" explanation no longer describes this code. (The torch backend's Gram measures
the same 1.8e6 maximum independently.) On that configuration the Numba kernels now sit
3.1e-15 from the `USE_NUMBA_RESAMPLER = False` SciPy path, four orders tighter than the
~2e-11 quoted above.

The amplification is also strongly *configuration*-dependent. The larger figures belong
to the multi-source stereo case, where the sensitivity comes from *correlated sources* —
six correlated Toeplitz columns — rather than from any frame being intrinsically
ill-conditioned. Keep ~1e-10 as the floor to design against for multi-source stereo, but
expect ~1e-15 on simple mono material, and do not reach for the `1e-15` diagonal
regularization as though every frame needed it.

### Parallel Execution

To run the test suite across multiple CPU cores using `pytest-xdist`, execute:

```bash
pytest -n auto
```

### What CI actually covers

Worth knowing before trusting a green tick, because the two workflows differ sharply:

| workflow | trigger | coverage |
| --- | --- | --- |
| `branch-tests.yml` | push to any non-`main` branch | ubuntu, **3.14 only** |
| `pr-tests.yml` | pull request | ubuntu 3.10 / 3.12 / 3.14, ubuntu 3.14 **without Numba**, windows 3.14 |
| `main-tests.yml` | push to `main` | ubuntu 3.14 |

A green branch run is therefore weaker evidence than it looks. Two paths it cannot
exercise at all:

- **The TorchScript path.** `torch.jit.script` is unsupported on Python 3.14, so the
  guarded scripting in `backend_torch/auditory_model.py` fails there and the eager
  fallback runs. Only the PR job's 3.10 and 3.12 legs actually script the adaptation
  loop — i.e. only a PR verifies the faster of the two fallback paths, and only a PR
  can catch a `jit.script` deprecation warning escaping the scoped filter.
- **The Numba-free fallback**, which only the `remove-numba` PR leg covers.

Open a pull request before merging anything that touches those paths. As a bonus, the
`remove-numba` leg doubles as an independent measurement of the Numba speedup: on
identical hardware and interpreter it measured 185s against 76s, a 2.4x gap consistent
with the 2.18-2.32x benchmarked locally.

---

## Known deviations from the MATLAB reference

### +0.257% level offset against the MATLAB gold WAVs

The port's decomposition output is a flat, frequency-independent factor **1.0025651**
(+0.0223 dB) larger in amplitude than the MATLAB v2.0.1 gold WAVs in
`tests/resources/matlab_reference/`, on all four output components. This is understood,
deliberate, and **not fixed**.

The cause is resampler filter normalization. `scipy.signal.firwin` defaults to
`scale=True`, which normalizes the Kaiser-windowed sinc to exactly unit DC gain
(`peass/backend_numpy/gammatone.py:797`, `peass/backend_torch/utils.py:85`). MATLAB's
`resample` filter is *not* DC-normalized; its raw DC gain is 0.9993253. The decomposition
performs four resamples per signal path (16k→24k, decimate by `Ndec`, interpolate by
`Ndec`, 24k→16k), so MATLAB accumulates:

| step | MATLAB filter DC gain |
| --- | --- |
| 16k → 24k and 24k → 16k | 0.999394194 each |
| decimate by `Ndec` and interpolate by `Ndec` | 0.999325320 each |
| product | 0.997441484 |
| reciprocal | **1.00256514** |

against a measured 1.0025651. Because the filter half-length is `10*pqmax` and the cutoff
is `1/(2*pqmax)`, the tap set is scale-invariant in `pqmax`, so the factor is identical for
every `Ndec` from 14 to 409 — which is why the discrepancy is perfectly frequency-flat
rather than a filter-shape difference.

Confirmed by a literal line-by-line MATLAB transcription: flipping only that one
normalization moves the ratio from 1.0025651 to 1.0000001, and the remaining 5.4e-4
residual is pure PCM16 quantization noise (0.575 LSB rms measured, 0.577 expected for a sum
of four independent 16-bit roundings, white spectrum).

Note also that the gold WAVs are byte-identical to the precomputed outputs shipped in the
PEASS distribution's `example/` folder, and the v2.0 and v2.0.1 shipped outputs are
identical to each other — the authors never regenerated them for 2.0.1. They are stale
artifacts, not a fresh run of the v2.0.1 source.

**Why it is not fixed:** a resampler should preserve DC gain. Matching the reference would
mean deliberately introducing a 0.26% (0.022 dB) attenuation to chase a reference that is
itself arguably wrong.

The offset originally went unflagged because the regression test asserted on
cross-correlation, which is scale-invariant, plus a `0.5 < ratio < 2.0` sanity band that
0.26% passes trivially. `tests/regression/test_matlab_regression.py` now asserts the RMS
ratio *equals* 1.0025651 to within 1e-3 instead, so the known offset is locked rather than
merely tolerated: any new gain regression, in either direction, breaks the test. The worst
measured deviation from the constant is 1.5e-5, on the `artifacts` component — the quietest
of the four, and so the one whose gold WAV sits closest to the PCM16 quantization floor.

---

## References

1. **V. Emiya, E. Vincent, N. Harlander, and V. Hohmann**,
   *"Subjective and objective quality assessment of audio source separation"*,
   IEEE Transactions on Audio, Speech, and Language Processing, 19(7):2046–2057, 2011.
2. **R. Huber and B. Kollmeier**,
   *"PEMO-Q — A New Method for Objective Audio Quality Assessment Using a Model of Auditory Perception"*,
   IEEE Transactions on Audio, Speech, and Language Processing, 14(6):1902–1911, 2006.
3. **V. Hohmann**,
   *"Frequency analysis and synthesis using a Gammatone filterbank"*,
   Acustica/Acta Acustica, 88(3):433–442, 2002.
