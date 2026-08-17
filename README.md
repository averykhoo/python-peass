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

One selection is **opt-in and deselected by default**, so a normal run reports it as
`deselected` rather than skipped:

```bash
pytest -m oracle          # ~40 s: fuzz both backends against reference/ as a live oracle
```

`tests/regression/test_reference_oracle_property.py` uses `hypothesis` (a `test` extra, and
in `requirements.txt`) to draw inputs, decompose each one with `reference/`, and assert that
both backends reproduce all four components within per-component, per-backend bars. It is
deliberately **not** wired into CI. A command-line `-m` replaces the default one, so
`pytest -m oracle` selects it and `pytest -m ""` clears the filter entirely.

### Regression Verification

We test our output waveforms directly against the original `.wav` reference waveforms generated by the official MATLAB
PEASS toolbox, checked in at `tests/resources/matlab_reference/`. Those four input WAVs are byte-identical to the
`example/` inputs shipped with MATLAB PEASS v2.0.1, so the gold outputs are that toolbox's own output on that input.
The thresholds live in `tests/regression/test_matlab_regression.py` and are **per component**, because the components
differ by ~75x in how closely they track MATLAB — a single bar set for the worst of them would let the best degrade
75-fold without failing. Expressed as the largest tolerated $1 - \text{corr}$, with the margin over what is actually
measured on the reference platform:

| component | measured $1-\text{corr}$ | allowed | margin |
| --- | --- | --- | --- |
| true_target | 4.4e-08 | 1e-05 | 228x |
| interference | 7.4e-08 | 1e-05 | 136x |
| target_distortion | 2.5e-07 | 1e-05 | 40x |
| artifacts | 3.3e-06 | 1e-04 | 30x |

The measured column is the 2026-08-17 capture and is now an **upper bound** rather than a
current reading: the 2026-08-18 fold improved all eight torch component/channel correlations
again, and numpy did not move at all. The allowed column is unchanged.

`artifacts` gets its own decade: it is the smallest-peak residual, and the least-squares Gram's minimum eigenvalue is
~3.4e-10, so a 1-ULP perturbation upstream arrives ~1e6 larger. NumPy and PyTorch share these floors — on CPU they
agree with each other to 5e-10 correlation across the whole measured history, and as of 2026-08-18 to ~1e-15 on this
clip — while CUDA, never measured against these WAVs, keeps a documented loose bound instead of inheriting them. There is also an RMS gain check against the
locked resampler offset, at 1e-4 against a measured worst of 1.17e-5.

The margins are deliberately generous rather than snug: these are one machine's numbers and CI also runs
`ubuntu-latest`. This test is a floor against gross regression, not a drift detector — fine-grained drift is caught by
comparing against a frozen capture with `benchmarks/compare.py`, which resolves to ~1e-16.

### The `reference/` transcription

`reference/` holds a frozen, deliberately unoptimized transcription of MATLAB PEASS
v2.0.1's decomposition path — 25 modules, each carrying its `.m` file's complete source as
interleaved comments. It is developer tooling, excluded from the sdist and never imported
by the library. Its value is being an *independent* second opinion, so it imports nothing
from `peass` and uses stock `scipy.signal.resample_poly` rather than this project's
resampler: the two share no code.

The interleaved format splits verification into three checks that can each be made
separately, rather than one act of faith:

```bash
python -m reference.verify_transcription   # the embedded MATLAB is byte-exact vs the .m files
```

That is the *copy* check, and it is mechanical — 25 modules, diffed line for line
including blank lines, licence headers and trailing newlines. The *port* is checked by
reading each Python block against the MATLAB directly above it. The *output* is checked
against the gold WAVs by `tests/regression/test_reference_transcription.py`.

What it established: the transcription reproduces the gold WAVs at correlation
0.999999956 / 0.999999752 / 0.999999930 / 0.999996689 — the same digits the optimized
backends produce. Two implementations sharing no code landing in the same place means the
residual gap to MATLAB belongs to the algorithm as specified rather than to this port, and
it gives the `+0.257%` offset documented below a second, independent derivation instead of
being asserted against our own output. See `ARCHIVE.md` for the full account, including
two latent bugs the transcription surfaced in the original MATLAB.

The MATLAB sources it transcribes are not redistributed here; the verifier skips cleanly
when they are absent, so a fresh clone and CI are unaffected.

### NumPy vs PyTorch backends

The package dispatches to a NumPy backend for array/file inputs and a PyTorch
backend for tensor inputs (selected automatically by input type). The NumPy
backend is the numerical reference and matches the MATLAB toolbox very closely
(cross-correlation > 0.999 with the default full-order resampling; lower
`DecompositionConfiguration.resample_filter_half_length_factor` from `10` toward
`3` to trade a little fidelity for ~25% faster decomposition — **values below `1` now raise
`ValueError`** from `__post_init__`, as of 2026-08-18. They used to be accepted and silently
produced O(1)-wrong output on the torch fast path, with the no-grad and gradient paths
disagreeing; see `ARCHIVE.md`). The PyTorch backend is designed to be **fully
differentiable** (usable inside a training loop): it replaces the hard
non-linearities and IIR recursions of the reference with smooth, backprop-safe
surrogates (softplus, FIR-truncated filters). As a result its outputs match the
NumPy backend by high correlation rather than to floating-point precision.

### PyTorch backend performance

#### Decomposition

The decomposition is ~2.2x faster as of 2026-08-10 (mono 2.850 s -> 1.268 s, stereo
7.286 s -> 3.531 s on the reference 5 s example), with a further ~1.21x mono / ~1.19x
stereo from the 2026-08-12 pass below — that figure composes the paired in-process A/B
measurements, because end-to-end wall clock on this machine drifts 6-8% between runs and
cannot resolve changes this size. None of these is an approximation: every one computes
the same quantity, correlation against the previous release is 1.0 to all 15 digits, and
correlation against the MATLAB gold WAVs is unchanged to 13 decimals (worst delta
-1.08e-14, with gain errors moving at most 2.3e-12 against a 1e-3 bound):

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
  the same graceful handling of rank-deficient frames that motivated pinv — those frames
  fall back to it, so a genuinely singular frame still resolves to a minimum-norm
  solution.

  As of 2026-08-15 the solve also applies MATLAB PEASS v2.0.1's own `1e-15` diagonal
  ridge (`lambda` in `LSDecompose.m`, "useful when a sequence of zeroes occurs in
  sources s"), which the NumPy backend and `reference/` always had and torch did not.
  That changes what happens on **silent** frames specifically: a silent frame's Gram is
  identically zero, `1e-15 * I` is positive definite, so `cholesky_ex` now succeeds on it
  instead of routing it to pinv. The two agree, because `gram` and `rhs` are assembled
  from the same Toeplitz factor — a zero factor forces `rhs` to be exactly zero, and the
  ridged solve returns bitwise-zero weights, which is what `pinv(0) @ 0` and MATLAB both
  give. Without the ridge, torch was wrong rather than merely different on rank-deficient
  input: against `reference/LSDecompose_tv.py` it was off by up to 2.5e3, and on a tonal
  two-source fixture its `target_distortion` and `interference` correlated with NumPy at
  0.0165. See `ARCHIVE.md` for the full blast radius on the reference clip.

- **The synthesis chain is one shift-accumulate rather than five passes** (2026-08-12).
  `GammatoneSynthesizerTorch.process` used to modulate, phase-align and take `.real`,
  `gather` a delay shift, `where` a pre-onset mask, then `einsum` the mixer gains — five
  full passes over the band block and 251 MB of allocation mono, 437 MB stereo. It is
  now a 32-iteration accumulate into a narrowed output view, with the mask falling out as
  the region the accumulate never touches. Allocation drops to 2.9 MB / 5.8 MB and the
  measured working-set rise across the chain from +70 MB mono / +377 MB stereo to
  +0.4 MB / +3.5 MB. Worth 2.6x on the chain and ~1.15-1.18x end-to-end; deviation
  8.0e-16 of each component's peak, from reassociating the phase multiply and summing
  the 32 bands in a different order than `einsum` did. Gradients are bit-identical.

  This change also cached the modulation and phase factors premultiplied as one 61.6 MB
  tensor. **That tensor no longer exists** — the 2026-08-18 fold below deletes it, and the
  decomposition now passes no alignment argument at all. The two wins are therefore the
  same win re-banked and must not be added.

A further pass on 2026-08-15 worked the same path. One of its changes is a correctness
fix and two of them move the output; the rest are exact. Each class below was
re-verified adversarially on 2026-08-17 — each bit-identity claim isolated by patching only
the changed function, with a 1-ULP perturbation injected into the old code first as a
negative control, so none of the comparisons was vacuous:

| change (2026-08-15) | class, verified 2026-08-17 | verified by |
| --- | --- | --- |
| MATLAB's `1e-15` diagonal ridge added to the per-frame solve | correctness fix; moves output | gold-WAV correlation *improved* on all 8 component/channel pairs; see the cumulative figures below |
| gammatone frequency grid built at `float64` instead of torch's default `float32` | accuracy improvement; moves output | see below |
| complex input no longer designs the Kaiser filter twice per rate | **bit-identical** | 5962 cases (86 rates x 12 lengths x 3 rows x 2 dtypes), plus 608 with the FFT fallback forced, plus cold/warm/post-eviction cache states and 56 degenerate cases |
| polyphase interior runs padless instead of copying the whole signal | ~1 ULP | 4.44e-16 **absolute** at `_polyphase_decimate` (2.22e-16 on the rates the filterbank drives); 1.33e-15 absolute / 3.39e-16 relative at `_polyphase_mixed`; end to end 1.674e-16 mono / 4.4801e-14 stereo |
| fused complex real/imag split and merge | **bit-identical** | ~3200 micro + 2472 consumer + 96 negative-pad cases, three full decompositions and two end-to-end gradients |
| analysis modulation folded into the band gather (**superseded 2026-08-18**, see below) | **bit-identical** | `torch.equal` on all four components, `dL/dest` and every `dL/dsrc`, mono and stereo |

**Corrected 2026-08-17, two of the numbers this table used to carry.** The ridge row said
"≤5.9e-8 of peak mono, 3.5e-7 stereo on the reference clip; gold-WAV correlation moves
≤3.0e-12". Those were per-change measurements and they do not describe the pass. Against a
capture frozen before any edit, the *cumulative* worst waveform move is **6.144e-05** of peak
(on `artifacts`, whose least-squares Gram has a ~3.4e-10 minimum eigenvalue, so a 1-ULP
upstream perturbation arrives ~1e6 larger) and the largest score move is **3.643e-04**
against a 1.0 tolerance — ~1600x larger than the ≤2.2e-7 the archive entry implied. Every
one of the eight correlations moved *toward* MATLAB, and torch↔numpy agreement tightened from
7.6e-12…4.6e-10 to 4.4e-16…1.7e-14. The padless row said "worst 1.94e-16 over a wide sweep",
which understated it and mixed units: the two functions state their bounds in *different*
units, absolute at `_polyphase_decimate` and relative at `_polyphase_mixed`, and the mixed
path reaches 1.33e-15 absolute at rate 147/160 — a rate the original sweep did not cover.

**Speed, measured in situ on 2026-08-17.** The three `utils.py` items (Kaiser dedup, padless
interior, fused split/merge) measure **1.0394x** together on a full torch mono decomposition,
95% CI [1.0310, 1.0467], and **1.0334x** stereo [1.0198, 1.0446] — both AGREE under phase
swapping. That is the only speed claim this pass supports, and it is a claim about the three
together: measured individually from an all-old baseline they read 1.0327x (fused split/pad),
1.0230x (padless) and 1.0053x (Kaiser, null), while measured by removing one at a time from
the finished tree **all three read null**, every CI spanning 1.0. The two real items overlap
on the same full-signal copy, so each is sufficient and neither is necessary. See
`ARCHIVE.md` before quoting a per-item figure.

Two of the six are memory changes with no speed claim at all: the analysis-modulation fold
removes a ~61 MB mono / ~123 MB stereo temporary and measures flat on the clock, and the
Kaiser dedup is a speed null in both A/B directions, standing on the 2.5 MB of duplicate
complex cache entries it removes.

**The modulation is folded into the polyphase filters** (2026-08-18), on both the analysis
and the synthesis side. Complex-exponential modulation distributes through convolution
exactly, so the full-length modulation multiplies fold into the 21-tap filters and the two
cached matrices — 61.44 MB of analysis modulation and 61.61 MB of synthesis alignment — are
gone outright, replaced by `fast_demodulate_decimate_torch` and
`fast_interpolate_modulate_torch`. What survives is a residual modulation at the *decimated*
rate, ~0.35% of one full-rate pass summed over the 32 bands. Two implementation details are
load-bearing rather than incidental: the folded kernel is a **true complex `zgemm`** (the
widened real variant that would have kept the real/imaginary split measures 0.717x), and the
phase is **range-reduced mod one turn before the exponential** in exact integer arithmetic
(`2*pi*fc*n/fs` reaches 2.3465e5 rad at the top band, where a naive `exp` loses ~4e-11 rad).

Measured **1.1673x** on a full torch mono decomposition, 95% CI [1.1631, 1.1751], and
**1.1571x** stereo [1.1486, 1.1643], both AGREE under phase swapping; the two halves are
independent rather than overlapping and resolve individually in both A/B directions, at
1.105x (analysis) and 1.05-1.07x (synthesis) of the whole decomposition. Resident complex
tensors fall 181.4 MiB → 68.9 MiB, a **117.9 MB** reduction, against a predicted ledger of
the same figure.

The folds themselves are a reassociation at 6.4e-16 of each component's peak end to end. The
range reduction is a separate, strictly-improving accuracy change and is where P5's movement
against the previous build lives: 2.3e-12 to 3.7e-12 of peak on the four components, toward
truth. Against the frozen pre-series capture, numpy is exactly unmoved and **all eight torch
correlations improved**; torch and numpy have now converged to the point that
`overall_perceptual_score` is identical on both backends to twelve digits.

One property is worth knowing before touching either half: the two matrices were **bitwise
exact conjugates**, so their unreduced-exponential phase errors cancelled end to end.
Range-reducing only one side is therefore a locally-correct change that makes the pipeline
*worse* — it breaks the cancellation and leaves the full error on the output. Both sides are
reduced by the same code path for exactly that reason. See `ARCHIVE.md`.

Note that `GammatoneAnalyzerTorch.process` and `.process_real` chunk their batch to
bound the frequency-domain intermediates. That is a memory bound, not a speedup; it
measured neutral. It is *not* bit-invariant in the chunk width — torch's batched FFT
does not sum a row the same way alone as alongside others (7.1e-16 measured) — so the
budget is a memory knob and not something to tune. See `_FFT_CHUNK_BUDGET_BYTES`.

#### Adaptation loop, and the haircell stage fused into it

The auditory-nerve adaptation recurrence is sequential in time and tiny per step
(one element per band), so in torch it is bound by kernel-launch latency rather
than by arithmetic — roughly 7 dispatches per timestep, and a 5 s clip is 120000
timesteps. On the common path (CPU, `float64`, no gradient required) the recurrence
therefore runs as a Numba kernel instead, which removes the dispatch entirely:

| clip | before | after | speedup |
| --- | --- | --- | --- |
| 1 s mono | 1.97 s | 0.85 s | 2.32x |
| 5 s mono | 9.25 s | 4.24 s | 2.18x |

As of 2026-08-15 that kernel also absorbs the stage in front of it. Half-wave
rectification, the 1 kHz inner-haircell lowpass, the absolute-threshold clamp, the five
adaptation stages and the closing dB affine now run in a single row-major pass
(`_numba_fused_haircell_adaptation`), behind the same four conditions
(`_can_fuse_haircell_adaptation`). The win is memory traffic, not arithmetic: the
unfused route materialises a relu, an rfft spectrum, an irfft output, a clamp, two full
transposes and two more full-size buffers for the affine, where this writes one buffer.
Measured 1.4918x CI [1.4501, 1.5215] on `calculate_auditory_quality_features` and
1.1761x CI [1.1604, 1.1850] end-to-end, by paired in-process A/B.

The adaptation stages are an operation-for-operation transcription of the torch loop
(running product, divide, then `c*(1-g) + s*g` as two multiplies and an add, with
`fastmath=False` keeping LLVM from reassociating it). On the reference platform —
Windows, CPython 3.10, torch 2.12.1+cpu, numba 0.65.1 — that half is exactly
**bit-identical**: `torch.equal` holds at every measured shape.

That equality is not portable, and should not be relied on as an invariant. Whether a
toolchain contracts `a*b + c` into a single FMA differs by LLVM and torch build; CPython
3.14 was measured 1.8e-14 from this kernel where 3.10 was exactly equal. What holds
everywhere is agreement to ~1e-14 relative, which is still fourteen orders below any
real transcription error — a wrong stage ordering costs O(1). `tests/unit/backend_torch/test_torch_auditory_model.py`
pins the two implementations together at that tolerance, with the measurements that set
it recorded alongside, and asserts bounds rather than equality for exactly this reason.

The haircell half is a different evaluation of the same filter, and it changed a
contract this section used to state. The torch function convolves the one-pole
filter's impulse response, truncated at 10 ms, by FFT; the kernel runs the recurrence
that impulse response comes from. They are the same filter to
`g**(0.01*fs) == exp(-20*pi) == 5.2e-28` relative — the exponent does not depend on
`fs`, so the truncation sits twelve orders below float64 eps at every rate — and
against a ~106-bit double-double oracle the recurrence is the *more* accurate of the
two (1.96e-16 worst absolute against the FFT path's 4.31e-16). What it costs is
Numba-neutrality. Scoring the reference clip with Numba present and absent, **3 of the
8 reported scores now differ**, where before the fusion all 8 compared equal with `==`:

| score | Numba present | Numba absent | delta | relative |
| --- | --- | --- | --- | --- |
| `overall_perceptual_score` | 17.665736258846312 | 17.665736256729744 | +2.117e-09 | 1.2e-10 |
| `interference_perceptual_score` | 20.484476217617885 | 20.484476217625293 | -7.407e-12 | 3.6e-13 |
| `artifact_perceptual_score` | 76.30118899632163 | 76.30118899634628 | -2.466e-11 | 3.2e-13 |

The other five are bit-identical between the two runs. (With only the fusion and not
the real-output analysis path below, it was 2 of 8: `target_perceptual_score`
+4.328e-11 and `artifact_perceptual_score` -8.214e-12. Which fields move, and by how
much, is not stable under unrelated ULP-level changes upstream — the *fact* that some
do is the durable part.) These are roundoff on a 0-100 scale, but they are a difference
where there was none, so installing or removing the optional `numba` extra no longer
guarantees the last digits of a score are untouched.

**Read that table as an illustration, not as a current measurement.** It was taken before
the 2026-08-15 decomposition pass, which moved torch numerics in three separate places
(the solver ridge, the `float64` frequency grid, the ~1 ULP padless polyphase), so which
three of the eight scores differ has very likely changed again. It has not been
re-measured. The durable claim is the one above and it is unaffected: *some* scores
differ, by ~1e-9 or less. Do not treat the specific field set as a contract, here or in a
test.

Still not re-measured as of 2026-08-18, and now stale in the values as well as in the field
set: torch numerics moved a third time when the gammatone modulation phase was range-reduced,
and `overall_perceptual_score` on this clip now reads `17.665620658898` rather than the
`17.6657362…` above. Neither frozen-capture gate touches this — the gate compares the tree
against itself before the edits and says nothing about Numba-present versus Numba-absent — so
the table stays flagged stale rather than being edited from memory. Re-measuring it is one
run with `numba` importable and one without; it is the cheapest open job in `TODO.md`.

Callers on CUDA/MPS, callers that need a gradient, and installs without Numba fall back
to the torch functions, which are unchanged; training is unaffected, because the
differentiable path was never touched. **`float32` callers do not fall back**, despite
the `float64` condition in `_can_fuse_haircell_adaptation` reading as though they
would: `GammatoneAnalyzerTorch` promotes unconditionally (`process` to `complex128`,
`process_real` to `float64`), so the subbands that reach the gate are float64 whatever
dtype the caller passed. A `float32` call to `calculate_auditory_quality_features`
therefore runs the fused kernel and moves with it — measured 2.4e-12 (2.3e-15 of peak)
on the internal representation, and at most 4.4e-16 absolute on the four features. This
was documented the other way round when the fusion first landed; it is corrected here
rather than "fixed" in code, because adding a float32 branch would be a behaviour
change and the promotion is intentional.

#### Real-output gammatone on the auditory path

The auditory model takes `.real` of the analyzer's output and never looks at the
imaginary part, so as of 2026-08-15 it calls `GammatoneAnalyzerTorch.process_real`,
which computes that real part directly: `irfft(rfft(x) * Hmod)` with a Hermitian-folded
half filter `Hmod[k] = (H[k] + conj(H[-k])) / 2`, instead of `ifft(fft(x) * H).real`.
That halves the inverse transform — one per band per row, and by far the dominant cost
here — halves the cached filter, and hands the next stage a contiguous real block
rather than the stride-2 view of a complex buffer that stays alive behind it. Measured
on the candidate: 1.1306x CI [1.1061, 1.1498] on the metric path, 1.0680x CI [1.0539,
1.0872] end-to-end.

`process` is untouched and stays the decomposition's entry point: that path genuinely
needs the analytic subbands. This is a second method rather than a flag on the first.

The identity is exact, but the evaluation is not bit-identical — `rfft`/`irfft` sum the
same terms in a different order, at the ULP level. Two details are easy to get wrong and
invisible under `allclose`, and `tests/unit/backend_torch/test_torch_gammatone.py` exists
to catch them: `fftfreq`'s Nyquist convention (−0.5) disagrees with `rfftfreq`'s (+0.5),
so the half filter must be built on the first half of the *full* grid rather than on
`rfftfreq`; and the Nyquist bin reflects onto itself, so the conjugate-grid formula does
not apply there and it is set directly.

Both details used to be worth ~1e-7 in that bin, because the frequency grid inherited
torch's default `float32` and `exp(∓iπ)` on it carried an ~8.7e-8 imaginary residue. As
of 2026-08-15 the grid is built at `float64` explicitly, which removes that term: getting
either detail wrong now costs ~1e-16 rather than ~1e-7. That is an accuracy improvement
with a consequence — the test's scalar bar no longer separates the right construction
from the two wrong ones, and replacing it with explicit negative tests is tracked in
`TODO.md`.

Taken together with the kernel fusion above, and measured as a pair against both
changes turned off — `calculate_auditory_quality_features` **1.7466x, 95% CI [1.6608,
1.8574]**, phase estimates 1.8245 and 1.7990, verdict AGREE; the full
`predict_perceptual_evaluation_scores` pipeline **1.2412x, 95% CI [1.2243, 1.2597]**,
phases 1.2430 and 1.2336, AGREE. The two are disjoint stages in series, so their
*savings* should add rather than their ratios multiply; the metric-path interval covers
both models and the phase estimates sit at the additive one, while the pipeline lands
just under both. Deviation against the frozen capture is not additive either, and this
is worth knowing before assuming it is: each change alone moved
`target_perceptual_score` to exactly the same value, and the pair reproduces the
fusion-only result bit-for-bit in all eight scores.

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
expect ~1e-15 on simple mono material.

**Corrected 2026-08-15.** This passage used to close by advising against reaching "for the
`1e-15` diagonal regularization as though every frame needed it". That advice was wrong,
and the conditioning statistics above are why it looked right: they describe the frames
that are *well* conditioned, and the ridge exists for the ones that are not. It is MATLAB
PEASS v2.0.1's own `lambda` in `LSDecompose.m`, transcribed in `reference/LSDecompose.py`
and present in the NumPy backend all along; the torch backend was the only one without it,
and on rank-deficient input that cost real correctness rather than a rounding digit. See
`ARCHIVE.md`, "The 2026-08-15 correctness and `utils.py` pass".

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
- **The `-m oracle` property suite**, which no workflow reselects — by decision, not by
  oversight. It is the only test that checks the backends' output *content* against an
  independent implementation rather than its form, and it runs locally or not at all.

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
(`peass/backend_numpy/gammatone.py:811`, `peass/backend_torch/utils.py:85`). MATLAB's
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
