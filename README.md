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

The validation suite implements rigorous numerical, physical, and integration checks.
You will need to `pip install -r requirements.txt` to set up.

### Regression Verification

We test our output waveforms directly against the original `.wav` reference waveforms generated by the official MATLAB
PEASS toolbox (located in `references/peass_master_22c7fc4e/v2.0.1/example/`).
Python's outputs must achieve a cross-correlation coefficient exceeding $0.95$ with the MATLAB reference to pass.

### Parallel Execution

To run the test suite across multiple CPU cores using `pytest-xdist`, execute:

```bash
pytest -n auto
```

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
