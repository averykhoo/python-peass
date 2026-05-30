# python-peass

[![Test Suite](https://github.com/averykhoo/python-peass/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/averykhoo/python-peass/actions)
[![PyPI version](https://img.shields.io/pypi/v/python-peass.svg)](https://pypi.org/project/python-peass/)

> This project was ported by Gemini 3.5 Flash from
> https://gitlab.inria.fr/bass-db/peass/-/tree/22c7fc4ef670f8bb6eea9ab4abea98323006b769/v2.0.1

An elegant, Pythonic, and fully-typed port of the **PEASS v2.0.1**
(Perceptual Evaluation methods for Audio Source Separation) toolkit [1].

This library replaces traditional energy ratios (SDR, SIR, SAR) with perceptually motivated objective scores—
**OPS, TPS, IPS, and APS**—which are highly correlated with human evaluations [1].

## Scientific Highlights

Traditional metrics evaluate separations via energy ratios [1].
However, human hearing relies heavily on non-linear auditory transduction and masking [2].
`python-peass` executes a multi-stage cognitive pipeline to assess quality:

1. **Subband Least-Squares Decomposition:**
   Signals are divided into subbands using a complex-valued Hohmann Gammatone Filterbank [1, 3].
   Overlapping temporal frames are projected onto estimated subspaces to isolate physical distortion artifacts [1].
2. **Inner Hair Cell Transduction:**
   Approximates the shearing limits of physical hair bundles via half-wave rectification
   and 1 kHz membrane-limit lowpass filters [1, 2].
3. **Auditory Nerve Adaptation:**
   Uses five stages of non-linear feedback loops modeling forward masking and metabolic neural depletion [2].
4. **Perceptual Assimilation:**
   Models cognitive masking where noise below a target threshold is partially assimilated or masked [2].
5. **Score Prediction:**
   Feeds weighted similarity percentiles into a multi-criteria trained sigmoidal neural network
   to output scores from `0` to `100` [1].

## Architectural Layout & MATLAB Mapping

Unlike loose research scripts, this package organizes functions into explicit scientific modules:

| Python Module          | Primary Classes / Functions                                                                        | Replaced MATLAB / C Files                              |
|:-----------------------|:---------------------------------------------------------------------------------------------------|:-------------------------------------------------------|
| `peass.gammatone`      | `GammatoneFilter`, `GammatoneAnalyzer`, `GammatoneDelay`, `GammatoneMixer`, `GammatoneSynthesizer` | `Gfb_Filter_new.m`, `Gfb_Analyzer_process.m`, etc.     |
| `peass.auditory_model` | `haircell_transduction`, `adaptation_loops`, `generate_internal_representation`                    | `haircell.c` (MEX), `adapt.c` (MEX), `pemo_internal.m` |
| `peass.decomposition`  | `extract_distortion_components`, `least_squares_decompose_time_varying`                            | `extractDistortionComponents.m`, `LSDecompose_tv.m`    |
| `peass.metrics`        | `pemo_similarity_metric`, `calculate_energy_ratios`, `audio_quality_features`                      | `pemo_metric.m`, `audioQualityFeatures.m`              |
| `peass.predictor`      | `predict_peass_scores`, `my_mapping`                                                               | `PEASS_ObjectiveMeasure.m`, `map2SubjScale.m`          |

## Installation

```bash
pip install python-peass
```

## Quick Start Examples

### 1. Perceptual Quality Score Evaluation (Predictor Pipeline)

Evaluate estimated audio files saved on disk:

```python
from peass import predict_peass_scores

original_files = [
    "audio/target_source.wav",
    "audio/interference_1.wav",
    "audio/interference_2.wav"
]
estimate_file = "audio/estimated_target.wav"

scores = predict_peass_scores(original_files, estimate_file)

print(f"Overall Perceptual Score (OPS): {scores['OPS']:.1f}/100")
print(f"Target Preservation Score (TPS): {scores['TPS']:.1f}/100")
print(f"Interference Rejection (IPS):   {scores['IPS']:.1f}/100")
print(f"Artifact-free Score (APS):      {scores['APS']:.1f}/100")
```

### 2. Exposing Physical Decompositions and Wav Files

If you wish to obtain the actual separated signal waveforms
(True target, Target distortion, Interference, and Artifacts) alongside the predicted scores:

```python
from peass import predict_peass_scores

original_files = ["audio/target.wav", "audio/noise.wav"]
estimate_file = "audio/estimate.wav"

# Setting return_decomposition=True triggers wave synthesis
results = predict_peass_scores(
    original_files,
    estimate_file,
    options={'destDir': './output_directory/'},
    return_decomposition=True
)

# 1. Print Scores
print(f"Overall Perceptual Score: {results['OPS']}")

# 2. Access Filepaths of newly generated wave files on disk
print(f"Decomposed Target saved at: {results['decomposition_files']['true_target']}")

# 3. Read raw numpy arrays directly from memory
target_distortion_array = results['decomposition_arrays']['target_distortion']
```

### 3. Running Independent Physical Decomposition (No ML Score Regressor)

You can also run the auditory Gammatone/least-squares decomposition engine by itself
to generate isolated WAV files or raw NumPy arrays:

```python
from peass import extract_distortion_components

# In-memory arrays
target_array = np.random.randn(16000, 1)
noise_array = np.random.randn(16000, 1)
estimate_array = target_array + 0.05 * noise_array

# Run subband least-squares decomposer
_, decomposed_arrays = extract_distortion_components(
    src_files=[target_array, noise_array],
    est_file=estimate_array,
    sampling_frequency=16000.0
)

true_target, target_distortion, interference, artifacts = decomposed_arrays
```

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
