> This project was ported by Gemini 3.5 Flash from
> https://gitlab.inria.fr/bass-db/peass/-/tree/22c7fc4ef670f8bb6eea9ab4abea98323006b769/v2.0.1
> (also in [peass_master_22c7fc4e](./peass_master_22c7fc4e) for the LLM to refer to)

# PEASS Toolkit (Python Port)

This repository contains a native Python implementation of the entire **PEASS (Perceptual Evaluation methods for Audio
Source Separation) toolkit (v2.0.1)** [README].

It performs time-varying subband least-squares error decomposition, computes standard energy ratios, simulates auditory
peripheral processing (PEMO-Q), and uses pre-trained neural networks to map perceptual features directly to predicted
subjective scores (OPS, TPS, IPS, and APS) [README, Doc 1 - p.5, PEASS_ObjectiveMeasure].

---

## 1. Original Source and Credits

The original MATLAB toolkit was authored by **Valentin Emiya, Emmanuel Vincent, Niklas Harlander, and Volker Hohmann** 
(INRIA / University of Oldenburg) [README].

* **Repository Reference:** Distributed via the [INRIA BASS-DB Project](https://gitlab.inria.fr/bass-db/peass) [README].
* **Academic References:**
    * V. Emiya, E. Vincent, N. Harlander, and V. Hohmann, *"Subjective and objective quality assessment of audio source
      separation"*, *IEEE Transactions on Audio, Speech, and Language Processing*, 19(7):2046-2057, 2011 [README].
    * E. Vincent, *"Improved perceptual metrics for the evaluation of audio source separation"*, *10th International
      Conference on Latent Variable Analysis and Signal Separation (LVA/ICA)*, pp.430-437, 2012 [README].

---

## 2. Ported Components

This is a complete translation of all non-compiled processing files in the PEASS pipeline:

### A. Decomposition & Evaluation

* `extractDistortionComponents.py` (Orchestrates time-varying subband decomposition) [extractDistortionComponents].
* `extractTSIA.py` (Decomposes estimate into target, interference, and artifact components) [extractTSIA].
* `LSDecompose_tv.py` (Slices subbands into temporal blocks and applies overlap-add) [LSDecompose_tv].
* `LSDecompose.py` (Applies frame-level weighted least-squares projections) [LSDecompose].
* `erbBW.py` (Computes Equivalent Rectangular Bandwidth) [erbBW].
* `ISR_SIR_SAR_fromNewDecomposition.py` (Calculates decibel ratios directly on decomposed
  arrays) [ISR_SIR_SAR_fromNewDecomposition].

### B. Auditory Periphery & PEMO-Q Modeling

* `pemo_internal.py` (Computes complex subband envelopes, haircell models, and adaptation loops) [pemo_internal].
* `pemo_metric.py` (Applies temporal assimilation and calculates weighted percentile similarity) [pemo_metric].
* `audioQualityFeatures.py` (Coordinates comparisons across multi-channel components) [audioQualityFeatures].
* `myPemoAnalysisFilterBank.py` & `myPemoSynthesisFilterBank.py` (Handles complex envelope modulation, subband
  decimation, and reconstruction) [myPemoAnalysisFilterBank, myPemoSynthesisFilterBank].
* `gammatone_helper.py` (Hohmann 2002 analyzer, synthesizer, delay, and mixer structures) [gammatone_helper].

### C. Subjective Mapping (Machine Learning)

* `map2SubjScale.py` (Handles logarithmic sensory warping and loads model weights) [map2SubjScale].
* `myMapping.py` (Executes the single-output feedforward neural network) [myMapping].
* `PEASS_ObjectiveMeasure.py` (The master script running all steps from end-to-end) [PEASS_ObjectiveMeasure].

---

## 3. Key Python Adaptations and Enhancements

* **No Compiled C/MEX Compiler Requirements:** The original MATLAB toolkit relied on compiled MEX files (`haircell.mex`,
  `adapt.mex`) to compute inner ear sensory representations [compile, pemo_internal]. To achieve an out-of-the-box
  Python environment, **this port translates the native MATLAB fallback loop equations** built into
  `pemo_internal.m` [pemo_internal]. You do not need GCC, Clang, or C-compiling setups to run the auditory periphery
  simulations [pemo_internal].
* **In-Memory NumPy Operations:** While the original MATLAB pipeline operated purely on `.wav` disk
  paths [extractDistortionComponents], the Python port accepts **both** file paths and direct in-memory NumPy arrays 
  (shape `L x Channels`), making it compatible with modern deep-learning validation loops (e.g., PyTorch, JAX).
* **Anti-Aliasing Polyphase Resampling:** The MATLAB `resample` function was replaced with *
  *`scipy.signal.resample_poly`** [myPemoAnalysisFilterBank, myPemoSynthesisFilterBank]. Array-length clipping and
  padding filters are implemented to prevent index out-of-bound errors during multi-channel subband OLA
  reconstruction [myPemoSynthesisFilterBank].
* **Dynamic Weight Loading and Index Conversion:** The neural network weights are loaded directly from the original
  MATLAB binary files (`paramTask1.mat` through `paramTask4.mat`) using `scipy.io.loadmat` [map2SubjScale]. The program
  automatically maps MATLAB's 1-based index feature arrays (`selec`) to 0-based Python indexing [map2SubjScale].

---

## 4. Requirements and Setup

Before executing, install the standard Python libraries:

```bash
pip install numpy scipy soundfile
```

### Critical Dependency:

To run the full objective evaluation (with PEMO-Q and neural network mapping),
you must place the original binary parameter files in your working directory [map2SubjScale]:

* `paramTask1.mat` [map2SubjScale]
* `paramTask2.mat` [map2SubjScale]
* `paramTask3.mat` [map2SubjScale]
* `paramTask4.mat` [map2SubjScale]

*(If these files are missing, you can still perform the physical signal decomposition
and retrieve the SDR, SIR, SAR, and ISR decibel ratios) [PEASS_ObjectiveMeasure].*

---

## 5. Quick Start Example

You can run the entire PEASS evaluation pipeline in Python using the master script [PEASS_ObjectiveMeasure]:

```python
import numpy as np
from peass_direct_python_port.PEASS_ObjectiveMeasure import PEASS_ObjectiveMeasure

# Set your sampling frequency
fs = 16000
duration = 2.0  # seconds
t = np.linspace(0, duration, int(duration * fs), endpoint=False)

# 1. Prepare clean speech (Target) and reference noise (Interference)
speech = np.sin(2 * np.pi * 440 * t)[:, np.newaxis]  # 440Hz vocal tone
noise = 0.5 * np.random.randn(len(t))[:, np.newaxis]

# 2. Prepare processed output (Simulated denoiser output)
denoised_output = 0.9 * speech + 0.05 * noise + 0.01 * np.random.randn(len(t))[:, np.newaxis]

print("Computing complete PEASS evaluation...")

# 3. Execute the full PEASS master pipeline
results = PEASS_ObjectiveMeasure(
  originalFiles=[speech, noise],
  estimateFile=denoised_output,
  fs=fs
)

# 4. Display objective ratios and predicted subjective ratings
print("\n--- Physical Ratios ---")
print(f"SDR: {results['SDR']:.2f} dB")
print(f"SIR: {results['SIR']:.2f} dB")
print(f"SAR: {results['SAR']:.2f} dB")

print("\n--- Predicted Subjective Scores (0 - 100) ---")
print(f"OPS (Overall Quality):                 {results['OPS']:.1f}")
print(f"TPS (Target/Speech Preservation):      {results['TPS']:.1f}")
print(f"IPS (Interference/Noise Suppression):  {results['IPS']:.1f}")
print(f"APS (Artifact Absence):                {results['APS']:.1f}")
```
