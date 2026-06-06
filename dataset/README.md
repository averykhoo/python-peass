# PEASS Subjective Evaluation Dataset

This folder houses the training data from the Emiya et al. subjective evaluation database used to calibrate overall and specific perceptual evaluations (OPS, TPS, IPS, APS) [1].

## Contents

*   `PEASS-subjdata.mat`: Human subject scores across 10 mixtures and 8 evaluated signals.
*   `audio/`: Complete set of original sources, interferences, and separation estimates.

## Reading the Data Structure with Python

You can read this dataset programmatically using Python and SciPy:

```python
import scipy.io as sio
import numpy as np

# Load subjective database
mat_data = sio.loadmat('PEASS-subjdata.mat')

# variables inside mat_data:
# 1. 'scores': Array of shape (320, 20).
#    Contains 320 sound evaluations rated by 20 reliable subjects.
scores = mat_data['scores']

# 2. 'soundNames': Cell array of shape (320, 1) with strings of evaluated audio file names.
#    Extract to standard Python strings:
names = [n[0][0] for n in mat_data['soundNames']]

# 3. Boolean masks of shape (320, 1) mapping each index to a specific evaluation category:
#    * IGlobalScore -> Overall Perceptual Score (OPS)
#    * ITargetPreservationScore -> Target Preservation Score (TPS)
#    * IOtherSourceSuppressionScore -> Interference Rejection Score (IPS)
#    * IArtificialNoiseAbsenceScore -> Artifact-free Score (APS)
#    Each mask has exactly 80 assertions (10 mixtures * 8 evaluated sounds = 80 per task).
mask_ops = mat_data['IGlobalScore'].ravel() == 1
ops_scores = scores[mask_ops, :] # Mean of axis 1 yields final target OPS values
```


---



### 4. Anatomy of `PEASS-subjdata.mat`

Because you are discarding the MATLAB parser scripts, here is exactly what is stored inside `PEASS-subjdata.mat` and how it maps to your inputs:

#### The Variables
1.  **`scores`** *(Shape: `(320, 20)`)*:
    This contains the actual grades assigned by the 20 subjects [1]. Each row represents a rated sound file, and each column is a subject's rating (on a scale from $0$ to $100$).
2.  **`soundNames`** *(Shape: `(320, 1)`)*:
    The filenames corresponding to each row of the `scores` array (e.g., `'exp01_test5'`, `'exp01_anchorArtif'`).
3.  **Boolean Category Masks** *(Shape: `(320, 1)`)*:
    These matrices contain indices of `1` (True) and `0` (False) to identify which of the 320 scores correspond to which evaluation task (each task comprises exactly 80 sounds):
    *   **`IGlobalScore`**: Masks elements evaluating global separation quality (**OPS**).
    *   **`ITargetPreservationScore`**: Masks elements evaluating target signal preservation (**TPS**).
    *   **`IOtherSourceSuppressionScore`**: Masks elements evaluating interferer rejection (**IPS**).
    *   **`IArtificialNoiseAbsenceScore`**: Masks elements evaluating the level of processed artifacts (**APS**).

#### How the 320 sounds are calculated
The database runs 10 evaluation trials (mixtures `exp01` to `exp10`). In each trial, 8 audio files are evaluated by subjects [1]:
*   **Sound 1**: Aligned reference target (`_target.wav`).
*   **Sound 2**: Processing Artifact anchor (`_anchorArtif.wav`).
*   **Sound 3**: Background leakage/Interference anchor (`_anchorInterf.wav`).
*   **Sound 4**: Target distortion anchor (`_anchorDistTarget.wav`).
*   **Sound 5-8**: Separation estimates output by separation models (`_test5.wav` through `_test8.wav`).

$10\text{ mixtures} \times 8\text{ audio files} \times 4\text{ evaluation categories} = 320\text{ total test files}$ evaluated in the dataset [1].

---

### References
[1] V. Emiya, E. Vincent, N. Harlander, and V. Hohmann, *"Subjective and objective quality assessment of audio source separation"*, IEEE Transactions on Audio, Speech, and Language Processing, 19(7):2046–2057, 2011.