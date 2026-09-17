# The PEASS Toolkit - Perceptual Evaluation methods for Audio Source Separation

Emiya, Vincent, Harlander & Hohmann (2010). LVA/ICA 2010 CONFERENCE POSTER. HAL inria-00545477. Page 1 is the HAL cover sheet; page 2 is the poster.

> **What this file is.** A transcription of `papers/2010-09-27_LVA_PEASS_emiya.pdf`, made on 2026-09-17 by reading
> every page image and typing out what is printed there. It replaces a mechanical `pdftotext`
> extraction that was unusable for mathematics. Each page was transcribed and then independently
> re-checked against the same page image by a second reader. On 2026-09-17 every page was
> then read once more against its own render, one reader per page, and each defect that
> reader claimed was re-measured independently before anything was changed; see
> `papers/README.md` for what that pass found.
>
> **Conventions.**
> - `<!-- page N -->` marks the start of PDF page N. All 2 pages are present.
> - `[Figure N: ...]` is the caption **as printed**. Any prose that follows it describes what the
>   plot shows and is **not printed on the page** - it is a reader's summary, and any values in it
>   were read off a plot by eye. Do not cite those numbers; read the figure in the PDF.
> - `[?]` marks a glyph that was genuinely illegible. It is never a guess.
> - Mathematics is LaTeX. Equation numbers are preserved as printed.
>
> **For anything that must be exact** - a coefficient you will implement, a threshold you will
> cite - check it against the PDF. This is a careful transcription, not a facsimile.

---

<!-- page 1 -->

[Image: HAL open science logo]

# The PEASS Toolkit - Perceptual Evaluation methods for Audio Source Separation

Valentin Emiya, Emmanuel Vincent, Niklas Harlander, Volker Hohmann

**► To cite this version:**

> Valentin Emiya, Emmanuel Vincent, Niklas Harlander, Volker Hohmann. The PEASS Toolkit - Perceptual
> Evaluation methods for Audio Source Separation. 9th Int. Conf. on Latent Variable Analysis and Signal
> Separation, Sep 2010, Saint-Malo, France. 2010. ⟨inria-00545477⟩

**HAL Id: inria-00545477**

**https://inria.hal.science/inria-00545477v1**

Submitted on 7 Dec 2011

**HAL** is a multi-disciplinary open access archive for the deposit and dissemination of scientific research documents, whether they are published or not. The documents may come from teaching and research institutions in France or abroad, or from public or private research centers.

L'archive ouverte pluridisciplinaire **HAL**, est destinée au dépôt et à la diffusion de documents scientifiques de niveau recherche, publiés ou non, émanant des établissements d'enseignement et de recherche français ou étrangers, des laboratoires publics ou privés.

[Image: "Autorisation HAL" stamp]

HAL Authorization

<!-- page 2 -->

# The PEASS Toolkit - Perceptual Evaluation methods for Audio Source Separation

Valentin EMIYA, Emmanuel VINCENT • METISS, INRIA Rennes - Bretagne Atlantique (France)
Niklas HARLANDER, Volker HOHMANN • Carl von Ossietzky Universität Oldenburg (Germany)

## THE PEASS TOOLKIT: overview

A toolkit for the perceptual evaluation of audio source separation

- The PEASS Software: a set of objective measures to predict the perceptual quality of the source/image estimates
- The PEASS Listening Test GUI: a Matlab MUSHRA GUI realized for the proposed test protocol
- The PEASS Subjective Database: a set of subjective measures resulting from listening tests (20 subjects × 80 sounds × 4 rating criteria)

**The PEASS Toolkit is freely available at**
**http://bass-db.gforge.inria.fr/peass/**

[Illustration: at the top right of the panel, a hexagon encloses the source instruments — a double bass ($s_1$), a grand piano ($s_2$) and a drum kit ($s_3$); a pair of microphones and a CD sit to its right, and dashed arrows point to the corresponding separated image estimates ($\hat{s}_1$, $\hat{s}_2$, $\hat{s}_3$), drawn as greyed/blurred copies of the same double bass, piano and drum kit.]

[1] **Subjective and objective quality assessment of audio source separation,** *V. Emiya, E. Vincent, N. Harlander, V. Hohmann,* IEEE Trans. on Audio, Speech and Language Processing, submitted, 2010.

[2] **Multi-criteria subjective and objective evaluation of audio source separation,** *V. Emiya, E. Vincent, N. Harlander, V. Hohmann,* AES 38th Int. Conf. on Sound Quality Evaluation, Pitea, Sweden, June 2010.

## MOTIVATION: the need for a multi-criteria perceptually-based evaluation

**Existing model for distortion decomposition [3]:**

$$\hat{s}_j(t) - s_j(t) = e_j^{\text{target}}(t) + e_j^{\text{interf}}(t) + e_j^{\text{artif}}(t) \tag{1}$$

- $e_j^{\text{target}}$ denotes the error component related to the target distortion,
- $e_j^{\text{interf}}$ denotes the interference from concurrent sources,
- $e_j^{\text{artif}}$ is the remaining distortion component (artifacts and noise).

Defining and estimating the distortion components $e_j^{\text{target}}$, $e_j^{\text{interf}}$, $e_j^{\text{artif}}$ is not trivial. Due to the allowed distortions in use today (time-invariant spatial and filtering distortions), the decomposition is not satisfying.

**Existing quality measures:** energy ratios SDR, ISR, SIR, SAR are poorly correlated with subjective scores.

**Proposed multi-criteria listening test protocol**

A series of 4 MUSHRA tests including several dedicated anchors:

($T_1$) Rate the *global quality* compared to the reference.
($T_2$) Rate the quality in terms of *preservation of the target source*.
($T_3$) Rate the quality in terms of *suppression of other sources*.
($T_4$) Rate the quality in terms of *absence of additional artificial* noise.

## PROPOSED OBJECTIVE MEASURES: what's new?

A **better distortion decomposition** is achieved by:

- splitting the signals into subbands using gammatone filters;
- segmenting each subband signal into overlapping frames;
- decomposing each frame into distortion components using a matched FIR filter
- reconstructing the full distortion components

Some **auditory-motivated features** are derived using PEMO-Q/PSM [4]:

$$q_j^{\text{overall}} \triangleq \text{PSM}(\hat{\mathbf{s}}_j, \mathbf{s}_j) \tag{2}$$

$$q_j^{\text{target}} \triangleq \text{PSM}(\hat{\mathbf{s}}_j, \hat{\mathbf{s}}_j - \mathbf{e}_j^{\text{target}}) \tag{3}$$

$$q_j^{\text{interf}} \triangleq \text{PSM}(\hat{\mathbf{s}}_j, \hat{\mathbf{s}}_j - \mathbf{e}_j^{\text{interf}}) \tag{4}$$

$$q_j^{\text{artif}} \triangleq \text{PSM}(\hat{\mathbf{s}}_j, \hat{\mathbf{s}}_j - \mathbf{e}_j^{\text{artif}}) \tag{5}$$

By combining the 4 features in a non-linear way to predict subjective scores $(T_1) - (T_4)$, a set of objective measures is finally output:

- **OPS:** the Overall Perceptual Score,
- **TPS:** the Target-related Perceptual Score,
- **IPS:** the Interference-related Perceptual Score,
- **APS:** the Artifacts-related Perceptual Score.

## EVALUATION RESULTS: prediction performance and evaluation at SiSEC 2010

Legend for the curves below (printed in a box above the Task 1 plots): - - - Old+SxR &nbsp;&nbsp; ─·─ New+SxR &nbsp;&nbsp; ······ Old+PSM &nbsp;&nbsp; ─── New+PSM

**Task 1: prediction of global quality** (rotated y-axis label) — three plots (Accuracy, Monotonicity, Consistency), y-axis ticks 0, 0.2, 0.4, 0.6, 0.8, 1, x-axis "Feature vector size" with tick marks at 1, 3, 4.

[Figure: Prediction results (cross-validation on the PEASS database) for the 4 tasks: curves are various combinations of the old/new decompositions with the energy ratio/PEMO-Q measures, as a function of the number of features.] A grid of small multiples: rows Task 2 (x ticks 1,2,3,4), Task 3 (x ticks 1,3,4) and Task 4 (x ticks 1,2,3,4), columns Accuracy, Monotonicity, Consistency; each subplot, like the three large Task-1 plots to its left, shows four curves — dashed (Old+SxR), dash-dot (New+SxR), dotted (Old+PSM) and solid (New+PSM) — plus a thick grey horizontal reference line near the top of the axes; y-axis ticks 0, 0.5, 1, and the shared x-axis label is again "Feature vector size".

[Figure: BSS eval vs. PEASS: scatter plots of the SiSEC 2010 results for the set of *Professionally produced music recordings*.] A rotated label "DEMO SiSEC 2010" runs up the left-hand side of this block. Four scatter plots of energy-ratio measures against the corresponding PEASS perceptual scores, each with a cloud of small filled blue dots and a few highlighted, labelled examples whose points are joined by coloured lines — three points into a triangle for Ex.1, Ex.3 and Ex.4, four into a quadrilateral for Ex.2, and only two, joined by a single near-flat segment at about SAR = 0 dB, for Ex.5:
- SDR (dB), ticks −5 to 15, vs. OPS (0–100), titled "Accuracy = 0.23; Monotonicity = 0.21;": ○ Ex.1 (red, open circle), ◇ Ex.2 (green, open diamond).
- ISR (dB), ticks −10 to 30, vs. TPS (0–100), titled "Accuracy = 0.29; Monotonicity = 0.31;": ◇ Ex.2 (green, open diamond), □ Ex.3 (magenta, open square).
- SIR (dB), ticks −10 to 30, vs. IPS (0–100), titled "Accuracy = 0.64; Monotonicity = 0.64;": ○ Ex.4 (black, open circle).
- SAR (dB), ticks −30 to 20, vs. APS (0–100), titled "Accuracy = 0.16; Monotonicity = 0.34;": ◇ Ex.5 (cyan, open diamond), ○ Ex.1 (red, open circle).

[3] E. Vincent, R. Gribonval, C. Févotte, *Performance measurement in blind audio source separation*, IEEE Trans. on Acoustics, Speech and Signal Proc., 14 (4), 2006.

[4] R. Huber, B. Kollmeier, *PEMO-Q – A New Method for Objective Audio Quality Assessment Using a Model of Auditory Perception*, IEEE Trans. on Acoustics, Speech and Signal Proc., 14 (6), 2006.

---

9th Int. Conf. on Latent Variable Analysis and Signal Separation, September 27-30, 2010, St. Malo, France

[Image: INRIA logo] [Image: Carl von Ossietzky Universität Oldenburg logo]
