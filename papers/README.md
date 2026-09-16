# Papers

The primary sources for this port, and a readable transcription of each under
`text/`.

**Read `text/*.md` to follow an argument or grep for a term. Open the PDF before
you rely on any single number, coefficient or equation.**

## What is here

| file | document | what it backs |
|---|---|---|
| `emiya2011.pdf` | Emiya, Vincent, Harlander & Hohmann (2011), "Subjective and objective quality assessment of audio source separation", *IEEE TASLP* 19(7):2046–2057. [doi:10.1109/TASL.2011.2109381](https://doi.org/10.1109/TASL.2011.2109381), HAL inria-00567152 | the reference description of PEASS; `CODE_REVIEW.md` treats it as ground truth alongside the MATLAB v2.0.1 source |
| `vincent_LVA12.pdf` | Vincent (2012), "Improved perceptual metrics for the evaluation of audio source separation", *LVA/ICA 2012*, pp. 430–437. HAL hal-00653196 | the revised metrics, i.e. what PEASS v2 computes |
| `2010-09-27_LVA_PEASS_emiya.pdf` | Emiya, Vincent, Harlander & Hohmann (2010), "The PEASS Toolkit", *LVA/ICA 2010*. HAL inria-00545477 | **a conference poster, not a paper** — 2 PDF pages, of which page 1 is the HAL cover sheet and page 2 is the poster itself. The earliest public description of the toolkit |
| `622101.pdf` | Dau, Püschel & Kohlrausch (1996), "A quantitative model of the 'effective' signal processing in the auditory system. I. Model structure", *JASA* 99(6):3615–3622. [doi:10.1121/1.414959](https://doi.org/10.1121/1.414959) | `peass/backend_numpy/auditory_model.py`, the PEMO ear model |
| `621984.pdf` | Dau, Kollmeier & Kohlrausch (1997), "Modeling auditory processing of amplitude modulation I. Detection and masking with narrow-band carriers", *JASA* 102(5, Pt. 1):2892–2905. [doi:10.1121/1.420344](https://doi.org/10.1121/1.420344) | the modulation filterbank in the same module |
| `706.pdf` | Dau (1996), *Modeling auditory processing of amplitude modulation*, PhD thesis, Universität Oldenburg (BIS Verlag). 133 pp. | the long-form derivation behind both JASA papers |

Not here: Hohmann (2002), which `peass/backend_numpy/gammatone.py` is ported from
and which `ARCHIVE.md` cites by equation number. Everything in this folder is
either PEASS or the Dau ear model.

## The transcriptions

`text/<name>.md` is a transcription of the matching PDF: 181 pages across the six
documents, produced on 2026-09-17 by reading every page image and typing out what
is printed there. Mathematics is LaTeX, so the files render and grep.

Conventions, identical in every file:

- `<!-- page N -->` marks the start of PDF page N. Every page is present, in order.
- `[Figure N: ...]` is the caption **as printed on the page**. Prose following a
  caption describes the plot and is **not printed anywhere** — it is a reader's
  summary, and any values in it were estimated off a plot by eye. Never cite those
  numbers; read the figure in the PDF.
- `[?]` marks a glyph that was genuinely illegible. It is never a guess.

`scripts/check_transcriptions.py` validates what a tool can validate: that the page
markers still cover each PDF exactly, that no artefact of the old broken extraction
has crept back in, and that no LaTeX delimiter is left open. It cannot tell you the
words are right. Run it after editing any of these files.

## Why these are transcribed and not extracted

The obvious approach — `pdftotext` — was tried first, shipped, and then withdrawn,
because it cannot represent these particular PDFs:

- **`706.pdf` embeds Type 3 fonts with custom encodings and no ToUnicode map.**
  There is simply no mapping from its glyph codes to characters, and every
  extractor guesses the same way: minus signs vanished entirely, `∞` came out as
  `1`, `′` as `0`, `∫` as `Z`, `Σ` as `X`, `√` as `p`, `·` as `Δ`, the conditional
  bar in `P(e|S)` as `j`, and `≈` as `π`. Prose survived; **not one** of the 18
  appendix equations did.
- **The two JASA PDFs carry a broken ToUnicode map.** Operators arrived as Coptic
  letters (`ϭ` for `=`, `Ϫ` for `−`, `Ϸ` for `≈`), parentheses as combining marks,
  `√` as `ͱ`, and every umlaut was split from its vowel across a line break. One
  whole display equation — (A5) in `622101` — was dropped silently.
- **`emiya2011` and `vincent_LVA12` extract cleanly except for one thing:** the hat
  on the estimated signal. `ŝ` became `s`, so eq. (1) read `s(t) − s(t) = …` and the
  metric definitions compared a signal to itself.

None of this is fixable by post-processing, because the information needed to undo
it is not in the file. Reading the rendered page is the only way to recover it.

## How far to trust this

Verified 2026-09-17.

Each page was transcribed, then independently re-checked against the same page
image by a second reader, who repaired 112 defects. A third, independent pass then
audited all 181 pages adversarially, looking specifically for invented content:

- **217 equations checked, 217 correct.**
- No wrong digits or symbols above trivial severity, in equations, tables, figure
  captions or reference lists.
- No omissions beyond cosmetic ones (a trailing period, an italic lost).
- 31 of 38 page-ranges judged faithful outright, 7 with minor issues, **none serious**.

What that audit *did* find was concentrated in one place: the figure descriptions.
Seven of the 76 described a plot inaccurately — a curve said to converge that does
not, an intersection put at the wrong x value, two curves said to run together where
the paper's own text remarks on their separation. All seven have been corrected, and
two passages that asserted things the page does not say were removed outright. This
is why the descriptions carry the warning they do: they are the one part of these
files not anchored to printed text, and they were measurably the least reliable part.

Everything above is a statement about a careful transcription, not a facsimile. For
a coefficient you are about to implement or a threshold you are about to cite, check
the PDF.
