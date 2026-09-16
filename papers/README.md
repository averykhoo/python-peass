# Papers

The primary sources for this port. `text/` holds a plain-text copy of each PDF so
that the sources can be grepped and read by tools that cannot open a PDF.

**Read `text/*.txt` for prose. Open the PDF for equations, tables and figures.**

## What is here

| file | document | what it backs |
|---|---|---|
| `emiya2011.pdf` | Emiya, Vincent, Harlander & Hohmann (2011), "Subjective and objective quality assessment of audio source separation", *IEEE TASLP* 19(7):2046–2057. [doi:10.1109/TASL.2011.2109381](https://doi.org/10.1109/TASL.2011.2109381), HAL inria-00567152 | the reference description of PEASS itself; `CODE_REVIEW.md` treats it as ground truth alongside the MATLAB v2.0.1 source |
| `2010-09-27_LVA_PEASS_emiya.pdf` | Emiya, Vincent, Harlander & Hohmann (2010), "The PEASS Toolkit — Perceptual Evaluation methods for Audio Source Separation", *LVA/ICA 2010*. HAL inria-00545477 | the toolkit's first description; a 4-page precursor to the above |
| `vincent_LVA12.pdf` | Vincent (2012), "Improved perceptual metrics for the evaluation of audio source separation", *LVA/ICA 2012*, pp. 430–437. HAL hal-00653196 | the revised metrics, i.e. what PEASS v2 computes |
| `622101.pdf` | Dau, Püschel & Kohlrausch (1996), "A quantitative model of the 'effective' signal processing in the auditory system. I. Model structure", *JASA* 99(6):3615–3622. [doi:10.1121/1.414959](https://doi.org/10.1121/1.414959) | `peass/backend_numpy/auditory_model.py`, the PEMO ear model |
| `621984.pdf` | Dau, Kollmeier & Kohlrausch (1997), "Modeling auditory processing of amplitude modulation I. Detection and masking with narrow-band carriers", *JASA* 102(5):2892–2905. [doi:10.1121/1.420344](https://doi.org/10.1121/1.420344) | the modulation filterbank in the same module |
| `706.pdf` | Dau (1996), *Modeling auditory processing of amplitude modulation*, PhD thesis, Universität Oldenburg (BIS Verlag). 133 pp. | the long-form derivation behind both JASA papers |

Not here: Hohmann (2002), which the gammatone filterbank in
`peass/backend_numpy/gammatone.py` is ported from and which `ARCHIVE.md` cites by
equation number. Everything in this folder is either PEASS or the Dau ear model.

## Regenerating `text/`

```
python scripts/extract_text.py            # rewrite every .txt
python scripts/extract_text.py --check    # audit what is on disk, extract nothing
```

Deterministic: a re-run reproduces all six files byte for byte, so a diff means a
real change. Needs `pdftotext` (poppler); Git for Windows ships it.

This is text extraction, not OCR — every one of the six PDFs is digital-born and
carries a real text layer, so nothing renders page images or calls a vision model.

## How far to trust it

Verified 2026-09-16, against the output the script produces today.

**Prose is sound in all six.** Extraction runs in `-raw` mode, which emits glyphs
in content-stream order — i.e. reading order, one column through and then the
next. The obvious-looking `-layout` is wrong for every document here: it rebuilds
the printed page geometrically, so both columns land side by side on each output
line and the gutter cuts every sentence in half. Measured on these six, that ruins
8.9%–67.5% of body lines; `-raw` scores 0.0%. The script fails if any file regresses
above 2%.

**`706.txt` needed repair, the other five did not.** The thesis is a pre-Unicode
TeX document that leaks its OT1 font slots into the output as control characters,
so `fl` arrives as a carriage return, `fi` as a form feed, and the dieresis of
"Göttingen" as a separate character two places to the left of its vowel. The
script maps them back, and `scripts/extract_text.py` documents each rule and the
evidence for it. After repair: 0 stray control characters, and "effective",
"filterbank", "reflects", "Göttingen", "Püschel", "großes" and "roll-off" all read
and grep correctly.

**Equations and plot legends in `706.txt` are an approximation.** Those same OT1
slots are reused by TeX's maths fonts, and pdftotext discards the font that would
tell them apart — character 14 is `ffi` in "difficult", delta in an equation and a
degree sign in "360°". Each slot is resolved by the rule its own occurrences
support, which is right for prose and best-effort inside maths. Where the document
does not settle it at all, the text carries a visible `[?N]` marker rather than a
plausible guess. There are 28: nine are plot-legend marker glyphs ("Carrier
bandwidth: `[?7]`: 3 Hz, N: 31 Hz"), six decorate the μ and σ of one appendix
formula, and thirteen are a single variable in the modulation-depth formulas of
chapter 2. Superscripts,
subscripts and fractions are flattened regardless, as they are in any text
extraction. **Read equations from `706.pdf`.**

Tables and figures lose their alignment everywhere, which is the price of reading
order and costs nothing that was legible in plain text to begin with.
