# Transcription notes — `reference/gammatone/`

The gammatone toolbox bundled with MATLAB PEASS v2.0.1 (`v2.0.1/gammatone/`),
transcribed one Python module per `.m` file per FORMAT.md. It is third-party
code (AG Medizinische Physik, Universität Oldenburg; Hohmann 2002, Herzke &
Hohmann 2007) that PEASS vendors verbatim, so it is kept self-contained: these
modules import numpy, scipy and the stdlib and nothing else — not `peass`, and
not `reference/_matlab_runtime.py` either. The handful of MATLAB idioms they
need (`round`, struct field access) are spelled out locally instead.

## Scope

Transcribed: the 17 functions PEASS's filterbank path reaches, plus the
constants script they all depend on.

| MATLAB | module |
| --- | --- |
| `Gfb_set_constants.m` | `gfb_set_constants.py` |
| `Gfb_erbscale2hz.m` / `Gfb_hz2erbscale.m` | `gfb_erbscale2hz.py` / `gfb_hz2erbscale.py` |
| `Gfb_center_frequencies.m` | `gfb_center_frequencies.py` |
| `Gfb_Filter_{new,process,clear_state,zresponse}.m` | `gfb_filter_*.py` |
| `Gfb_Analyzer_{new,process,clear_state,zresponse}.m` | `gfb_analyzer_*.py` |
| `Gfb_Delay_{new,process}.m` | `gfb_delay_*.py` |
| `Gfb_Mixer_{new,process}.m` | `gfb_mixer_*.py` |
| `Gfb_Synthesizer_{new,process}.m` | `gfb_synthesizer_*.py` |

`Gfb_set_constants.m` is included because six of the seventeen depend on it:
`Gfb_erbscale2hz`, `Gfb_hz2erbscale`, `Gfb_Filter_new` and `Gfb_Analyzer_new`
read `GFB_L`, `GFB_Q` and `GFB_PREFERED_GAMMA_ORDER` from it, and
`Gfb_Mixer_new` reads `GFB_GAINCALC_ITERATIONS`.

Not transcribed, because nothing in the set above calls them:
`Gfb_Delay_clear_state.m`, `Gfb_Synthesizer_clear_state.m`, `Gfb_plot.m`,
`Example_*.m`, and the MEX sources `Gfb_analyze.c` / `Gfb_Analyzer_fprocess.c`.

## Deviations

Two, both marked `# !!! DEVIATION` in the source.

1. **`gfb_analyzer_process.py` — the `analyzer.fast` MEX path.**
   `Gfb_Analyzer_process` branches on `analyzer.fast` and, when set, delegates
   to the compiled `Gfb_Analyzer_fprocess` MEX. There is no MEX here, so `fast`
   is hard-wired to false and the pure-MATLAB `else` branch is always taken.
   This matters because PEASS *does* set it: `myPemoAnalysisFilterBank.m` says
   `analyzer.fast = true`. The two branches are meant to be numerically the
   same computation — that is the stated reason `Gfb_Filter_process`
   pre-multiplies and post-divides its filter state by the filter coefficient,
   "for compatibility of the filter state with the MEX extension" — so taking
   the slow branch is the intended fallback, not a different algorithm.

2. **`gfb_center_frequencies.py` — MATLAB's colon operator.**
   `[start:(1/filters_per_ERBaud):upper]` is not `np.arange`. MATLAB chooses
   the element count by rounding `(limit-base)/step` to the nearest integer and
   stepping back only if that overshoots the limit; `np.arange` compares
   against the limit directly and so can drop or gain a final element under
   roundoff — one band more or fewer. The round-then-adjust rule is reproduced
   explicitly. MATLAB's builtin additionally refines the values from both ends
   of the range, which this port does not; that refinement can only move a
   value by an ulp, and with the PEASS parameters (`filters_per_ERB = 1.0`, so
   `step = 1`) it is exact anyway.

## Idioms interpreted rather than transliterated

* **Structs.** `Gfb_Filter`, `Gfb_Analyzer`, `Gfb_Delay`, `Gfb_Mixer` and
  `Gfb_Synthesizer` are small classes defined in the corresponding `*_new`
  module, carrying the MATLAB field names verbatim. MATLAB structs are
  *values*: passing one to a function copies it, and only the copy the function
  returns carries the change. Python objects are references, so every function
  that returns a modified struct starts with `x = x.copy()` (a deepcopy). This
  is what makes the MATLAB calling convention — `[output, analyzer] =
  Gfb_Analyzer_process(analyzer, input)`, kept here as a returned tuple —
  behave the way callers expect; the caller's analyzer keeps its old state
  until it rebinds the name. Attribute-style fields also let PEASS's
  `mstruct_get` / `mstruct_set` add `analyzer.fsOrig`, `analyzer.Ndec`, etc.
* **Struct arrays.** `analyzer.filters` is a MATLAB 1×N struct array, grown by
  assigning to `analyzer.filters(1,band)` and read back as either
  `analyzer.filters(band)` (linear) or `analyzer.filters(1,band)` (subscript).
  Both address the same element, and both become `analyzer.filters[band - 1]`
  on a Python list.
* **`Gfb_set_constants` is a script, not a function.** Callers declare `global
  GFB_L` and then run the script, which assigns the globals. Here the constants
  are module-level names and `Gfb_set_constants()` re-assigns them, which is
  the same effect; callers read them back through the module object
  (`gfb_set_constants.GFB_L`) so they always see the current value, the way a
  MATLAB `global` declaration does. The script is also run once at import.
* **Row and column vectors are both 1-D arrays.** MATLAB distinguishes 1×N from
  N×1; numpy 1-D arrays carry no orientation. So `z(:)` in
  `Gfb_Analyzer_zresponse` is `reshape(-1)`, `center_frequencies(:)` in
  `Gfb_Mixer_new` likewise, and `mixer.gains = mixer.gains.'` is a no-op with a
  comment. Where MATLAB's `*` is a matrix product on those vectors —
  `f_response * mixer.gains`, `mixer.gains * input` — the Python uses `@`.
  Genuinely 2-D quantities (`zresponse`, the band × sample matrices) stay 2-D.
* **Preallocation dtype.** MATLAB's `zeros`/`ones` are real and are silently
  promoted when something complex is assigned into them. numpy does not
  promote, it truncates, so `Gfb_Analyzer_process`, `Gfb_Analyzer_zresponse`
  and `Gfb_Delay_new` allocate `dtype=complex` where MATLAB relies on
  promotion. `Gfb_Delay_new`'s `delay.memory` and `Gfb_Delay_process`'s
  `output` stay real, because MATLAB only ever stores `real(...)` into them.
* **`length()` is `max(size())`,** not `size(...,1)`. `Gfb_Analyzer_process`
  sizes its output with `max(np.shape(input))` for that reason.
* **`lambda` is a Python keyword,** so the MATLAB variable in
  `Gfb_Filter_new` is spelled `lambda_`. Nothing else about it changes. The
  other awkward MATLAB names are kept as-is, including the ones that shadow
  Python builtins (`input`, `filter`) — `filter` even shadows the MATLAB
  builtin `filter` inside `Gfb_Analyzer_zresponse`, which is why
  `Gfb_Filter_process` names its struct `filter_obj`.
* **`nargin`** is reconstructed from `None` defaults (`nargin = 2` plus the
  number of trailing arguments actually supplied, etc.). This works because
  every optional MATLAB argument here is positional and trailing.
* **Integer conversions.** `math.factorial` needs an `int` where MATLAB's
  `factorial` takes a double, and `np.zeros` needs an integer length where
  MATLAB's `zeros` accepts `4.0`. Both are wrapped in `int(...)`.
* **`round`** in `Gfb_Synthesizer_new` is MATLAB's — half away from zero.
  Python's `round` and `np.round` do banker's rounding, so it is written out as
  `sign(x) * floor(abs(x) + 0.5)`.
* **`filter` → `scipy.signal.lfilter`.** The filter coefficients here are
  complex (`lambda * exp(i*beta)`). `lfilter` handles complex `b`, `a` and `zi`
  and uses the same transposed-direct-form-II state as MATLAB's `filter`, so
  the returned `zf` is MATLAB's final-condition output element for element;
  verified against a hand-written scalar recurrence and against the gold values
  below. Each of the `gamma_order` stages is a first-order section, so `zi` is
  a length-1 array — MATLAB's scalar `filter_state(i)`.

## Judgement calls worth knowing about

* **`Gfb_Mixer_new` tests `nargin < 4` in a three-argument function.** The
  effect is that `iterations` is *always* overwritten by
  `GFB_GAINCALC_ITERATIONS` (100), even when the caller passes a value. That
  reads like a bug in the original (a leftover from a signature that had one
  more argument), but it is what MATLAB does, so the transcription reproduces
  it exactly, `nargin` emulation and all. PEASS never passes `iterations`, so
  nothing downstream is affected either way.
* **`Gfb_Delay_new` can index off the front of the impulse response.** It
  computes `impulse_response(band, band_max_index-1)`, and if a band's response
  peaked at its very first sample that subscript is 0 and MATLAB raises. Python
  would silently wrap around to the *last* sample and return a plausible wrong
  answer, so the port raises `IndexError` there instead. Not reachable with the
  PEASS parameters (the gammatone responses all rise over several samples), but
  a silent difference would be much worse than a loud one.
* **The pure-MATLAB branch of `Gfb_Analyzer_process` only supports a row
  vector,** despite the doc comment offering "a matrix containing different
  input signals for the different bands". The branch hands the whole of `input`
  to every band's filter, so a matrix input would fail the `output(band,:)`
  assignment. Only the MEX implements the documented matrix case. Transcribed
  as written; PEASS always passes a row vector (`x(:).'`).
* **`[dummy, max_indices] = max(...)`.** Both MATLAB's `max` and numpy's
  `argmax` report the *first* maximum, so ties resolve identically. `dummy` is
  kept, unused, as in the original.
* **Line endings.** The `.m` files are CRLF; every module under `reference/` is
  LF. The embedded MATLAB is byte-exact once line terminators are normalized
  (which `verify_transcription.py` does for all of `reference/`, not just
  these). A file's final newline is carried as a trailing empty embedded line —
  a bare `#` inside the last fence — because the verifier compares
  `text.split("\n")` element for element.

## Cross-checks run

* `README_examples.txt` publishes actual MATLAB output for two of these
  functions, and the port reproduces every printed digit:
  `Gfb_Filter_new(10000, 1000, 100, 3, 4)` → `coefficient` 0.7526+0.5468i,
  `normalization_factor` 4.7434e-05; and after a 200-sample impulse,
  `filter.state` = 1e-4 · [0.0000-0.0000i, 0.0000-0.0000i, 0.0043-0.0031i,
  0.2907-0.2112i].
* The documented filterbank example (`fs = 16276`, 70–6700 Hz, base 1000,
  1.0 filters/ERB) produces 30 bands, as `README_examples.txt` shows
  (`center_frequencies_hz: [1x30 double]`), which exercises the colon-operator
  rule above.
* `|Gfb_Analyzer_zresponse|` at each band's own centre frequency is exactly 2,
  the peak gain the normalization factor is designed for, and it agrees with
  the FFT of that band's impulse response.
* Processing a signal in two halves equals processing it in one go to 9e-19,
  i.e. the filter state really does round-trip through the returned struct.
* Analysis → synthesis of an impulse (`fs = 16276`, 4 ms delay, the
  `Example_Synthesis.m` parameters) peaks at sample 65 = round(0.004·16276),
  and the overall response sits in -5.7 .. 0.0 dB over 200–5000 Hz, falling to
  -7.7 dB at the edge bands. That ripple is the filterbank's, not the port's:
  at 1.0 filters/ERB the bands are sparse, and the gain optimisation in
  `Gfb_Mixer_new` only equalises the response *at the centre frequencies*.
