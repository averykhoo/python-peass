# todo

- train a different model on the peass data (which was removed in PR #3, commit `7ad923b3` on 2026-06-06 17:01)
  - get more data from https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS
- use peass decomposition as an ablation for haspi metrics
- note to self: add `-n auto` for pycharm to speed up tests,
  examples here: https://www.jetbrains.com/help/pycharm/performing-tests.html#run-tests-in-parallel
- add reflection padding milliseconds as a configurable option, it's helpful for short files (especially under 1s), 
  see `test_torch_decomposition.py:test_torch_decomposition_gain_invariance_with_padding` for details
- implement `segmentation_factor` — it is declared on `DecompositionConfiguration`
  (`peass/config.py`) but nothing reads it, so any value other than `1` is silently
  ignored. MATLAB does the work in `extractDistortionComponents.m`: the
  `options.segmentationFactor > 1` branch at ~lines 107-110 short-circuits into
  `aux_segmentAndDecompose` (~lines 270-386), which cuts every source and the estimate
  into `segmentationFactor` segments of `ceil(nSamples/segmentationFactor)` samples via
  `aux_cutWav`, recurses with `segmentationFactor = 1` and with `shadeInMs`/`shadeOutMs`
  forced to 0 except on the first/last segment respectively, then `aux_mergeWav`
  overlap-adds each of the four components under a periodic Hann window and divides by the
  accumulated window where it is non-zero. The config field already exists, so the public
  API surface does not change — this is purely a decomposition-path implementation, plus
  removing the "accepted but ignored" notes in `peass/config.py` and README.

## deferred from the 2026-07 code review (not release-blocking)

- perf/deprecation: the torch adaptation loop uses `@torch.jit.script`, which
  torch 2.x deprecates in favor of `torch.compile`/`torch.export`. `torch.compile`
  would additionally fuse the ~7 per-timestep dispatches into one kernel, which is
  exactly what this dispatch-bound loop wants — but it cannot be evaluated on this
  machine: inductor's CPU backend needs MSVC `cl`, which is not installed
  (`InductorError: Compiler: cl is not found`). Revisit on a box with a working
  inductor backend or a CUDA runtime.
- perf/backprop: the torch metrics (auditory adaptation loop) still dominate
  backprop — ~1.2x backward/forward, ~39s for 2s audio (2026-07-30, after the
  vectorized loop below) — because it is BPTT through the sequential recurrence
  (the decomposition's pinv backward is cheap by comparison). The gradient path
  cannot use the fast `_adaptation_loop_forward` since it needs the
  straight-through max, so it runs ~4.7x slower than the no-grad path. A custom
  autograd.Function with a hand-derived analytic backward for the 5-stage cascade
  would make training-scale backprop practical; it's substantial and needs careful
  gradient validation.
- perf: the adaptation loop is now dispatch-bound at ~7 tensor ops per timestep.
  Going meaningfully faster in pure torch needs either kernel fusion (see the
  `torch.compile` item above) or an approximate parallelization — e.g. DEER-style
  Newton iteration over the sequence, where each Newton step is a linear
  recurrence solvable by associative scan. That is research-grade and would break
  the current bit-level agreement with the NumPy reference, so it was not pursued.
- the `_EXPECTED_SCORES` characterization values in `test_matlab_regression.py`
  are Python-reference numbers (the decomposition now matches MATLAB to ~0.9999,
  but we still don't have MATLAB's published OPS/TPS/IPS/APS to assert against);
  replace with MATLAB's actual scores for the example clips if/when available.

Closed investigation (do not reopen) — the +0.257% level offset against the MATLAB gold
WAVs is root-caused and deliberately **not** fixed. Our decomposition output is a flat
factor 1.0025651 larger in amplitude than `tests/resources/matlab_reference/` on all four
components, because `scipy.signal.firwin` defaults to `scale=True` (unit DC gain) while
MATLAB's `resample` filter is not DC-normalized (raw DC gain 0.9993253); four resamples per
path give 0.999394194^2 * 0.999325320^2 = 0.997441484, whose reciprocal is 1.00256514. It
is frequency-flat because the tap set is scale-invariant in `pqmax`. A line-by-line MATLAB
transcription confirms flipping only that normalization takes the ratio to 1.0000001, with
the residual being PCM16 quantization noise. The gold WAVs are also stale — byte-identical
to the v2.0 shipped outputs, never regenerated for 2.0.1. Full write-up in README under
"Known deviations from the MATLAB reference".

Resolved — gammatone filter constructor now uses MATLAB's ERB form `24.7 + fc/9.265`
(`Gfb_Filter_new.m` / `Gfb_set_constants.m`, Hohmann 2002 eq. 17) instead of the
algebraically-equivalent-in-intent `erbBW` form `24.7*(0.00437*fc + 1)`. MATLAB-parity fix;
numerical impact on typical clips is negligible.

Resolved — shade-in/shade-out window now matches MATLAB's
`hann(2*round(ms/1000*fs + 1), 'periodic')(2:end/2)` strict-interior ramp instead of a full
0→1 ramp. MATLAB-parity fix; the shape difference is confined to the first/last few
milliseconds and is negligible on typical clips. The same fix also corrected the window
*length*: MATLAB's `round` breaks ties away from zero while Python/NumPy round half to
even, so `round(ms/1000*fs)` was one sample short wherever the product lands exactly on
`.5` — which includes the common cases 44100 Hz @ 5 ms and @ 25 ms, and 22050 Hz @ 10 ms.
Length is now `floor(x + 0.5)`.

Resolved 2026-07-30 — torch backend ~5.7x faster end-to-end (10.8s -> 1.9s for 1s
audio, 54.2s -> 9.7s for 5s; scores unchanged to 6 decimals). The adaptation-loop
"cannot be exactly parallelized" note was true about the *time* axis but missed
that the 5-stage *cascade* collapses exactly: every division at step t reads step
t-1 state, so it is one `cumprod` + one divide rather than five interleaved pairs
(~75 dispatches per timestep -> ~7). Also: FFT convolutions were running at
arbitrary (often near-prime) lengths, and the resampler re-transformed its FIR
filter on all ~200 calls per decomposition.

Resolved 2026-07 (torch runtime now available locally, KMP_DUPLICATE_LIB_OK=TRUE):
the `test_torch_decomposition` relaxed tolerance is the shared Gammatone
reconstruction floor (numpy hits the same ~1.9e-3), not a torch gap — comment
corrected and real numpy-vs-torch parity assertions added; the redundant
`test_linalg_solve_fallback_parity` (tested library internals, not our code) was
removed.