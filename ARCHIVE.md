# archive

Settled work: investigations that are root-caused, fixes that have landed, and options
deliberately declined. Nothing here is open — it lives outside `TODO.md` so the handoff
list stays short, and it is kept rather than deleted because the reasoning is what stops
the same ground being re-covered. Entries are append-only; if one is genuinely reopened,
move it back to `TODO.md`.

---

## Declined

### `segmentation_factor` — fail loudly instead of porting (2026-08-05)

`DecompositionConfiguration.segmentation_factor` now raises `NotImplementedError` in
`__post_init__` for any value other than `1` (`peass/config.py`, guarded by
`tests/unit/test_config.py`). The MATLAB path was **not** ported, by decision: it is not a
quality knob — MATLAB documents it purely as a peak-memory relief valve ("increase this
integer if you experienced out of memory problems", `example.m:26`) — and implementing it
would mean validating decomposition accuracy across the whole reasonable range of the
parameter for no accuracy gain. Failing loudly costs nothing and gives a MATLAB user
chasing an OOM the full story from the traceback.

Reopen only if someone actually hits an out-of-memory decomposition. The port spec, for
whoever does:

MATLAB does the work in `extractDistortionComponents.m`. The `options.segmentationFactor > 1`
branch at ~lines 107-110 short-circuits into `aux_segmentAndDecompose` (~lines 270-386),
which cuts every source and the estimate via `aux_cutWav` into windows of
`N = 2*round(TCut*fs/2)` samples at a 50% hop, where `TCut = ceil(nSamples/segmentationFactor)/fs`
— so the actual segment count is ~`2*segmentationFactor - 1` overlapping windows, not
`segmentationFactor` disjoint ones. It recurses with `segmentationFactor = 1` and with
`shadeInMs`/`shadeOutMs` forced to 0 except on the first/last segment respectively, then
`aux_mergeWav` overlap-adds each of the four components under `flipud(hann(N,'periodic'))`
and divides by the accumulated window where it is non-zero. The config field already
exists, so the public API surface would not change — it is purely a decomposition-path
implementation, plus dropping the `__post_init__` guard and its test.

---

## Closed investigations (do not reopen)

### +0.257% level offset against the MATLAB gold WAVs

Root-caused and deliberately **not** fixed. Our decomposition output is a flat factor
1.0025651 larger in amplitude than `tests/resources/matlab_reference/` on all four
components, because `scipy.signal.firwin` defaults to `scale=True` (unit DC gain) while
MATLAB's `resample` filter is not DC-normalized (raw DC gain 0.9993253); four resamples per
path give 0.999394194^2 * 0.999325320^2 = 0.997441484, whose reciprocal is 1.00256514. It
is frequency-flat because the tap set is scale-invariant in `pqmax`. A line-by-line MATLAB
transcription confirms flipping only that normalization takes the ratio to 1.0000001, with
the residual being PCM16 quantization noise. The gold WAVs are also stale — byte-identical
to the v2.0 shipped outputs, never regenerated for 2.0.1.

The offset is now locked rather than tolerated: `test_matlab_regression.py` asserts the
measured ratio *equals* `_MATLAB_RESAMPLER_GAIN_OFFSET = 1.0025651` to within 1e-3, so a new
gain regression in either direction breaks the test. Full write-up in README under "Known
deviations from the MATLAB reference".

---

## Resolved

### Gammatone ERB form (MATLAB parity)

The gammatone filter constructor now uses MATLAB's ERB form `24.7 + fc/9.265`
(`Gfb_Filter_new.m` / `Gfb_set_constants.m`, Hohmann 2002 eq. 17) instead of the
algebraically-equivalent-in-intent `erbBW` form `24.7*(0.00437*fc + 1)`. Numerical impact on
typical clips is negligible.

### Shade-in/shade-out window shape and length (MATLAB parity)

The shade window now matches MATLAB's `hann(2*round(ms/1000*fs + 1), 'periodic')(2:end/2)`
strict-interior ramp instead of a full 0→1 ramp. The shape difference is confined to the
first/last few milliseconds and is negligible on typical clips. The same fix also corrected
the window *length*: MATLAB's `round` breaks ties away from zero while Python/NumPy round
half to even, so `round(ms/1000*fs)` was one sample short wherever the product lands exactly
on `.5` — which includes the common cases 44100 Hz @ 5 ms and @ 25 ms, and 22050 Hz @ 10 ms.
Length is now `floor(x + 0.5)`.

### Torch backend ~5.7x faster end-to-end (2026-07-30)

10.8s -> 1.9s for 1s audio, 54.2s -> 9.7s for 5s; scores unchanged to 6 decimals. The
adaptation-loop "cannot be exactly parallelized" note was true about the *time* axis but
missed that the 5-stage *cascade* collapses exactly: every division at step t reads step t-1
state, so it is one `cumprod` + one divide rather than five interleaved pairs (~75 dispatches
per timestep -> ~7). Also: FFT convolutions were running at arbitrary (often near-prime)
lengths, and the resampler re-transformed its FIR filter on all ~200 calls per decomposition.

### Torch decomposition tolerance was not a torch gap (2026-07)

With a torch runtime available locally (`KMP_DUPLICATE_LIB_OK=TRUE`): the
`test_torch_decomposition` relaxed tolerance is the shared Gammatone reconstruction floor
(numpy hits the same ~1.9e-3), not a torch gap — comment corrected and real numpy-vs-torch
parity assertions added. The redundant `test_linalg_solve_fallback_parity` (it tested library
internals, not our code) was removed.
