# todo

- train a different model on the peass data (which was removed in PR #3, commit `7ad923b3` on 2026-06-06 17:01)
  - get more data from https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS
- use peass decomposition as an ablation for haspi metrics
- note to self: add `-n auto` for pycharm to speed up tests,
  examples here: https://www.jetbrains.com/help/pycharm/performing-tests.html#run-tests-in-parallel
- add reflection padding milliseconds as a configurable option, it's helpful for short files (especially under 1s), 
  see `test_torch_decomposition.py:test_torch_decomposition_gain_invariance_with_padding` for details

## deferred from the 2026-07 code review (not release-blocking)

- perf: `backend_torch/auditory_model.py` runs the auditory-nerve adaptation as a
  `torch.jit.script` per-sample loop at the up-sampled rate (~24 kHz). This is a
  performance cliff for multi-second audio. It is an inherently sequential
  *nonlinear* first-order recurrence (each timestep divides by the previous
  state), so it cannot be exactly parallelized with an associative scan; any
  speedup is an approximation that needs a torch runtime to verify against the
  numpy reference (torch is not installed in the dev env). Left as-is.
- tests: `test_torch_decomposition.py:~115` relaxes tolerances with a stale TODO
  ("will pass once linalg.pinv is added") even though `torch.linalg.pinv` is now
  used. Re-check whether the strict tolerance passes and, if not, track the gap
  via `xfail(strict=True)` + an issue. Needs a torch runtime to verify.
- tests: `test_linalg_solve_fallback_parity` in the differential file compares
  scipy vs torch library reimplementations rather than the production code path;
  convert to exercise the real solver/fallback or remove.
- the `_EXPECTED_SCORES` characterization values in `test_matlab_regression.py`
  are Python-reference (not MATLAB-published) numbers; replace with MATLAB's
  actual OPS/TPS/IPS/APS for the example clips if/when available.