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
  performance cliff for multi-second audio. Vectorize via an associative/scan
  formulation, or run it after decimation. Deferred because it needs a torch
  runtime to verify numerical parity (torch is not installed in the dev env).
- tests: `test_differential_numpy_vs_torch.py` still contains two debug-harness
  tests worth cleaning up — `test_linalg_solve_fallback_parity` compares scipy vs
  torch reimplementations rather than the production code path, and
  `test_synthesis_fixed_delay_parity` monkey-patches a hand-copied fork of
  `run_auditory_synthesis_filterbank_torch` that can silently drift. Convert to
  real production-path tests or remove.
- tests: `test_torch_decomposition.py:~115` relaxes tolerances with a TODO
  comment; track the backend accuracy gap via `xfail(strict=True)` + an issue
  instead of loosening the assertion.
- tests: a few numpy unit tests assert only shapes (e.g.
  `test_numpy_gammatone.py::test_gammatone_analysis_reconstruction`) or use loose
  0.85 reconstruction thresholds — tighten to value/fidelity checks.
- the `_EXPECTED_SCORES` characterization values in `test_matlab_regression.py`
  are Python-reference (not MATLAB-published) numbers; replace with MATLAB's
  actual OPS/TPS/IPS/APS for the example clips if/when available.