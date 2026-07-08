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
- perf/deprecation: the torch adaptation loop uses `@torch.jit.script`, which
  torch 2.x deprecates in favor of `torch.compile`/`torch.export`. Migrating is
  non-trivial (the loop is a sequential nonlinear recurrence) and orthogonal to
  the perf cliff; revisit together. (The loop was unrolled for ~1.35x in 2026-07;
  the haircell OOM was fixed via FFT convolution.)
- perf/backprop: the torch metrics (auditory adaptation loop) dominate backprop —
  ~10.5x backward/forward, ~76s for 0.5s audio — because it is BPTT through the
  ~12000-step sequential recurrence (the decomposition's pinv backward is cheap by
  comparison). A custom autograd.Function with a hand-derived analytic backward for
  the 5-stage adaptation cascade would make training-scale backprop practical;
  it's substantial and needs careful gradient validation. (The torch<->numpy score
  divergence itself was fixed via the straight-through max in 2026-07.)
- the `_EXPECTED_SCORES` characterization values in `test_matlab_regression.py`
  are Python-reference numbers (the decomposition now matches MATLAB to ~0.9999,
  but we still don't have MATLAB's published OPS/TPS/IPS/APS to assert against);
  replace with MATLAB's actual scores for the example clips if/when available.

Resolved 2026-07 (torch runtime now available locally, KMP_DUPLICATE_LIB_OK=TRUE):
the `test_torch_decomposition` relaxed tolerance is the shared Gammatone
reconstruction floor (numpy hits the same ~1.9e-3), not a torch gap — comment
corrected and real numpy-vs-torch parity assertions added; the redundant
`test_linalg_solve_fallback_parity` (tested library internals, not our code) was
removed.