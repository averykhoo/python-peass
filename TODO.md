# todo

Open work only. Settled items — landed fixes, closed investigations, declined options —
live in `ARCHIVE.md`.

- train a different model on the peass data (which was removed in PR #3, commit `7ad923b3` on 2026-06-06 17:01)
  - get more data from https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS
- use peass decomposition as an ablation for haspi metrics
- note to self: add `-n auto` for pycharm to speed up tests,
  examples here: https://www.jetbrains.com/help/pycharm/performing-tests.html#run-tests-in-parallel
- add reflection padding milliseconds as a configurable option, it's helpful for short files (especially under 1s), 
  see `test_torch_decomposition.py:test_torch_decomposition_gain_invariance_with_padding` for details

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
  cascade-collapse work in `ARCHIVE.md`) — because it is BPTT through the sequential
  recurrence (the decomposition's pinv backward is cheap by comparison). The gradient
  path cannot use the fast `_adaptation_loop_forward` since it needs the
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
