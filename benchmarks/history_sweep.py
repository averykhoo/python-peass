"""
PEASS git-history sweep: accuracy vs the MATLAB gold WAVs, and decomposition
speed, measured at a list of commits with ONE fixed yardstick.

Usage
-----
    python benchmarks/history_sweep.py                       # default commit list
    python benchmarks/history_sweep.py --commits a1b2c3 d4e5f6
    python benchmarks/history_sweep.py --repeats 5 --results-dir DIR
    python benchmarks/history_sweep.py --resume                 # after an interruption

Writes
    <results>/history_raw.json    everything, per commit x pass (gitignored)
    benchmarks/history.csv        tidy/long summary (this one is committed)
                                  override with --csv PATH

THE ONE FIXED YARDSTICK
-----------------------
A history sweep is only meaningful if every commit is judged by the *same*
measuring stick. So exactly one thing varies:

    * the library under test -- `peass/` is imported from a detached
      `git worktree` of the commit being measured.

Everything else is pinned to the checkout this script lives in, and is
deliberately NOT taken from the commit under test:

    * the gold WAVs and input audio (`tests/resources/`, from THIS checkout),
    * `_MATLAB_RESAMPLER_GAIN_OFFSET` and the component -> gold-filename map,
    * the nonlinear-estimate and stereo-synthesis conventions,
    * the measurement code itself (this file).

Note the difference from `measure.py`, which imports the gain offset and the
locked score dicts from `tests/regression/test_matlab_regression.py` so that it
always tracks the bar the test asserts. That coupling is exactly wrong here:
those constants *changed over the history being swept*
(`_MATLAB_RESAMPLER_GAIN_OFFSET` did not exist before commit 4ab223a, "lock the
resampler gain offset"), so importing them from the commit under test would
move the yardstick with the thing being measured. They are pinned literally
below instead, at their current-HEAD values.

TIMING METHOD
-------------
TODO.md ground rule 3: sequential wall-clock on this machine drifts 6-8%, which
is why `ab.py` (Thue-Morse interleaved A/B) is the only quotable timing tool.
Interleaving is impossible across git worktrees -- each candidate is a separate
process against a separate checkout -- so this harness does the next best
thing: it sweeps the whole commit list **forward, then again in reverse**, and
reports both passes. Thermal/background drift that accumulates along the sweep
lands on opposite ends of the list in the two passes, so a speed difference
that survives both is real and one that flips is drift. Per-commit agreement
between the passes is computed and anything disagreeing by more than
`--disagree-threshold` (default 10%) is flagged. Treat a flagged commit as "no
number", the same way `ab.py` reports UNRELIABLE.

Accuracy is deterministic and MUST be bit-identical between the two passes;
the aggregation checks that and reports it as a free correctness check on the
whole sweep.

FROZEN CONVENTIONS (copied from measure.py -- keep in step by hand)
    * nonlinear estimate: est = tanh(3*mix)/3 + shaped noise, mix = target +
      interferer, noise = 4th-order Butterworth lowpass at 4 kHz over white
      noise from np.random.default_rng(_NOISE_SEED), scaled to
      _NOISE_RMS_FRACTION of the mix RMS.
    * stereo input is synthesised from the mono exp01 clips as
      [x, 0.85 * x delayed by 17 samples + 2% independent noise], with distinct
      seeds for target and interferer. The independent noise is load-bearing:
      without it the per-frame Gram is rank-deficient and the backends diverge.
    * execution order inside each worker process is fixed (torch imported up
      front, accuracy first, then timings numpy-before-torch) because MKL/OMP
      state is process-global and order-sensitive.

API DRIFT
---------
Each commit is called through `_decompose`, which tries the current signature
first and falls back to older plausible ones (positional args, no
`DecompositionConfiguration`, no `sampling_frequency_hz`). A commit that cannot
be measured at all is recorded in the CSV with status=failed and an error note
rather than crashing the sweep or being silently dropped.
"""

import os

# Must precede `import torch`: conda MKL and the torch wheel each ship
# libiomp5md.dll and the second one to initialize aborts the process. Same
# escape hatch the repo's root conftest.py and measure.py use.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

_THIS = pathlib.Path(__file__).resolve()
# The repo root, derived from this file's location, so the harness runs from any
# checkout on any machine. Nothing under benchmarks/ may contain a machine-specific
# path.
REPO = _THIS.parents[1]

# ------------------------------------------------------------------ pinned --
# PINNED, NOT IMPORTED. See "THE ONE FIXED YARDSTICK" above: importing these
# from the commit under test would move the yardstick with the measurement.
# Values are HEAD's, from tests/regression/test_matlab_regression.py.
_MATLAB_RESAMPLER_GAIN_OFFSET = 1.0025651
_GAIN_TOLERANCE = 1e-3
_CORR_THRESHOLD = {"numpy": 0.999, "torch": 0.99}

_VALIDATION_MAP = [
    ("true_target", "targetEstimate_true.wav"),
    ("target_distortion", "targetEstimate_eTarget.wav"),
    ("interference", "targetEstimate_eInterf.wav"),
    ("artifacts", "targetEstimate_eArtif.wav"),
]
_COMPONENTS = [name for name, _ in _VALIDATION_MAP]

_SCORE_FIELDS = [
    "overall_perceptual_score",
    "target_perceptual_score",
    "interference_perceptual_score",
    "artifact_perceptual_score",
    "source_to_distortion_ratio",
    "source_to_spatial_distortion_ratio",
    "source_to_interference_ratio",
    "source_to_artifacts_ratio",
]

_NOISE_SEED = 20260810
_NOISE_RMS_FRACTION = 0.01  # shaped noise at -40 dB relative to the mix RMS
_NOISE_CUTOFF_HZ = 4000.0
_STEREO_DELAY = 17
_STEREO_GAIN = 0.85
_STEREO_DECORR = 0.02  # independent noise in ch1, as a fraction of the clip RMS
_STEREO_SEEDS = {"target": 20260811, "interferer": 20260812}

# The commits worth measuring: the ones that plausibly moved accuracy or speed,
# newest first, plus 5ed534a as a CONTROL. 5ed534a is documentation-only relative
# to c51f76a, so its accuracy numbers must come out byte-identical to c51f76a's.
# If they do not, something in this harness is not actually pinned.
# The two oldest entries are the pre-"full-order resampling" era (e281b56 is
# 3079de1's parent) and the torch time-reversed-haircell fix, kept because they
# are the earliest points where the accuracy trajectory can still be evaluated.
DEFAULT_COMMITS = [
    "04de76a", "ae0bdc2", "fda3030", "5ed534a", "c51f76a", "cd061f3",
    "07ba346", "4ab223a", "26c94a4", "76e53f3", "d866075", "3079de1",
    "e281b56", "c2c7e76",
]
CONTROL_PAIR = ("5ed534a", "c51f76a")

_ACCURACY_METRICS = ("correlation", "rms_ratio", "gain_error")
_TIMING_METRICS = (
    "median_seconds", "min_seconds", "audio_seconds",
    "seconds_per_audio_second", "realtime_factor", "spread_pct",
)

CSV_COLUMNS = [
    "pass", "order_index", "commit_short", "commit_full", "author_date", "subject",
    "kind", "backend", "layout", "component", "channel", "metric", "value",
    "status", "error",
]


# ============================================================== worker side ==
def _worker(peass_root: pathlib.Path, repo: pathlib.Path, repeats: int, out_path: pathlib.Path):
    """Measure one commit, in a fresh process, and write a JSON blob.

    `peass_root` is the git worktree of the commit under test -- the ONLY thing
    that varies. `repo` is this checkout, and supplies every input and every
    constant.
    """
    # peass must resolve to the worktree, ahead of anything else on sys.path.
    sys.path.insert(0, str(peass_root))

    import numpy as np
    import scipy.signal as sig
    import soundfile as sf
    import torch

    result = {
        "peass_root": str(peass_root),
        "repeats": repeats,
        "status": "ok",
        "error": "",
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "constants": {
            "matlab_resampler_gain_offset": _MATLAB_RESAMPLER_GAIN_OFFSET,
            "gain_tolerance": _GAIN_TOLERANCE,
            "noise_seed": _NOISE_SEED,
            "noise_rms_fraction": _NOISE_RMS_FRACTION,
            "noise_cutoff_hz": _NOISE_CUTOFF_HZ,
            "stereo_delay": _STEREO_DELAY,
            "stereo_gain": _STEREO_GAIN,
            "stereo_decorr": _STEREO_DECORR,
            "stereo_seeds": _STEREO_SEEDS,
        },
        "accuracy": {},
        "scores": {},
        "timings": {},
        "notes": [],
    }

    try:
        import peass
        result["peass_file"] = str(getattr(peass, "__file__", "?"))
        if not str(pathlib.Path(result["peass_file"]).resolve()).startswith(str(peass_root)):
            raise RuntimeError(f"peass resolved outside the worktree: {result['peass_file']}")
        decompose_distortion_components = peass.decompose_distortion_components
        predict_perceptual_evaluation_scores = peass.predict_perceptual_evaluation_scores
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"import peass: {type(exc).__name__}: {exc}"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0

    try:
        from peass.config import DecompositionConfiguration
    except Exception:
        DecompositionConfiguration = None
        result["notes"].append("no peass.config.DecompositionConfiguration")

    # ------------------------------------------------------------ helpers --
    def to_backend(data, backend):
        if backend == "torch":
            if isinstance(data, list):
                return [to_backend(x, backend) for x in data]
            return torch.tensor(np.asarray(data), device=torch.device("cpu"), dtype=torch.float64)
        return data

    def to_np(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(np.float64)
        return np.asarray(data, dtype=np.float64)

    def as_2d(x):
        x = np.asarray(x, dtype=np.float64)
        return x[:, None] if x.ndim == 1 else x

    def make_stereo(mono, role):
        mono = np.asarray(mono, dtype=np.float64).reshape(-1)
        delayed = np.concatenate([np.zeros(_STEREO_DELAY), mono[:-_STEREO_DELAY]])
        rng = np.random.default_rng(_STEREO_SEEDS[role])
        decorr = _STEREO_DECORR * np.sqrt(np.mean(mono ** 2)) * rng.standard_normal(mono.shape)
        return np.stack([mono, _STEREO_GAIN * delayed + decorr], axis=1)

    def nonlinear_estimate(target, interferer, fs):
        target = as_2d(target)
        interferer = as_2d(interferer)
        mix = target + interferer
        est = np.tanh(3.0 * mix) / 3.0
        rng = np.random.default_rng(_NOISE_SEED)
        noise = rng.standard_normal(mix.shape)
        b, a = sig.butter(4, _NOISE_CUTOFF_HZ / (fs / 2.0), btype="low")
        noise = sig.lfilter(b, a, noise, axis=0)
        scale = (_NOISE_RMS_FRACTION * np.sqrt(np.mean(mix ** 2))
                 / (np.sqrt(np.mean(noise ** 2)) + 1e-20))
        return est + scale * noise

    # ------------------------------------------------- API-drift tolerant --
    def _config():
        """Current config, degrading to whatever the old dataclass accepted."""
        if DecompositionConfiguration is None:
            return None
        for kwargs in (
            {"shade_in_milliseconds": 10.0, "shade_out_milliseconds": 10.0},
            {"shadeIn": 10.0, "shadeOut": 10.0},
            {},
        ):
            try:
                return DecompositionConfiguration(**kwargs)
            except TypeError:
                continue
        return None

    _call_style = {"style": None}

    def _decompose(sources, estimate, fs, backend):
        srcs = to_backend([as_2d(s) for s in sources], backend)
        est = to_backend(as_2d(estimate), backend)
        cfg = _config()
        styles = [
            ("kw_current", lambda: decompose_distortion_components(
                source_files=srcs, estimate_file=est, configuration=cfg,
                sampling_frequency_hz=float(fs))),
            ("kw_no_fs", lambda: decompose_distortion_components(
                source_files=srcs, estimate_file=est, configuration=cfg)),
            ("positional", lambda: decompose_distortion_components(srcs, est, cfg, float(fs))),
            ("positional_no_cfg", lambda: decompose_distortion_components(srcs, est)),
        ]
        if _call_style["style"] is not None:
            styles = [s for s in styles if s[0] == _call_style["style"]]
        last = None
        for name, fn in styles:
            try:
                out = fn()
            except TypeError as exc:
                last = exc
                continue
            _call_style["style"] = name
            # DecompositionResult.waveforms today; older shapes may be the
            # waveforms object itself.
            return getattr(out, "waveforms", out)
        raise RuntimeError(f"no working decompose signature: {last}")

    def _predict(sources, estimate, fs, backend):
        srcs = to_backend([as_2d(s) for s in sources], backend)
        est = to_backend(as_2d(estimate), backend)
        for fn in (
            lambda: predict_perceptual_evaluation_scores(srcs, est, sampling_frequency_hz=float(fs)),
            lambda: predict_perceptual_evaluation_scores(srcs, est),
        ):
            try:
                return fn()
            except TypeError:
                continue
        raise RuntimeError("no working predict signature")

    matlab_dir = repo / "tests" / "resources" / "matlab_reference"
    db_dir = repo / "tests" / "resources" / "database"

    # ======================================= (A) accuracy vs the gold WAVs ==
    target_src, fs = sf.read(matlab_dir / "targetSrc.wav")
    interf1, _ = sf.read(matlab_dir / "interfSrc1.wav")
    interf2, _ = sf.read(matlab_dir / "interfSrc2.wav")
    estimate, _ = sf.read(matlab_dir / "targetEstimate.wav")
    gold = {comp: sf.read(matlab_dir / fn)[0] for comp, fn in _VALIDATION_MAP}

    for backend in ("numpy", "torch"):
        try:
            t0 = time.perf_counter()
            waveforms = _decompose([target_src, interf1, interf2], estimate, fs, backend)
            elapsed = time.perf_counter() - t0
            per_component = {}
            for comp in _COMPONENTS:
                py_val = to_np(getattr(waveforms, comp))
                gold_val = as_2d(gold[comp])
                n = min(len(py_val), len(gold_val))
                entry = {"length_py": int(len(py_val)), "length_gold": int(len(gold_val)),
                         "channels": []}
                for ch in range(py_val.shape[1]):
                    py_ch = py_val[:n, ch]
                    gold_ch = gold_val[:n, ch]
                    gold_std = float(np.std(gold_ch))
                    ch_entry = {"channel": ch, "gold_std": gold_std,
                                "py_std": float(np.std(py_ch))}
                    if gold_std < 1e-4:
                        ch_entry["silent"] = True
                    else:
                        corr = float(np.corrcoef(py_ch, gold_ch)[0, 1])
                        rms_ratio = float(np.sqrt(np.mean(py_ch ** 2))
                                          / (np.sqrt(np.mean(gold_ch ** 2)) + 1e-20))
                        ch_entry.update({
                            "silent": False,
                            "correlation": corr,
                            "rms_ratio": rms_ratio,
                            "gain_error": rms_ratio / _MATLAB_RESAMPLER_GAIN_OFFSET - 1.0,
                            "corr_pass": corr > _CORR_THRESHOLD[backend],
                            "gain_pass": abs(rms_ratio / _MATLAB_RESAMPLER_GAIN_OFFSET - 1.0)
                            < _GAIN_TOLERANCE,
                        })
                    entry["channels"].append(ch_entry)
                per_component[comp] = entry
            result["accuracy"][backend] = {"decompose_seconds": elapsed,
                                           "components": per_component,
                                           "call_style": _call_style["style"]}
            print(f"    accuracy {backend}: ok ({elapsed:.1f}s)", flush=True)
        except Exception as exc:
            result["status"] = "partial"
            result["error"] += f"[accuracy/{backend}] {type(exc).__name__}: {exc} "
            result["accuracy"][backend] = {"error": traceback.format_exc(limit=4)}
            print(f"    accuracy {backend}: FAILED {exc}", flush=True)

    # -------------------------------------------------- 8 scores / ratios --
    for backend in ("numpy", "torch"):
        try:
            t0 = time.perf_counter()
            scores = _predict([target_src, interf1, interf2], estimate, fs, backend)
            elapsed = time.perf_counter() - t0
            vals = {}
            for field in _SCORE_FIELDS:
                v = getattr(scores, field, None)
                if v is None:
                    continue
                vals[field] = float(v.item() if isinstance(v, torch.Tensor) else v)
            result["scores"][backend] = {"seconds": elapsed, "values": vals}
            print(f"    scores {backend}: ok ({elapsed:.1f}s)", flush=True)
        except Exception as exc:
            result["status"] = "partial"
            result["error"] += f"[scores/{backend}] {type(exc).__name__}: {exc} "
            result["scores"][backend] = {"error": traceback.format_exc(limit=4)}
            print(f"    scores {backend}: FAILED {exc}", flush=True)

    # ============================================ (C) exp01 speed, per layout ==
    mono_target, fs_db = sf.read(db_dir / "exp01_target.wav")
    mono_interf, _ = sf.read(db_dir / "exp01_InterfSrc1.wav")
    cases = {
        "mono": [as_2d(mono_target), as_2d(mono_interf)],
        "stereo": [make_stereo(mono_target, "target"), make_stereo(mono_interf, "interferer")],
    }
    prepared = {}
    for layout, srcs in cases.items():
        # Each layout is normalized by ITS OWN duration; they need not match.
        audio_seconds = float(len(srcs[0]) / float(fs_db))
        prepared[layout] = (srcs, nonlinear_estimate(srcs[0], srcs[1], fs_db), audio_seconds)

    # numpy before torch, always: MKL/OMP state is process-global and order matters.
    for backend in ("numpy", "torch"):
        for layout in ("mono", "stereo"):
            key = f"{backend}_{layout}"
            srcs, est, audio_seconds = prepared[layout]
            try:
                t0 = time.perf_counter()
                w = _decompose(srcs, est, fs_db, backend)   # warm-up
                warm = time.perf_counter() - t0
                del w
                reps = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    w = _decompose(srcs, est, fs_db, backend)
                    reps.append(time.perf_counter() - t0)
                    del w
                median = float(np.median(reps))
                result["timings"][key] = {
                    "warmup_seconds": warm,
                    "repeats": reps,
                    "min": float(min(reps)),
                    "median": median,
                    "max": float(max(reps)),
                    "spread_pct": float((max(reps) - min(reps)) / min(reps) * 100.0),
                    "audio_seconds": audio_seconds,
                    "seconds_per_audio_second": median / audio_seconds,
                    "realtime_factor": audio_seconds / median,
                }
                print(f"    {key:14s} median={median:.3f}s  rtf={audio_seconds / median:.2f}x",
                      flush=True)
            except Exception as exc:
                result["status"] = "partial"
                result["error"] += f"[timing/{key}] {type(exc).__name__}: {exc} "
                result["timings"][key] = {"error": traceback.format_exc(limit=4)}
                print(f"    {key:14s} FAILED {exc}", flush=True)

    if not result["accuracy"] and not result["timings"]:
        result["status"] = "failed"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


# ======================================================== orchestrator side ==
def _git(*args, cwd=REPO):
    return subprocess.run(["git", "-C", str(cwd), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def _commit_meta(ref):
    out = _git("show", "-s", "--format=%H%x1f%h%x1f%aI%x1f%s", ref)
    full, short, date, subject = out.split("\x1f")
    return {"commit_full": full, "commit_short": short, "author_date": date, "subject": subject}


def _run_worker(peass_root, repeats, out_path, timeout):
    cmd = [sys.executable, str(_THIS), "--worker",
           "--peass-root", str(peass_root),
           "--repo", str(REPO),
           "--repeats", str(repeats),
           "--worker-out", str(out_path)]
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # cwd deliberately NOT the repo and NOT the worktree: '' on sys.path would
    # otherwise decide which `peass` wins by accident rather than by the explicit
    # sys.path insertion the worker makes.
    proc = subprocess.run(cmd, cwd=str(out_path.parent), env=env, timeout=timeout,
                          capture_output=True, text=True)
    return proc


def sweep(commits, results_dir, csv_path, repeats, timeout, keep_worktrees, disagree_threshold,
          resume=False):
    results_dir.mkdir(parents=True, exist_ok=True)
    wt_root = results_dir / "_worktrees"
    wt_root.mkdir(parents=True, exist_ok=True)
    worker_dir = results_dir / "_worker"
    worker_dir.mkdir(parents=True, exist_ok=True)

    meta = {c: _commit_meta(c) for c in commits}
    created = []
    raw = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "head": _commit_meta("HEAD"),
        "commits": meta,
        "repeats": repeats,
        "yardstick": {
            "resources_from": str(REPO / "tests" / "resources"),
            "matlab_resampler_gain_offset": _MATLAB_RESAMPLER_GAIN_OFFSET,
            "gain_tolerance": _GAIN_TOLERANCE,
            "note": "constants pinned in history_sweep.py, NOT imported from the commit under test",
        },
        "env": {"python": sys.version.split()[0], "platform": platform.platform()},
        "passes": {},
    }

    registered = {pathlib.Path(line.split()[0]).resolve()
                  for line in _git("worktree", "list").splitlines()[1:]}

    t_start = time.perf_counter()
    try:
        # ------------------------------------------------ materialize worktrees --
        for c in commits:
            path = wt_root / meta[c]["commit_short"]
            if resume and path.resolve() in registered and (path / "peass").is_dir():
                created.append(path)
                continue
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
                subprocess.run(["git", "-C", str(REPO), "worktree", "prune"],
                               capture_output=True, text=True)
            _git("worktree", "add", "--detach", str(path), meta[c]["commit_full"])
            created.append(path)
            print(f"worktree {meta[c]['commit_short']} -> {path}", flush=True)

        # ---------------------------------------------------------- the passes --
        for pass_name, order in (("forward", list(commits)), ("reverse", list(reversed(commits)))):
            print(f"\n===== pass: {pass_name} =====", flush=True)
            raw["passes"][pass_name] = {}
            for idx, c in enumerate(order):
                short = meta[c]["commit_short"]
                print(f"  [{idx + 1}/{len(order)}] {short}  {meta[c]['subject'][:64]}", flush=True)
                out_path = worker_dir / f"{pass_name}_{short}.json"
                entry = {"order_index": idx}
                if resume and out_path.exists():
                    try:
                        cached = json.loads(out_path.read_text(encoding="utf-8"))
                    except Exception:
                        cached = None
                    if cached and cached.get("status") in ("ok", "partial"):
                        entry.update(cached)
                        entry["resumed"] = True
                        raw["passes"][pass_name][short] = entry
                        print("    reused cached worker result", flush=True)
                        continue
                try:
                    proc = _run_worker(wt_root / short, repeats, out_path, timeout)
                    if out_path.exists():
                        entry.update(json.loads(out_path.read_text(encoding="utf-8")))
                    else:
                        entry.update({
                            "status": "failed",
                            "error": f"worker exit {proc.returncode}: "
                                     f"{(proc.stderr or proc.stdout)[-600:]}",
                            "accuracy": {}, "scores": {}, "timings": {},
                        })
                        print(f"    WORKER FAILED (exit {proc.returncode})", flush=True)
                except subprocess.TimeoutExpired:
                    entry.update({"status": "failed", "error": f"timeout after {timeout}s",
                                  "accuracy": {}, "scores": {}, "timings": {}})
                    print("    WORKER TIMEOUT", flush=True)
                raw["passes"][pass_name][short] = entry
    finally:
        # A failure mid-sweep must never leave worktrees registered against the repo.
        if keep_worktrees:
            print(f"\n--keep-worktrees: left {len(created)} worktrees under {wt_root}")
        else:
            for path in created:
                subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(path)],
                               capture_output=True, text=True)
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
            subprocess.run(["git", "-C", str(REPO), "worktree", "prune"],
                           capture_output=True, text=True)
            print(f"\ncleaned up {len(created)} worktrees")

    raw["total_seconds"] = time.perf_counter() - t_start
    raw["checks"] = _checks(raw, meta, commits, disagree_threshold)

    raw_path = results_dir / "history_raw.json"
    raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    rows = _to_rows(raw, meta, commits)
    _write_csv(csv_path, rows)
    print(f"\nwrote {raw_path}")
    print(f"wrote {csv_path}  ({len(rows)} rows)")
    _print_summary(raw, meta, commits, disagree_threshold)
    return raw


def _acc_value(entry, backend, comp, ch, metric):
    try:
        chans = entry["accuracy"][backend]["components"][comp]["channels"]
        return chans[ch].get(metric)
    except Exception:
        return None


def _checks(raw, meta, commits, disagree_threshold):
    """Accuracy must be identical across passes; timing must agree within threshold."""
    checks = {"accuracy_pass_identical": True, "accuracy_mismatches": [],
              "timing_disagreements": [], "control": {}}
    fwd = raw["passes"].get("forward", {})
    rev = raw["passes"].get("reverse", {})
    for c in commits:
        short = meta[c]["commit_short"]
        f, r = fwd.get(short), rev.get(short)
        if not f or not r:
            continue
        for backend in ("numpy", "torch"):
            for comp in _COMPONENTS:
                for ch in (0, 1):
                    for metric in _ACCURACY_METRICS:
                        a = _acc_value(f, backend, comp, ch, metric)
                        b = _acc_value(r, backend, comp, ch, metric)
                        if a is None and b is None:
                            continue
                        if a != b:
                            checks["accuracy_pass_identical"] = False
                            checks["accuracy_mismatches"].append(
                                {"commit": short, "backend": backend, "component": comp,
                                 "channel": ch, "metric": metric, "forward": a, "reverse": b})
        for key in sorted(set(f.get("timings", {})) | set(r.get("timings", {}))):
            a = f.get("timings", {}).get(key, {}).get("median")
            b = r.get("timings", {}).get(key, {}).get("median")
            if a and b:
                rel = abs(a - b) / min(a, b)
                if rel > disagree_threshold:
                    checks["timing_disagreements"].append(
                        {"commit": short, "key": key, "forward": a, "reverse": b,
                         "disagreement_pct": 100.0 * rel})

    # The control: 5ed534a is docs-only relative to c51f76a -> identical accuracy.
    ctrl, base = CONTROL_PAIR
    def _short_of(ref):
        for c, m in meta.items():
            if m["commit_full"].startswith(ref) or m["commit_short"] == ref or c == ref:
                return m["commit_short"]
        return None
    cs, bs = _short_of(ctrl), _short_of(base)
    if cs and bs:
        diffs = []
        for pass_name in ("forward", "reverse"):
            p = raw["passes"].get(pass_name, {})
            if cs not in p or bs not in p:
                continue
            for backend in ("numpy", "torch"):
                for comp in _COMPONENTS:
                    for ch in (0, 1):
                        for metric in _ACCURACY_METRICS:
                            a = _acc_value(p[cs], backend, comp, ch, metric)
                            b = _acc_value(p[bs], backend, comp, ch, metric)
                            if a is None and b is None:
                                continue
                            if a != b:
                                diffs.append({"pass": pass_name, "backend": backend,
                                              "component": comp, "channel": ch,
                                              "metric": metric, ctrl: a, base: b})
        checks["control"] = {"control_commit": ctrl, "reference_commit": base,
                             "identical": not diffs, "differences": diffs[:20]}
    return checks


def _to_rows(raw, meta, commits):
    rows = []
    for pass_name in ("forward", "reverse"):
        for short, entry in raw["passes"].get(pass_name, {}).items():
            m = next(meta[c] for c in commits if meta[c]["commit_short"] == short)
            base = {
                "pass": pass_name,
                "order_index": entry.get("order_index", ""),
                "commit_short": short,
                "commit_full": m["commit_full"],
                "author_date": m["author_date"],
                "subject": m["subject"],
                "status": entry.get("status", "failed"),
                # Kept short on purpose: this column repeats on every row of the
                # commit. The full traceback lives in history_raw.json.
                "error": " ".join((entry.get("error") or "").split())[:120],
            }

            def add(kind, backend, layout, component, channel, metric, value):
                if value is None:
                    return
                rows.append({**base, "kind": kind, "backend": backend, "layout": layout,
                             "component": component, "channel": channel,
                             "metric": metric, "value": value})

            add("meta", "", "", "", "", "measured",
                1 if entry.get("status") in ("ok", "partial") else 0)

            for backend, acc in (entry.get("accuracy") or {}).items():
                for comp, comp_entry in (acc.get("components") or {}).items():
                    for ch_entry in comp_entry.get("channels", []):
                        if ch_entry.get("silent"):
                            continue
                        for metric in _ACCURACY_METRICS:
                            add("accuracy", backend, "matlab_reference", comp,
                                ch_entry["channel"], metric, ch_entry.get(metric))
            for backend, sc in (entry.get("scores") or {}).items():
                for field, value in (sc.get("values") or {}).items():
                    add("score", backend, "matlab_reference", "", "", field, value)
            for key, t in (entry.get("timings") or {}).items():
                if "median" not in t:
                    continue
                backend, layout = key.split("_", 1)
                for metric in _TIMING_METRICS:
                    src = {"median_seconds": t.get("median"), "min_seconds": t.get("min")}
                    add("timing", backend, layout, "", "", metric,
                        src.get(metric, t.get(metric)))
    return rows


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            v = row.get("value")
            if isinstance(v, float):
                row = {**row, "value": repr(v)}
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def _print_summary(raw, meta, commits, disagree_threshold):
    checks = raw["checks"]
    print("\n===== checks =====")
    print(f"accuracy identical across passes: {checks['accuracy_pass_identical']}"
          f"  ({len(checks['accuracy_mismatches'])} mismatches)")
    if checks.get("control"):
        print(f"control {checks['control']['control_commit']} vs "
              f"{checks['control']['reference_commit']}: "
              f"identical={checks['control']['identical']}")
    print(f"timing disagreements > {disagree_threshold * 100:.0f}%: "
          f"{len(checks['timing_disagreements'])}")
    for d in checks["timing_disagreements"]:
        print(f"  FLAG {d['commit']:9s} {d['key']:14s} "
              f"fwd={d['forward']:.3f}s rev={d['reverse']:.3f}s ({d['disagreement_pct']:.1f}%)")

    print("\n===== accuracy: correlation vs MATLAB gold (forward pass, ch0) =====")
    hdr = f"{'commit':10s} {'date':11s} " + " ".join(f"{c[:12]:>13s}" for c in _COMPONENTS)
    for backend in ("numpy", "torch"):
        print(f"-- {backend}")
        print(hdr)
        for c in commits:
            short = meta[c]["commit_short"]
            e = raw["passes"]["forward"].get(short, {})
            cells = []
            for comp in _COMPONENTS:
                v = _acc_value(e, backend, comp, 0, "correlation")
                cells.append(f"{v:13.9f}" if v is not None else f"{'-':>13s}")
            print(f"{short:10s} {meta[c]['author_date'][:10]:11s} " + " ".join(cells))

    print("\n===== speed: realtime factor (audio seconds per compute second) =====")
    keys = ["numpy_mono", "numpy_stereo", "torch_mono", "torch_stereo"]
    print(f"{'commit':10s} " + " ".join(f"{k:>22s}" for k in keys))
    for c in commits:
        short = meta[c]["commit_short"]
        cells = []
        for k in keys:
            vals = []
            for p in ("forward", "reverse"):
                t = raw["passes"].get(p, {}).get(short, {}).get("timings", {}).get(k, {})
                vals.append(t.get("realtime_factor"))
            if vals[0] and vals[1]:
                cells.append(f"{vals[0]:9.3f}/{vals[1]:<9.3f}   ".rjust(22))
            else:
                cells.append(f"{'-':>22s}")
        print(f"{short:10s} " + " ".join(cells))
    print("(forward/reverse; both shown deliberately -- see TIMING METHOD in the docstring)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--commits", nargs="+", default=DEFAULT_COMMITS,
                        help="commits to sweep, newest first")
    parser.add_argument("--repeats", type=int, default=5,
                        help="timed decomposition repeats per backend x layout (after a warm-up)")
    parser.add_argument("--timeout", type=int, default=3600, help="per-commit worker timeout (s)")
    parser.add_argument("--csv", default=str(REPO / "benchmarks" / "history.csv"),
                        help="tidy CSV output path")
    parser.add_argument("--disagree-threshold", type=float, default=0.10,
                        help="flag a commit whose two passes differ by more than this fraction")
    parser.add_argument("--keep-worktrees", action="store_true",
                        help="do not remove the git worktrees afterwards (debugging)")
    parser.add_argument("--resume", action="store_true",
                        help="reuse per-commit worker JSONs already in <results>/_worker/ instead "
                             "of re-measuring them (the full sweep takes over an hour; this makes "
                             "an interrupted run recoverable). Pass order is unchanged, but a "
                             "resumed pass has a wall-clock gap in it -- see TIMING METHOD.")
    parser.add_argument("--results-dir", default=None, metavar="DIR",
                        help="where history_raw.json and scratch go "
                             "(default: $PEASS_BENCH_RESULTS or <repo>/benchmarks/results)")
    # worker mode
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--peass-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repo", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        return _worker(pathlib.Path(args.peass_root).resolve(),
                       pathlib.Path(args.repo).resolve(),
                       args.repeats,
                       pathlib.Path(args.worker_out).resolve())

    # Same --results-dir convention as measure.py / compare.py, via the shared helper.
    sys.path.insert(0, str(REPO))
    from benchmarks._paths import resolve_results_dir
    results_dir = resolve_results_dir(args.results_dir)
    sweep(args.commits, results_dir, pathlib.Path(args.csv).resolve(),
          args.repeats, args.timeout, args.keep_worktrees, args.disagree_threshold,
          resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
