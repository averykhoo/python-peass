"""
PEASS performance / accuracy measurement harness.

Usage:  python benchmarks/measure.py <tag> [--results-dir DIR]
        e.g.  python benchmarks/measure.py baseline

Writes (see `_paths.py` for how the directory is resolved):
    <results>/<tag>.json          all numbers (accuracy, scores, timings)
    <results>/<tag>_wav/*.npy     raw float64 component waveforms

Sections
--------
(A) ACCURACY VS MATLAB -- reproduces tests/regression/test_matlab_regression.py's
    comparison (same inputs, same config, same gold WAVs, same gain-offset
    constant) but records the numbers instead of asserting them. Also records the
    8 scores/ratios that test_numpy_perceptual_scores_characterization locks.

(B) SELF-CONSISTENCY ARTIFACTS -- saves every decomposed component waveform as
    float64 .npy, for both backends, on both the matlab_reference input and the
    exp01_* nonlinear-estimate input, so a later tag can be diffed waveform-vs-
    waveform at full precision.

(C) TIMINGS -- decomposition wall time for {torch,numpy} x {mono,stereo} on
    tests/resources/database/exp01_*, using the nonlinear estimate described in
    ARCHIVE.md ("Decomposition: torch ~2.2x, numpy ~1.47x"), warmed, six repeats.
    These timings are a coarse signal only: this machine drifts 6-8% between
    runs (TODO.md ground rule 3), so anything under ~10% needs `ab.py` instead.

FROZEN CONVENTIONS (must not change between tags, or comparisons are meaningless)
    * nonlinear estimate:  est = tanh(3*mix)/3 + shaped noise,
      mix = target + interferer, noise = 4th-order Butterworth lowpass at 4 kHz
      over white noise from np.random.default_rng(_NOISE_SEED), scaled to
      _NOISE_RMS_FRACTION of the mix RMS.  ARCHIVE.md fixes the tanh form; the
      noise shaping/seed/level are pinned here.
    * stereo input is synthesised from the *mono* exp01 clips (the only exp01
      files that exist) as
          [x, 0.85 * x delayed by 17 samples + 2% independent shaped-free noise]
      applied to target and interferer with distinct seeds.  The independent
      noise term is NOT cosmetic: without it channel 1 is an exactly FIR-
      realisable image of channel 0 (the LS filter is 0.04 s = 640 taps, far
      longer than the 17-sample delay), the per-frame Gram is rank-deficient, and
      `target_distortion` / `interference` become minimum-norm-solution garbage
      with peaks ~1.7 that differ between numpy and torch by 9e-2 relative.  With
      the noise the two backends agree to ~4e-6 and peaks are ~0.1.
    * execution order inside the process is fixed (torch is imported up front,
      accuracy first, then timings numpy-before-torch) because MKL/OMP state is
      process-global and order-sensitive.
"""

import os

# Must precede `import torch`: conda MKL and the torch wheel each ship
# libiomp5md.dll and the second one to initialize aborts the process. Same
# escape hatch the repo's root conftest.py uses.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.signal as sig  # noqa: E402
import soundfile as sf  # noqa: E402

# The repo root, derived from this file's location (benchmarks/measure.py), so the
# harness runs from any checkout on any machine. On sys.path so that `peass`, the
# `benchmarks` package itself and the `tests` package all import whether this is run
# as `python benchmarks/measure.py` or `python -m benchmarks.measure`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks._paths import REPO  # noqa: E402
from benchmarks._paths import add_results_dir_argument  # noqa: E402
from benchmarks._paths import resolve_results_dir  # noqa: E402

import torch  # noqa: E402

from peass import decompose_distortion_components  # noqa: E402
from peass import predict_perceptual_evaluation_scores  # noqa: E402
from peass.config import DecompositionConfiguration  # noqa: E402

# ---------------------------------------------------------------- constants --
# Imported from the regression test, NOT copied: this harness exists to record
# the numbers that test asserts, so the day someone retunes the gain offset or
# the locked scores, the harness must move with it. The names are private and
# are imported anyway on purpose -- a public alias in the test module would just
# be a second definition to keep in sync, and the coupling is deliberate. (The
# import pulls in pytest, which is a dev dependency and always present in an
# environment where running this harness makes sense.)
from tests.regression.test_matlab_regression import _EXPECTED_RATIOS  # noqa: E402
from tests.regression.test_matlab_regression import _EXPECTED_SCORES  # noqa: E402
from tests.regression.test_matlab_regression import _GAIN_TOLERANCE  # noqa: E402
from tests.regression.test_matlab_regression import _MATLAB_RESAMPLER_GAIN_OFFSET  # noqa: E402

# These two are *function-local* in the test (`validation_map` and
# `corr_threshold` inside test_regression_against_matlab_references), so there
# is nothing importable to bind to; they are mirrored here instead. Keep them in
# step with that function by hand.
_CORR_THRESHOLD = {"numpy": 0.999, "torch": 0.99}

_VALIDATION_MAP = [
    ("true_target", "targetEstimate_true.wav"),
    ("target_distortion", "targetEstimate_eTarget.wav"),
    ("interference", "targetEstimate_eInterf.wav"),
    ("artifacts", "targetEstimate_eArtif.wav"),
]
_COMPONENTS = [name for name, _ in _VALIDATION_MAP]

_SCORE_FIELDS = list(_EXPECTED_SCORES) + list(_EXPECTED_RATIOS)

_NOISE_SEED = 20260810
_NOISE_RMS_FRACTION = 0.01  # shaped noise at -40 dB relative to the mix RMS
_NOISE_CUTOFF_HZ = 4000.0
_STEREO_DELAY = 17
_STEREO_GAIN = 0.85
_STEREO_DECORR = 0.02  # independent noise in ch1, as a fraction of the clip RMS
_STEREO_SEEDS = {"target": 20260811, "interferer": 20260812}

_REPEATS = 6

MATLAB_DIR = REPO / "tests" / "resources" / "matlab_reference"
DB_DIR = REPO / "tests" / "resources" / "database"


# ------------------------------------------------------------------ helpers --
def to_backend(data, backend):
    """Same conversion tests/conftest.py::to_backend_format performs (CPU only)."""
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
    """[x, 0.85 * delay17(x) + 2% independent noise] -- see FROZEN CONVENTIONS.

    The independent noise makes the per-frame Gram full rank; a pure delayed
    copy is exactly FIR-realisable from channel 0 and leaves the projection
    rank-deficient.
    """
    mono = np.asarray(mono, dtype=np.float64).reshape(-1)
    delayed = np.concatenate([np.zeros(_STEREO_DELAY), mono[:-_STEREO_DELAY]])
    rng = np.random.default_rng(_STEREO_SEEDS[role])
    decorr = _STEREO_DECORR * np.sqrt(np.mean(mono ** 2)) * rng.standard_normal(mono.shape)
    return np.stack([mono, _STEREO_GAIN * delayed + decorr], axis=1)


def nonlinear_estimate(target, interferer, fs):
    """ARCHIVE.md's benchmark estimate: tanh(3*mix)/3 plus shaped noise."""
    target = as_2d(target)
    interferer = as_2d(interferer)
    mix = target + interferer
    est = np.tanh(3.0 * mix) / 3.0

    rng = np.random.default_rng(_NOISE_SEED)
    noise = rng.standard_normal(mix.shape)
    b, a = sig.butter(4, _NOISE_CUTOFF_HZ / (fs / 2.0), btype="low")
    noise = sig.lfilter(b, a, noise, axis=0)
    scale = _NOISE_RMS_FRACTION * np.sqrt(np.mean(mix ** 2)) / (np.sqrt(np.mean(noise ** 2)) + 1e-20)
    return est + scale * noise


def decompose(sources, estimate, fs, backend):
    config = DecompositionConfiguration(shade_in_milliseconds=10.0, shade_out_milliseconds=10.0)
    result = decompose_distortion_components(
        source_files=to_backend([as_2d(s) for s in sources], backend),
        estimate_file=to_backend(as_2d(estimate), backend),
        configuration=config,
        sampling_frequency_hz=float(fs),
    )
    return result.waveforms


def save_waveforms(out_dir, prefix, waveforms):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for comp in _COMPONENTS:
        arr = to_np(getattr(waveforms, comp))
        path = out_dir / f"{prefix}__{comp}.npy"
        np.save(path, np.ascontiguousarray(arr, dtype=np.float64))
        names.append(path.name)
    return names


# --------------------------------------------------------------------- main --
def main(tag, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{tag}.json"
    out_wav = results_dir / f"{tag}_wav"
    out_wav.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    report = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "cuda_available": bool(torch.cuda.is_available()),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
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
            "repeats": _REPEATS,
        },
    }
    try:
        report["git_commit"] = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        report["git_dirty"] = bool(subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout.strip())
    except Exception as exc:  # pragma: no cover
        report["git_commit"] = f"<unavailable: {exc}>"

    # ============================================ (A) accuracy vs MATLAB gold =
    print("== (A) accuracy vs MATLAB reference ==", flush=True)
    target_src, fs = sf.read(MATLAB_DIR / "targetSrc.wav")
    interf1, _ = sf.read(MATLAB_DIR / "interfSrc1.wav")
    interf2, _ = sf.read(MATLAB_DIR / "interfSrc2.wav")
    estimate, _ = sf.read(MATLAB_DIR / "targetEstimate.wav")
    gold = {comp: sf.read(MATLAB_DIR / fn)[0] for comp, fn in _VALIDATION_MAP}

    report["accuracy"] = {}
    for backend in ("numpy", "torch"):
        t0 = time.perf_counter()
        waveforms = decompose([target_src, interf1, interf2], estimate, fs, backend)
        elapsed = time.perf_counter() - t0
        save_waveforms(out_wav, f"matlabref_{backend}", waveforms)

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
                ch_entry = {"channel": ch, "gold_std": gold_std, "py_std": float(np.std(py_ch))}
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
                        "gain_error_db": 20.0 * np.log10(rms_ratio / _MATLAB_RESAMPLER_GAIN_OFFSET),
                        "corr_pass": corr > _CORR_THRESHOLD[backend],
                        "gain_pass": abs(rms_ratio / _MATLAB_RESAMPLER_GAIN_OFFSET - 1.0) < _GAIN_TOLERANCE,
                    })
                entry["channels"].append(ch_entry)
                if not ch_entry.get("silent"):
                    print(f"  {backend:5s} {comp:17s} ch{ch}  corr={ch_entry['correlation']:.15f}"
                          f"  rms_ratio={ch_entry['rms_ratio']:.9f}"
                          f"  gain_err={ch_entry['gain_error']:+.3e}", flush=True)
            per_component[comp] = entry
        report["accuracy"][backend] = {"decompose_seconds": elapsed, "components": per_component}

    # ----------------------------------------------------- 8 scores / ratios --
    print("== (A2) perceptual scores ==", flush=True)
    report["scores"] = {}
    for backend in ("numpy", "torch"):
        t0 = time.perf_counter()
        scores = predict_perceptual_evaluation_scores(
            to_backend([as_2d(target_src), as_2d(interf1), as_2d(interf2)], backend),
            to_backend(as_2d(estimate), backend),
            sampling_frequency_hz=float(fs),
        )
        elapsed = time.perf_counter() - t0
        vals = {}
        for field in _SCORE_FIELDS:
            v = getattr(scores, field)
            vals[field] = float(v.item() if isinstance(v, torch.Tensor) else v)
        expected = {**_EXPECTED_SCORES, **_EXPECTED_RATIOS}
        report["scores"][backend] = {
            "seconds": elapsed,
            "values": vals,
            "expected": expected,
            "deviation_from_expected": {k: vals[k] - expected[k] for k in _SCORE_FIELDS},
        }
        for field in _SCORE_FIELDS:
            print(f"  {backend:5s} {field:36s} {vals[field]:.12f}"
                  f"  (test expects ~{expected[field]})", flush=True)

    # ============================ (B/C) exp01 nonlinear estimate: waveforms + timings =
    print("== (B/C) exp01 nonlinear estimate: waveforms + timings ==", flush=True)
    mono_target, fs_db = sf.read(DB_DIR / "exp01_target.wav")
    mono_interf, _ = sf.read(DB_DIR / "exp01_InterfSrc1.wav")
    cases = {
        "mono": ([as_2d(mono_target), as_2d(mono_interf)], None),
        "stereo": ([make_stereo(mono_target, "target"),
                    make_stereo(mono_interf, "interferer")], None),
    }
    for layout in cases:
        srcs = cases[layout][0]
        cases[layout] = (srcs, nonlinear_estimate(srcs[0], srcs[1], fs_db))

    report["timings"] = {}
    report["exp01_component_stats"] = {}
    # numpy before torch, always: MKL/OMP state is process-global and order matters.
    for backend in ("numpy", "torch"):
        for layout in ("mono", "stereo"):
            srcs, est = cases[layout]
            key = f"{backend}_{layout}"

            # warm-up run (JIT/numba compile, allocator warm) -- also the run whose
            # output waveforms are frozen for (B).
            t0 = time.perf_counter()
            waveforms = decompose(srcs, est, fs_db, backend)
            warm = time.perf_counter() - t0
            save_waveforms(out_wav, f"exp01{layout}_{backend}", waveforms)
            report["exp01_component_stats"][key] = {
                comp: {
                    "peak": float(np.max(np.abs(to_np(getattr(waveforms, comp))))),
                    "rms": float(np.sqrt(np.mean(to_np(getattr(waveforms, comp)) ** 2))),
                    "shape": list(to_np(getattr(waveforms, comp)).shape),
                } for comp in _COMPONENTS
            }
            del waveforms

            repeats = []
            for _ in range(_REPEATS):
                t0 = time.perf_counter()
                w = decompose(srcs, est, fs_db, backend)
                repeats.append(time.perf_counter() - t0)
                del w
            report["timings"][key] = {
                "warmup_seconds": warm,
                "repeats": repeats,
                "min": float(min(repeats)),
                "median": float(np.median(repeats)),
                "max": float(max(repeats)),
                "spread_pct": float((max(repeats) - min(repeats)) / min(repeats) * 100.0),
            }
            print(f"  {key:14s} min={min(repeats):.3f}s  median={np.median(repeats):.3f}s"
                  f"  max={max(repeats):.3f}s  spread={report['timings'][key]['spread_pct']:.1f}%"
                  f"  repeats={[round(x, 3) for x in repeats]}", flush=True)

    report["waveform_dir"] = str(out_wav)
    report["waveform_files"] = sorted(p.name for p in out_wav.glob("*.npy"))
    report["total_seconds"] = time.perf_counter() - t_start
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")
    print(f"wrote {len(report['waveform_files'])} waveforms to {out_wav}")
    print(f"total {report['total_seconds']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("tag", help="name this capture, e.g. 'baseline'")
    add_results_dir_argument(parser)
    args = parser.parse_args()
    main(args.tag, resolve_results_dir(args.results_dir))
