"""
Compare two tags produced by measure.py.

Usage:  python benchmarks/compare.py <baseline_tag> <current_tag> [--results-dir DIR]
        e.g.  python benchmarks/compare.py baseline current

Both tags are read from the same directory measure.py wrote them to (see
`_paths.py`: `--results-dir`, then `$PEASS_BENCH_RESULTS`, then
`<repo>/benchmarks/results/`).

Prints:
  1. timing speedups per configuration (baseline_min / current_min)
  2. correlation + gain-error deltas vs the MATLAB gold WAVs
  3. score/ratio deltas (all 8)
  4. worst waveform deviation between the two tags, expressed RELATIVE TO EACH
     COMPONENT'S OWN PEAK -- the convention ARCHIVE.md reports deviations in.

Section 1 is the weakest part of this report and is kept only as a sanity check:
whole-run wall clock cannot resolve a change worth less than ~10% on this
machine (TODO.md ground rule 3). Use `ab.py` for the timing claim.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks._paths import add_results_dir_argument  # noqa: E402
from benchmarks._paths import resolve_results_dir  # noqa: E402


def load(tag, results_dir):
    path = results_dir / f"{tag}.json"
    if not path.exists():
        raise SystemExit(f"no such tag: {path}")
    return json.loads(path.read_text(encoding="utf-8")), results_dir / f"{tag}_wav"


def rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main(tag_a, tag_b, results_dir):
    a, wav_a = load(tag_a, results_dir)
    b, wav_b = load(tag_b, results_dir)

    print(f"results dir = {results_dir}")
    print(f"A = {tag_a}   {a.get('timestamp')}   commit {a.get('git_commit', '?')[:10]}"
          f"{' (dirty)' if a.get('git_dirty') else ''}")
    print(f"B = {tag_b}   {b.get('timestamp')}   commit {b.get('git_commit', '?')[:10]}"
          f"{' (dirty)' if b.get('git_dirty') else ''}")
    if a.get("constants") != b.get("constants"):
        print("!! WARNING: harness constants differ between tags -- comparison is not apples-to-apples")
        print(f"   A: {a.get('constants')}")
        print(f"   B: {b.get('constants')}")

    # ------------------------------------------------------------- timings --
    rule("1. TIMINGS (min of six, seconds) -- coarse; use ab.py below ~10%")
    print(f"{'config':16s} {tag_a:>10s} {tag_b:>10s} {'speedup':>9s}   {'spreadA':>8s} {'spreadB':>8s}")
    keys = [k for k in a.get("timings", {}) if k in b.get("timings", {})]
    for key in keys:
        ta, tb = a["timings"][key], b["timings"][key]
        speed = ta["min"] / tb["min"] if tb["min"] else float("nan")
        print(f"{key:16s} {ta['min']:10.3f} {tb['min']:10.3f} {speed:8.3f}x"
              f"   {ta['spread_pct']:7.1f}% {tb['spread_pct']:7.1f}%")
    for key in set(a.get("timings", {})) ^ set(b.get("timings", {})):
        print(f"{key:16s}  <present in only one tag>")

    # ------------------------------------------------- accuracy vs MATLAB --
    rule("2. ACCURACY VS MATLAB (delta = B - A; corr should not fall, gain_error should not grow)")
    print(f"{'backend/component':30s} {'ch':>2s} {'corr A':>18s} {'corr B':>18s} {'d corr':>11s}"
          f" {'gainerr A':>11s} {'gainerr B':>11s} {'d gainerr':>11s}")
    worst_corr_drop = (0.0, "")
    worst_gain_growth = (0.0, "")
    for backend in sorted(set(a.get("accuracy", {})) & set(b.get("accuracy", {}))):
        for comp, entry_a in a["accuracy"][backend]["components"].items():
            entry_b = b["accuracy"][backend]["components"].get(comp)
            if entry_b is None:
                continue
            for cha, chb in zip(entry_a["channels"], entry_b["channels"]):
                if cha.get("silent") or chb.get("silent"):
                    print(f"{backend + '/' + comp:30s} {cha['channel']:2d}  <silent channel>"
                          f"  std A={cha['py_std']:.3e} B={chb['py_std']:.3e}")
                    continue
                dcorr = chb["correlation"] - cha["correlation"]
                dgain = abs(chb["gain_error"]) - abs(cha["gain_error"])
                flag = ""
                if not chb.get("corr_pass", True) or not chb.get("gain_pass", True):
                    flag = "  << FAILS TEST BOUND"
                print(f"{backend + '/' + comp:30s} {cha['channel']:2d} {cha['correlation']:18.15f}"
                      f" {chb['correlation']:18.15f} {dcorr:+11.2e}"
                      f" {cha['gain_error']:+11.2e} {chb['gain_error']:+11.2e} {dgain:+11.2e}{flag}")
                if dcorr < worst_corr_drop[0]:
                    worst_corr_drop = (dcorr, f"{backend}/{comp} ch{cha['channel']}")
                if dgain > worst_gain_growth[0]:
                    worst_gain_growth = (dgain, f"{backend}/{comp} ch{cha['channel']}")
    print(f"\nworst correlation drop : {worst_corr_drop[0]:+.3e}  ({worst_corr_drop[1] or 'none'})")
    print(f"worst |gain error| growth: {worst_gain_growth[0]:+.3e}  ({worst_gain_growth[1] or 'none'})")

    # --------------------------------------------------------------- scores --
    rule("3. SCORES / RATIOS (delta = B - A)")
    print(f"{'backend':8s} {'field':38s} {tag_a:>18s} {tag_b:>18s} {'delta':>13s}")
    for backend in sorted(set(a.get("scores", {})) & set(b.get("scores", {}))):
        for field, va in a["scores"][backend]["values"].items():
            vb = b["scores"][backend]["values"].get(field)
            if vb is None:
                continue
            print(f"{backend:8s} {field:38s} {va:18.12f} {vb:18.12f} {vb - va:+13.3e}")

    # ------------------------------------------------------------ waveforms --
    rule("4. WAVEFORM DEVIATION (max |B-A| relative to component's own peak in A)")
    files_a = {p.name for p in wav_a.glob("*.npy")} if wav_a.exists() else set()
    files_b = {p.name for p in wav_b.glob("*.npy")} if wav_b.exists() else set()
    common = sorted(files_a & files_b)
    if not common:
        print("no shared .npy files")
    only = (files_a ^ files_b)
    if only:
        print(f"!! files present in only one tag: {sorted(only)}")

    worst = (0.0, "")
    print(f"{'file':46s} {'peak(A)':>12s} {'rel dev':>11s} {'abs dev':>11s} {'corr':>18s}")
    for name in common:
        xa = np.load(wav_a / name).astype(np.float64)
        xb = np.load(wav_b / name).astype(np.float64)
        if xa.shape != xb.shape:
            print(f"{name:46s}  SHAPE MISMATCH {xa.shape} vs {xb.shape}")
            n = min(len(xa), len(xb))
            xa, xb = xa[:n], xb[:n]
            if xa.shape != xb.shape:
                continue
        peak = float(np.max(np.abs(xa)))
        absdev = float(np.max(np.abs(xb - xa)))
        rel = absdev / peak if peak > 0 else float("nan")
        fa, fb = xa.reshape(-1), xb.reshape(-1)
        corr = float(np.corrcoef(fa, fb)[0, 1]) if np.std(fa) > 0 and np.std(fb) > 0 else float("nan")
        print(f"{name:46s} {peak:12.5e} {rel:11.3e} {absdev:11.3e} {corr:18.15f}")
        if rel > worst[0]:
            worst = (rel, name)
    print(f"\nWORST waveform deviation (relative to own peak): {worst[0]:.3e}  ({worst[1] or 'none'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two tags produced by measure.py.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("baseline_tag", help="the frozen reference capture")
    parser.add_argument("current_tag", help="the capture to judge against it")
    add_results_dir_argument(parser)
    args = parser.parse_args()
    main(args.baseline_tag, args.current_tag, resolve_results_dir(args.results_dir))
