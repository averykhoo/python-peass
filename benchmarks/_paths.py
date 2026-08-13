"""Shared location handling for the benchmark scripts.

`measure.py` writes tags here and `compare.py` reads them from here, so the
location lives in exactly one place: if the two ever disagreed, `compare.py`
would silently report "no such tag" for data that exists.

Resolution order (first hit wins):
    1. `--results-dir DIR` on the command line
    2. `$PEASS_BENCH_RESULTS`
    3. `<repo>/benchmarks/results/`

The default is inside the repo but `benchmarks/.gitignore` ignores `results/`,
so tags never enter the working tree's status. A tag is ~20 MB of float64
waveforms; keep the frozen baseline, delete the rest when done.
"""

import argparse
import os
import pathlib

# Derived from this file's location, never hardcoded: benchmarks/_paths.py ->
# <repo>. Nothing under benchmarks/ may contain a machine-specific path.
REPO = pathlib.Path(__file__).resolve().parents[1]

RESULTS_ENV_VAR = "PEASS_BENCH_RESULTS"
DEFAULT_RESULTS_DIR = REPO / "benchmarks" / "results"


def add_results_dir_argument(parser: argparse.ArgumentParser) -> None:
    """Register the `--results-dir` flag shared by measure.py and compare.py."""
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="DIR",
        help=f"where tags are written/read (default: ${RESULTS_ENV_VAR} or {DEFAULT_RESULTS_DIR})",
    )


def resolve_results_dir(cli_value: str | None = None) -> pathlib.Path:
    """CLI flag > environment variable > `<repo>/benchmarks/results/`."""
    if cli_value:
        return pathlib.Path(cli_value).expanduser().resolve()
    from_env = os.environ.get(RESULTS_ENV_VAR)
    if from_env:
        return pathlib.Path(from_env).expanduser().resolve()
    return DEFAULT_RESULTS_DIR
