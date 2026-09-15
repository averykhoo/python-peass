"""Check that every module under `reference/` really does carry its MATLAB source.

Each transcription module declares `MATLAB_SOURCE` and embeds the complete text
of that `.m` file as fenced comment chunks (see FORMAT.md). This script
concatenates a module's chunks, diffs the result against the real `.m` file, and
reports PASS/FAIL per module. It also collects every `# !!! DEVIATION` block and
prints them as a report, so a deliberate departure from the MATLAB is always
visible and a new one is never silent.

Usage::

    python -m reference.verify_transcription [MATLAB_ROOT]

`MATLAB_ROOT` defaults to the vendored copy under `.scratch/` (see
`default_matlab_root`). That directory is gitignored, so on a fresh clone or in
CI it will not exist; the script then reports every module as SKIPPED, still
prints the deviation report, and exits 0.

Exit status is nonzero only if a module that could be checked FAILED.

Note on line endings: the `.m` files are CRLF, and a Python source file's line
endings are not something the author controls (editors, `git core.autocrlf`).
Both sides are therefore normalized to `\\n` before comparison. That is the only
normalization applied -- indentation, trailing whitespace, blank lines and the
presence or absence of a final newline are all compared exactly.

Imports nothing from `peass`, and does not import the modules it checks: they
are parsed as text, so the check works even if a transcription is mid-edit or
its dependencies are missing.
"""

import argparse
import dataclasses
import difflib
import pathlib
import re
import sys

CHUNK_OPEN = "# >>> MATLAB"
CHUNK_CLOSE = "# <<< MATLAB"
DEVIATION_MARKER = "# !!! DEVIATION"

_MATLAB_SOURCE_RE = re.compile(r'^MATLAB_SOURCE\s*=\s*[\'"]([^\'"]+)[\'"]', re.M)

# Path of the vendored MATLAB PEASS v2.0.1 tree, relative to the repository
# root. Kept relative so nothing here is machine-specific.
DEFAULT_MATLAB_ROOT_PARTS = (
    ".scratch", "original-matlab", "peass_master_22c7fc4e", "v2.0.1",
)


def repository_root():
    """The repo root, i.e. the parent of the `reference/` package."""
    return pathlib.Path(__file__).resolve().parent.parent


def default_matlab_root():
    return repository_root().joinpath(*DEFAULT_MATLAB_ROOT_PARTS)


def reference_root():
    return pathlib.Path(__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class Deviation:
    """One `# !!! DEVIATION` block."""
    module: str
    line: int
    summary: str
    text: str


@dataclasses.dataclass
class Module:
    """A parsed `reference/` module."""
    path: pathlib.Path
    name: str
    matlab_source: str | None
    chunk_lines: list[str]
    deviations: list[Deviation]
    parse_error: str | None = None

    @property
    def is_transcription(self):
        return self.matlab_source is not None


@dataclasses.dataclass
class Result:
    module: Module
    status: str  # "PASS" | "FAIL" | "SKIP"
    detail: str
    diff: str = ""


def _normalize(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_module(path, root=None):
    """Parse one `reference/` .py file without importing it."""
    root = root or reference_root()
    text = _normalize(path.read_text(encoding="utf-8"))
    lines = text.split("\n")

    name_parts = path.relative_to(root.parent).with_suffix("").parts
    name = ".".join(name_parts)

    match = _MATLAB_SOURCE_RE.search(text)
    matlab_source = match.group(1) if match else None

    chunk_lines = []
    inside = [False] * len(lines)
    in_chunk = False
    error = None
    open_at = None

    for index, line in enumerate(lines):
        if line == CHUNK_OPEN:
            if in_chunk:
                error = f"line {index + 1}: nested '{CHUNK_OPEN}'"
                break
            in_chunk = True
            open_at = index + 1
            continue
        if line == CHUNK_CLOSE:
            if not in_chunk:
                error = f"line {index + 1}: '{CHUNK_CLOSE}' without an opening fence"
                break
            in_chunk = False
            continue
        if in_chunk:
            inside[index] = True
            if line == "#":
                chunk_lines.append("")
            elif line.startswith("# "):
                chunk_lines.append(line[2:])
            else:
                error = (f"line {index + 1}: chunk line must be '#' or start with "
                         f"'# ', got {line!r}")
                break
    if error is None and in_chunk:
        error = f"line {open_at}: '{CHUNK_OPEN}' is never closed"

    deviations = _collect_deviations(name, lines, inside)
    return Module(path=path, name=name, matlab_source=matlab_source,
                  chunk_lines=chunk_lines, deviations=deviations, parse_error=error)


def _collect_deviations(module_name, lines, inside_chunk):
    """Every `# !!! DEVIATION` block outside the embedded MATLAB.

    A block is the marker line plus every immediately following comment line at
    the same indentation, which is how a wrapped explanation is written.
    """
    found = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if inside_chunk[index] or not stripped.startswith(DEVIATION_MARKER):
            index += 1
            continue
        indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
        block = [stripped]
        cursor = index + 1
        while cursor < len(lines) and not inside_chunk[cursor]:
            following = lines[cursor]
            if not following.startswith(indent + "#"):
                break
            if following.strip().startswith(DEVIATION_MARKER):
                break
            block.append(following.strip())
            cursor += 1
        text = "\n".join(block)
        # Everything after the marker, on one line, as a stable identity for
        # the block. Leading "!!! DEVIATION:" and comment hashes are stripped.
        body = " ".join(line.lstrip("#").strip() for line in block)
        body = body[len(DEVIATION_MARKER.lstrip("#").strip()):].lstrip(": ").strip()
        found.append(Deviation(module=module_name, line=index + 1,
                               summary=" ".join(body.split()), text=text))
        index = cursor
    return found


def discover_modules(root=None):
    """Every .py file under `reference/`, this script excluded, sorted by path."""
    root = root or reference_root()
    here = pathlib.Path(__file__).resolve()
    modules = []
    for path in sorted(root.rglob("*.py")):
        if path.resolve() == here:
            continue
        if "__pycache__" in path.parts:
            continue
        modules.append(parse_module(path, root=root))
    return modules


def verify_module(module, matlab_root):
    """Diff a module's embedded MATLAB against the real `.m` file."""
    if not module.is_transcription:
        return Result(module, "SKIP", "no MATLAB_SOURCE (not a transcription)")
    if module.parse_error:
        return Result(module, "FAIL", f"malformed chunks: {module.parse_error}")
    if not module.chunk_lines:
        return Result(module, "FAIL", "declares MATLAB_SOURCE but embeds no chunks")

    source_path = pathlib.Path(matlab_root) / module.matlab_source
    if not source_path.is_file():
        return Result(module, "SKIP", f"MATLAB source not found: {source_path}")

    actual = _normalize(source_path.read_bytes().decode("utf-8")).split("\n")
    if module.chunk_lines == actual:
        return Result(module, "PASS",
                      f"{len(actual)} lines match {module.matlab_source}")

    diff = "\n".join(difflib.unified_diff(
        actual, module.chunk_lines,
        fromfile=str(module.matlab_source), tofile=f"{module.name} (embedded)",
        lineterm="",
    ))
    return Result(module, "FAIL",
                  f"embedded MATLAB differs from {module.matlab_source} "
                  f"({len(module.chunk_lines)} embedded lines vs {len(actual)})",
                  diff=diff)


def verify_all(matlab_root=None, root=None):
    matlab_root = pathlib.Path(matlab_root or default_matlab_root())
    return [verify_module(module, matlab_root) for module in discover_modules(root)]


def collect_deviations(root=None):
    found = []
    for module in discover_modules(root):
        found.extend(module.deviations)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "matlab_root", nargs="?", default=None,
        help="root of the MATLAB PEASS v2.0.1 tree "
             "(default: .scratch/original-matlab/peass_master_22c7fc4e/v2.0.1)",
    )
    parser.add_argument("--quiet", action="store_true",
                        help="only print failures and the summary")
    args = parser.parse_args(argv)

    matlab_root = pathlib.Path(args.matlab_root or default_matlab_root())
    print(f"MATLAB source root: {matlab_root}")
    if not matlab_root.is_dir():
        print("  -> not present (it is gitignored); transcription checks will be "
              "skipped.")
    print()

    results = verify_all(matlab_root)
    width = max((len(result.module.name) for result in results), default=0)
    failures = 0
    for result in results:
        if result.status == "FAIL":
            failures += 1
        elif args.quiet:
            continue
        print(f"{result.status:4}  {result.module.name:<{width}}  {result.detail}")
        if result.diff:
            print()
            print(result.diff)
            print()

    print()
    counts = {status: sum(1 for r in results if r.status == status)
              for status in ("PASS", "FAIL", "SKIP")}
    print(f"transcription: {counts['PASS']} passed, {counts['FAIL']} failed, "
          f"{counts['SKIP']} skipped")

    deviations = collect_deviations()
    print()
    print(f"=== deliberate deviations ({len(deviations)}) ===")
    if not deviations:
        print("(none)")
    for deviation in deviations:
        print()
        print(f"{deviation.module}:{deviation.line}")
        for line in deviation.text.split("\n"):
            print(f"    {line}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
