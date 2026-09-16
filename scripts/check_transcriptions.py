"""Validate the transcriptions in papers/text/*.md against what they claim to be.

These files are not mechanically reproducible -- they were transcribed from the
page images by hand (see papers/README.md), so re-running a tool cannot prove
them right. What a tool CAN do is refuse the failure modes that would otherwise
pass unnoticed: a page silently missing from the middle of a document, a stray
glyph from the old broken extraction surviving into the new file, or a LaTeX
delimiter left open so every equation after it renders as prose.

    python scripts/check_transcriptions.py

Exits non-zero on any failure, and names the file and page.
"""

from __future__ import annotations

import pathlib
import re
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXT = ROOT / "papers" / "text"

# Page counts are the PDFs' own, checked with pdftotext -f/-l at transcription
# time. A transcription that no longer covers its source exactly is a defect.
EXPECTED_PAGES = {
    "706": 133,
    "621984": 15,
    "622101": 9,
    "emiya2011": 13,
    "vincent_LVA12": 9,
    "2010-09-27_LVA_PEASS_emiya": 2,
}

PAGE_MARK = re.compile(r"<!-- page (\d+) -->")

# Signatures of the pdftotext extraction these files replaced. Any of them
# reappearing means someone has pasted old output back in. See papers/README.md
# for what each one actually stood for.
BROKEN = [
    ("digit{digit (en dash lost to an OT1 slot)", re.compile(r"\d\{\d")),
    ("C0 control character", re.compile(r"[\x00-\x09\x0b-\x1f]")),
    ("DEL / OT1 dieresis slot", re.compile(r"\x7f")),
]


def bad_codepoints(text: str) -> list[str]:
    """Coptic letters and combining marks: the broken ToUnicode map's calling card."""
    out = []
    for ch in sorted(set(text)):
        if 0x0300 <= ord(ch) <= 0x036F:
            out.append("combining mark U+%04X" % ord(ch))
        elif "COPTIC" in unicodedata.name(ch, ""):
            out.append("%s U+%04X" % (unicodedata.name(ch), ord(ch)))
    return out


def unescaped_dollars(body: str) -> int:
    """Count $ that actually delimit maths -- \\$ is a literal dollar, not a delimiter."""
    stripped = re.sub(r"\\\$", "", body)          # drop escaped dollars first
    stripped = re.sub(r"\$\$.*?\$\$", "", stripped, flags=re.S)   # then display maths
    return len(re.findall(r"\$", stripped))


def unbalanced_left_right(body: str) -> tuple[int, int]:
    r"""\left/\right must pair. \leftarrow is a different macro and must not count."""
    left = len(re.findall(r"\\left(?![a-zA-Z])", body))
    right = len(re.findall(r"\\right(?![a-zA-Z])", body))
    return left, right


def check(stem: str, expected: int) -> list[str]:
    path = TEXT / (stem + ".md")
    if not path.exists():
        return ["%s.md: missing" % stem]

    text = path.read_text(encoding="utf-8")
    body = text.split("\n---\n", 1)[-1]
    faults = []

    pages = [int(m.group(1)) for m in PAGE_MARK.finditer(text)]
    if not pages:
        faults.append("no page markers at all")
    else:
        if sorted(pages) != list(range(1, expected + 1)):
            missing = sorted(set(range(1, expected + 1)) - set(pages))
            extra = sorted(set(pages) - set(range(1, expected + 1)))
            dupes = sorted({p for p in pages if pages.count(p) > 1})
            faults.append(
                "page markers do not cover 1..%d exactly (missing=%s extra=%s duplicated=%s)"
                % (expected, missing or "-", extra or "-", dupes or "-"))
        if pages != sorted(pages):
            faults.append("page markers are out of order")

    for label, pattern in BROKEN:
        m = pattern.search(body)
        if m:
            faults.append("old-extraction artefact present (%s) at offset %d" % (label, m.start()))

    bad = bad_codepoints(body)
    if bad:
        faults.append("broken-ToUnicode codepoints present: " + ", ".join(bad[:4]))

    if len(re.findall(r"\$\$", body)) % 2:
        faults.append("odd number of $$ display-maths delimiters")
    if unescaped_dollars(body) % 2:
        faults.append("odd number of inline $ delimiters -- a maths span is left open")

    left, right = unbalanced_left_right(body)
    if left != right:
        faults.append("\\left/\\right mismatch: %d vs %d" % (left, right))

    return ["%s.md: %s" % (stem, f) for f in faults]


def main() -> int:
    problems = []
    for stem, expected in EXPECTED_PAGES.items():
        faults = check(stem, expected)
        problems += faults
        path = TEXT / (stem + ".md")
        size = path.stat().st_size if path.exists() else 0
        print("%-32s %3d pages  %7d bytes  %s"
              % (stem, expected, size, "FAIL" if faults else "ok"))

    if problems:
        print("\n%d problem(s):" % len(problems), file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1
    print("\nall %d transcriptions intact" % len(EXPECTED_PAGES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
