"""Extract papers/*.pdf to papers/text/*.txt, one .txt per PDF.

Why this exists: the six PDFs under papers/ are the primary sources for the port
in peass/ and reference/, and a PDF does not grep. This writes a plain-text copy
of each one that an LLM -- or a human with ripgrep -- can read directly. Read the
.txt for prose; open the PDF for a figure, a table, or an equation.

This is text extraction, not OCR. Every one of the six PDFs is digital-born and
carries a real text layer, so nothing here renders page images or calls a vision
model. Re-running is cheap and byte-for-byte deterministic.

Requires `pdftotext` (poppler-utils); it ships with Git for Windows at
/mingw64/bin/pdftotext.

    python scripts/extract_text.py            # re-extract everything
    python scripts/extract_text.py --check    # audit what is on disk, extract nothing

Five of the six PDFs extract cleanly and need none of what follows. All of it is
about papers/706.pdf, Dau's 1996 Oldenburg thesis, which is a TeX document from
before Unicode and leaks its font encoding into the output. Four traps, every one
of which fails *silently* -- each produces text that reads plausibly until you
look closely:

1.  `-layout`, the mode that looks obviously right, is wrong for every document
    here. It reconstructs the printed page by glyph position, so on a two-column
    paper it lays both columns side by side on each output line and every
    sentence is cut in half by the gutter -- 59% to 68% of the body lines of
    four of these papers, and 9% to 11% of the other two. On the thesis it goes
    further and merges the columns character by character, turning 14 lines into
    "tIcnihcacirnoancttdheiertiisotthnircseiswshi". `-raw` emits glyphs in
    content-stream order instead, which is reading order: one column through,
    then the next, and a gutter break rate of 0.0% on all six. The cost is table
    and figure alignment, which was unusable in text form either way.

2.  pdftotext defaults to Latin-1, so "Universität Oldenburg" lands as a lone
    0xE4 byte and the file is not valid UTF-8. `-enc UTF-8` fixes it.

3.  Reading the extraction as *text* destroys the `fl` ligature. TeX's OT1 font
    encoding puts `fl` at character 13, which is also CR, so every layer that
    helpfully normalises line endings -- Python's own open() in its default text
    mode, an editor, a git text-mode checkout -- rewrites that glyph as a line
    break, and "reflects" silently becomes "re\nects": a word still on the page
    that can never be grepped again. pdftotext itself is innocent and no -eol
    flag changes this; the 107 bare CRs in the thesis survive all three. So the
    bytes go from pdftotext's stdout straight into decode(), untranslated, and
    every CR still standing at that point is a ligature. By the time the file is
    written it is an ordinary "fl" and nothing downstream can eat it.

4.  Those OT1 slots collide with TeX's maths fonts, and pdftotext has discarded
    the font that would tell them apart. Character 14 is `ffi` in running text
    ("difficult"), delta in an equation, and a degree sign after a number
    ("360°"); character 12 is `fi` but also pdftotext's own page-break form feed;
    character 25 is `ß` in the German front matter but pi in the appendix. Each
    slot below therefore carries its own rule, and each rule was read off the
    document's own occurrences -- tabulated by neighbouring character, all 1149
    of them -- rather than taken from a font table on faith. Where the document
    does not settle it, UNRESOLVED marks the spot instead of guessing.

Equations and plot legends in the thesis survive as an approximation and should
be read from the PDF. Its prose is sound. See papers/README.md.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "papers"
OUT = SRC / "text"

CR = chr(13)
LF = chr(10)
FF = chr(12)

# Documents typeset in TeX whose OT1 font slots leak through as control
# characters; see traps 3 and 4. Only these get repair_ot1() -- the other five
# PDFs extract with zero control characters and need no repair, and running a
# repair over a document that does not need it can only invent damage.
TEX_OT1 = {"706"}

# Slots whose surroundings do NOT settle which font they came from: plot-legend
# marker glyphs and a couple of operators, 28 occurrences across the document.
# They get a visible placeholder rather than a plausible guess, because a wrong
# symbol silently printed inside an equation is worse than a gap that is
# obviously a gap and greps as one.
UNRESOLVED = {2, 3, 4, 7, 26}

# Slots with a single unambiguous maths reading, confirmed against their own
# surroundings in the thesis rather than assumed from an encoding table:
#   1/(sigma sqrt(2 pi)) e^-(...)  fixes 25 as pi and 27 as sigma
#   Phi(x) = integral phi(x') dx'  fixes 8 as Phi and 30 as phi
#   0 <= fmod <= delta-f / 2       fixes 20 as <= and 21 as >=
#   P(e|S) = prod_nu P(e_nu|S)     fixes 5 as prod and 23 as nu
#   "the mean mu and variance"     fixes 22 as mu
#   "(m)" and "(e_nu - s_nu)^2"    fix 28 and 29 as the two parentheses
MATH = {1: "Δ", 5: "Π", 8: "Φ", 17: "≡", 20: "≤",
        21: "≥", 22: "μ", 23: "ν", 27: "σ", 28: "(",
        29: ")", 30: "φ"}

# OT1 position 127 is a combining dieresis that precedes its vowel, separated by
# whatever whitespace happened to fall there.
UMLAUT = chr(127)
VOWELS = "aouAOU"
UMLAUT_RE = re.compile(re.escape(UMLAUT) + r"\s*([" + VOWELS + "])")

# TeX renders ``quoted'' through OT1 position 92, which arrives as a backslash.
# All 247 backslashes in the thesis are followed by an alphanumeric -- every one
# opens a quotation and none is a literal backslash -- which is what makes this
# safe to rewrite. The closing quote arrives as an ordinary " and needs nothing.
QUOTE_OPEN = re.compile(r"\\(?=[0-9A-Za-z])")

# Interleaved columns (trap 1) show up as very long letter runs, but so do two
# harmless things: German compounds ("Sonderforschungsbereich", 23) and words
# that `-raw` joins across a line break ("amplitudemodulationdetection", 28).
# What separates them is that merging two columns puts a capital inside the run,
# and that the runs it makes are far longer -- the real ones reach 60-plus.
# Either signal alone gives false positives; together they separate the two cases
# cleanly on all six documents.
RUN = re.compile(r"[A-Za-z]{40,}|[A-Za-z]{10,}[A-Z][a-z]*[A-Za-z]{10,}")

# Anything outside this set is a control character that should not have survived
# repair. FF is kept deliberately: what is left of it is the page break.
ALLOWED_CONTROL = {LF, FF}

# A surviving page break sits at the start of a line; anything else is a `fi`
# that repair_form_feeds() never got to.
INLINE_FF = re.compile("(?<![" + LF + FF + "])" + re.escape(FF))

# Two blocks of text separated by a run of spaces on one line: the signature of
# a column gutter reconstructed by `-layout`.
GUTTER = re.compile(r"\S {3,}\S")


def repair_form_feeds(text: str) -> str:
    """Resolve slot 12, which is `fi` and pdftotext's page break at once.

    Neighbours settle most of it: a form feed inside or at the start of a word
    ("modi<12>ed", "bandpass-<12>ltered", " <12>tting") is the ligature, 542
    times, while one followed by a digit is a page break turning over onto a page
    number, 108 times. What neither settles is a form feed that follows a newline
    and precedes a letter, 62 times: that is "\\n<12>ltering" continuing a word
    at a line start, but it is equally "\\n<12>Abstract" opening a new page.

    So those are decided against the document's own vocabulary, built from the
    occurrences that were never in doubt. "fi" + "ltering" is a word this thesis
    uses elsewhere; "fi" + "Abstract" is not. That splits the 62 39/23, and the
    23 are all page-opening headings (Abstract, Contents, Danksagung, Lebenslauf)
    while the 39 are all fi-words (figure, filterbank, finally, findings). It
    also brings the page-break total to 131 against a 133-page document, which is
    the independent check that the split is right: the last page has no trailing
    break.
    """
    # Pass one: the unambiguous ligatures. A form feed after a newline or after
    # another form feed is excluded -- doubled form feeds are a page break, and
    # letting one through here poisons the vocabulary with "fiModeling".
    lig = re.compile("(?<=[^" + LF + FF + "])" + re.escape(FF) + r"(?=[A-Za-z])")
    resolved = lig.sub("fi", text)
    vocab = {w.lower() for w in re.findall(r"[A-Za-z]+", resolved)}

    # Pass two: the ambiguous ones, decided by that vocabulary.
    ambiguous = re.compile("(?<=[" + LF + FF + "])" + re.escape(FF) + r"(?=([A-Za-z]+))")
    return ambiguous.sub(
        lambda m: "fi" if ("fi" + m.group(1)).lower() in vocab else FF, resolved)


def repair_ot1(text: str) -> str:
    """Map leaked OT1 font slots back to the characters they stand for."""
    # Umlauts first, and as a regex rather than character by character: the
    # dieresis is separated from its vowel by whatever whitespace happened to
    # fall there, a space in "Universit<127> at" but a line break in
    # "G<127>\nottingen", and that whitespace has to be consumed along with the
    # accent or the word still will not grep.
    text = UMLAUT_RE.sub(
        lambda m: unicodedata.normalize("NFC", m.group(1) + "̈"), text)
    text = repair_form_feeds(text)

    out = []
    for i, ch in enumerate(text):
        code = ord(ch)
        prev = text[i - 1] if i else ""
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if code in UNRESOLVED:
            out.append("[?%d]" % code)
        elif code == 11:
            # Every one of the 256 follows a letter, including word-final ones
            # ("roll-off", "off-frequency"). None is the alpha of that slot.
            out.append("ff" if prev.isalpha() else "α")
        elif code == 13:
            # All 107 precede a letter, at a word start ("fluctuation") or inside
            # one ("reflects"). None is the gamma of that slot.
            out.append("fl" if nxt.isalpha() else "γ")
        elif code == 14:
            # Three readings, separated by what sits either side: letters both
            # sides is the ligature ("difficult"), a preceding digit is the
            # degree sign ("360°"), and the rest are maths.
            out.append("ffi" if prev.isalpha() and nxt.isalpha()
                       else "°" if prev.isdigit() else "δ")
        elif code == 15:
            # No occurrence has letters on both sides, so the ffl ligature never
            # actually fires here; all 19 are the epsilon. The branch stays for
            # the next TeX document that needs it.
            out.append("ffl" if prev.isalpha() and nxt.isalpha() else "ε")
        elif code == 25:
            # Eszett in the German front matter ("großes", "schloß ich"), pi in
            # the appendix ("e^{iπB}"). All three occurrences that touch a letter
            # are separated by case: the eszett sits between lower-case letters,
            # the pi is flanked by a single-letter variable and a capital.
            out.append("ß" if prev.islower() and not nxt.isupper() else "π")
        elif code in MATH:
            out.append(MATH[code])
        else:
            out.append(ch)
    return QUOTE_OPEN.sub("“", "".join(out))


def extract(pdf: Path, dest: Path) -> None:
    """Run pdftotext, repair, and write dest."""
    proc = subprocess.run(
        ["pdftotext", "-raw", "-enc", "UTF-8", "-eol", "unix", str(pdf), "-"],
        check=True,
        capture_output=True,
    )
    # decode(), not a text-mode read: see trap 3. With -eol unix every line
    # ending is already LF, so a surviving CR can only be the `fl` glyph.
    text = proc.stdout.decode("utf-8")
    # Trap 3 is the one failure here that leaves nothing behind to audit: if the
    # CRs are translated away upstream of this line, the ligatures are gone
    # before repair runs and every later check still passes on text that reads
    # perfectly. So refuse outright. A TeX document long enough to be worth
    # extracting always contains some `fl`.
    if pdf.stem in TEX_OT1 and CR not in text:
        raise RuntimeError(
            "%s: no CR in pdftotext output -- the `fl` ligatures were translated "
            "away before repair could run. Something is reading these bytes as "
            "text with universal newlines." % pdf.stem)
    if pdf.stem in TEX_OT1:
        text = repair_ot1(text)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8", newline=LF)


def audit(text: str) -> list[str]:
    """Return the reasons this extraction is unfit to read, if any."""
    faults = []
    stray = {c for c in text if ord(c) < 32 or ord(c) == 127} - ALLOWED_CONTROL
    if stray:
        faults.append("stray control characters: " + ", ".join(
            "chr(%d)" % ord(c) for c in sorted(stray, key=ord)))
    runs = RUN.findall(text)
    if runs:
        faults.append("%d over-long letter run(s), columns look interleaved "
                      "(e.g. %.30s)" % (len(runs), runs[0]))
    # A page break only ever follows a line break or another page break. One
    # sitting anywhere else is an unrepaired `fi`, still eating the letter it
    # stands for -- and because FF is otherwise a legal character here, nothing
    # else in this audit would notice.
    inline = len(INLINE_FF.findall(text))
    if inline:
        faults.append("%d form feed(s) not at a page boundary -- `fi` ligatures "
                      "left unrepaired" % inline)
    # Column geometry reconstructed instead of reading order: two columns land
    # side by side and every sentence is cut in half by the gutter. Measured
    # across these six, `-raw` scores 0.0% and `-layout` 8.9% to 67.5%, so
    # anything above a couple of percent means the mode has regressed.
    body = [ln for ln in text.split(LF) if len(ln.strip()) > 25]
    gutter = sum(1 for ln in body if GUTTER.search(ln))
    if body and 100 * gutter / len(body) > 2.0:
        faults.append("%.1f%% of body lines are split by a column gutter"
                      % (100 * gutter / len(body)))
    if len(text.strip()) < 500:
        faults.append("extraction is empty or near-empty")
    return faults


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="audit the .txt files already on disk, extract nothing")
    args = ap.parse_args()

    pdfs = sorted(SRC.glob("*.pdf"))
    if not pdfs:
        print("no PDFs under %s" % SRC, file=sys.stderr)
        return 1

    problems = []
    for pdf in pdfs:
        dest = OUT / (pdf.stem + ".txt")
        if args.check:
            if not dest.exists():
                problems.append("%s: not extracted" % pdf.stem)
                print("MISS  %s" % pdf.stem)
                continue
        else:
            extract(pdf, dest)

        text = dest.read_text(encoding="utf-8")
        faults = audit(text)
        problems += ["%s: %s" % (pdf.stem, f) for f in faults]
        print("%-32s %6d words%s"
              % (pdf.stem, len(text.split()),
                 "  <-- " + "; ".join(faults) if faults else ""))

    if problems:
        print("\n%d problem(s):" % len(problems), file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1
    print("\nall %d documents extracted cleanly" % len(pdfs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
