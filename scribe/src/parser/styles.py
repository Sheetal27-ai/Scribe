"""
Span-level typography analysis for Scribe.

Turns PyMuPDF's `page.get_text("dict")` blocks into Markdown that keeps the
document's hierarchy: headings become `#`/`##`/`###`, bold and italic runs
become `**...**` and `*...*`.

Sizes are profiled across the whole document, not per page: a heading size
must map to the same level on every page, and pages that are mostly figures
or references would otherwise skew the baseline.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pymupdf

# PyMuPDF span flag bits (see TEXT_FONT_* in the PyMuPDF docs).
FLAG_SUPERSCRIPT = 1 << 0
FLAG_ITALIC = 1 << 1
FLAG_MONOSPACE = 1 << 3
FLAG_BOLD = 1 << 4

# Many embedded fonts do not set the flag bits, so the name is checked too.
_BOLD_NAME = re.compile(r"bold|black|heavy|semibold|demi|[-,]med(i|ium)?\b|-medi", re.I)
_ITALIC_NAME = re.compile(r"ital|oblique|slant", re.I)
# TeX math fonts mark every variable as italic; that is not emphasis.
_MATH_FONT = re.compile(r"^(?:[A-Z]{6}\+)?(?:CMMI|CMSY|CMEX|CMBSY|CMMIB|MSAM|MSBM|EUFM|EUSM|RSFS|STIX.*Math|.*Math)", re.I)
# "3", "3.1", "3.2.1", "A.2" followed by a title.
_SECTION_NUMBER = re.compile(r"^(?:\d+|[A-Z])(?:\.\d+)*\.?\s+\S")

_HEADING_MAX_WORDS = 15
_SIZE_TOLERANCE = 0.5      # 9.9 / 10.0 / 10.1 are all "10pt"
_MAX_HEADING_LEVEL = 6


# ---------------------------------------------------------------------- #
# Flag helpers
# ---------------------------------------------------------------------- #
def is_bold(span: dict) -> bool:
    return bool(span.get("flags", 0) & FLAG_BOLD) or bool(_BOLD_NAME.search(span.get("font", "")))


def is_math_font(span: dict) -> bool:
    return bool(_MATH_FONT.match(span.get("font", "")))


def is_italic(span: dict) -> bool:
    if is_math_font(span):
        return False
    return bool(span.get("flags", 0) & FLAG_ITALIC) or bool(_ITALIC_NAME.search(span.get("font", "")))


def is_superscript(span: dict) -> bool:
    return bool(span.get("flags", 0) & FLAG_SUPERSCRIPT)


def _size_key(size: float) -> float:
    """Groups sizes that differ only by rendering noise (9.9 ≈ 10.0 ≈ 10.1)."""
    return round(size * 2) / 2


# ---------------------------------------------------------------------- #
# Document profile
# ---------------------------------------------------------------------- #
@dataclass
class StyleProfile:
    body_size: float
    # size key -> heading level, largest size first (17pt -> 1, 12pt -> 2, ...)
    heading_levels: Dict[float, int] = field(default_factory=dict)

    def level_for_size(self, size: float) -> Optional[int]:
        return self.heading_levels.get(_size_key(size))

    @property
    def body_heading_base(self) -> int:
        """Level used for body-sized numbered headings ("3.1 ...")."""
        return max(len(self.heading_levels), 1)


def build_profile(doc: pymupdf.Document) -> StyleProfile:
    """
    Finds the body text size (most characters) and the distinct heading
    sizes above it, then ranks heading sizes into levels.
    """
    char_count: Counter = Counter()
    heading_sizes: Counter = Counter()
    blocks_by_page = []
    for page in doc:
        try:
            blocks = page.get_text("dict")["blocks"]
        except Exception as e:
            print(f"   [warn] page {page.number + 1}: cannot read text styles: {e}")
            blocks = []
        blocks_by_page.append(blocks)
        for block in blocks:
            for span in _spans(block):
                char_count[_size_key(span["size"])] += len(span["text"].strip())

    body = char_count.most_common(1)[0][0] if char_count else 10.0

    for blocks in blocks_by_page:
        for block in blocks:
            info = _block_info(block)
            if info and _is_size_heading(info, body):
                heading_sizes[_size_key(info.size)] += 1

    levels = {
        size: min(i + 1, _MAX_HEADING_LEVEL)
        for i, size in enumerate(sorted(heading_sizes, reverse=True))
    }
    return StyleProfile(body_size=body, heading_levels=levels)


# ---------------------------------------------------------------------- #
# Block -> Markdown
# ---------------------------------------------------------------------- #
@dataclass
class _BlockInfo:
    text: str            # plain text, whitespace-normalised
    size: float          # dominant font size (by characters)
    all_bold: bool       # every letter is bold
    horizontal: bool     # all lines run left-to-right
    color: Optional[str] = None  # dominant non-default colour (CSS), if any


def block_to_markdown(block: dict, profile: StyleProfile) -> Tuple[str, str]:
    """
    Converts one `get_text("dict")` text block.

    Returns (plain_text, markdown). The plain text is used for layout logic
    (caption detection, reading order); the Markdown goes to the translator.
    Returns ("", "") for empty blocks.
    """
    info = _block_info(block)
    if info is None:
        return "", ""

    level = heading_level(info, profile)
    if level:
        # Headings carry no emphasis markup ("## **Intro**" is redundant),
        # but keep a non-default colour.
        return info.text, f"{'#' * level} {_color_wrap(_escape(info.text), info.color)}"
    return info.text, _inline_markdown(block)


def heading_level(info: _BlockInfo, profile: StyleProfile) -> Optional[int]:
    """Markdown heading level for a block, or None for body text."""
    if not info.horizontal or len(info.text.split()) > _HEADING_MAX_WORDS:
        return None
    if _is_size_heading(info, profile.body_size):
        return profile.level_for_size(info.size)
    # Body-sized bold headings are only trusted when numbered ("3.1 Encoder"),
    # otherwise bold author names or run-in labels would become headings.
    if info.all_bold and _SECTION_NUMBER.match(info.text) and not info.text.endswith((".", ":")):
        depth = info.text.split()[0].rstrip(".").count(".") + 1
        return min(profile.body_heading_base + depth - 1, _MAX_HEADING_LEVEL)
    return None


def _is_size_heading(info: _BlockInfo, body: float) -> bool:
    """Larger than body text and shaped like a heading."""
    if not info.horizontal or not info.text:
        return False
    if info.size < body + _SIZE_TOLERANCE + 0.5:
        return False
    if len(info.text.split()) > _HEADING_MAX_WORDS or info.text.endswith("."):
        return False
    # Bold, or clearly larger (titles in Word documents are often not bold).
    return info.all_bold or info.size >= body * 1.3


_INHERIT = object()  # colour placeholder for whitespace runs


def _inline_markdown(block: dict) -> str:
    """
    Joins spans into one line. Bold/italic runs are wrapped in ** / *, and
    runs in a non-default colour in <span style="color: rgb(...)">.
    """
    runs: List[List] = []  # [text, bold, italic, colour]
    for li, line in enumerate(block.get("lines", [])):
        if li and runs:
            runs.append([" ", None, None, _INHERIT])  # line break -> space
        for span in line.get("spans", []):
            text = span.get("text", "")
            if not text:
                continue
            if is_superscript(span) or is_math_font(span) or not any(c.isalnum() for c in text):
                style = (None, None)  # inherit neighbour style: no marker churn
            else:
                style = (is_bold(span), is_italic(span))
            # Colour is a real property of every glyph, punctuation included,
            # so only pure whitespace borrows it from its neighbours.
            colour = span_color(span) if text.strip() else _INHERIT
            runs.append([text, *style, colour])

    # Style-neutral runs (spaces, punctuation, math, footnote marks) join a
    # style only when sandwiched by it: "**Q, K and V**" stays one run, but a
    # trailing "∗" after a bold name stays outside: "**Ashish Vaswani**∗".
    for i, run in enumerate(runs):
        if run[1] is None:
            before = next(((r[1], r[2]) for r in reversed(runs[:i]) if r[1] is not None), None)
            after = next(((r[1], r[2]) for r in runs[i + 1:] if r[1] is not None), None)
            run[1], run[2] = before if before == after and before else (False, False)
        if run[3] is _INHERIT:
            before = next((r[3] for r in reversed(runs[:i]) if r[3] is not _INHERIT), None)
            after = next((r[3] for r in runs[i + 1:] if r[3] is not _INHERIT), None)
            run[3] = before if before == after else None

    # Group by colour first so a red sentence with one bold word becomes one
    # <span> with ** inside, not three separate spans.
    groups: List[Tuple[Optional[str], List[List]]] = []
    for run in runs:
        if groups and groups[-1][0] == run[3]:
            groups[-1][1].append(run)
        else:
            groups.append((run[3], [run]))

    out = []
    for colour, group in groups:
        text = _emphasis_markdown(group)
        out.append(_color_wrap(text, colour))
    return _tidy("".join(out))


def _emphasis_markdown(runs: List[List]) -> str:
    """Merges same-style neighbours and wraps them in ** / * / ***."""
    merged: List[List] = []
    for text, bold, italic, _ in runs:
        if merged and merged[-1][1] == bold and merged[-1][2] == italic:
            merged[-1][0] += text
        else:
            merged.append([text, bold, italic])

    out = []
    for text, bold, italic in merged:
        marker = "***" if bold and italic else "**" if bold else "*" if italic else ""
        # Italicising a lone letter is almost always a variable, not emphasis.
        if marker == "*" and sum(c.isalnum() for c in text) < 2:
            marker = ""
        out.append(_wrap(_escape(text), marker))
    # Runs keep their own whitespace, so "Vaswani∗" is not split into "Vaswani ∗".
    return "".join(out)


# ---------------------------------------------------------------------- #
# Colour
# ---------------------------------------------------------------------- #
def span_color(span: dict) -> Optional[str]:
    """
    CSS colour for a span, or None for "default text colour".

    PyMuPDF gives sRGB as one int (0xRRGGBB). Near-black is treated as the
    default so the viewer's theme colour is used (keeps dark mode readable);
    near-white is dropped too, as it would vanish on a light background.
    """
    value = span.get("color", 0)
    if not isinstance(value, int):
        return None
    r, g, b = (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF
    if max(r, g, b) < 0x40 or min(r, g, b) > 0xF0:
        return None
    return f"rgb({r}, {g}, {b})"


def _color_wrap(text: str, colour: Optional[str]) -> str:
    """Wraps text in a coloured <span>, keeping outer whitespace outside."""
    core = text.strip()
    if not colour or not core:
        return text
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):]
    return f'{lead}<span style="color: {colour};">{core}</span>{trail}'


def _wrap(text: str, marker: str) -> str:
    """Wraps text in a marker, keeping surrounding whitespace outside it
    ("** bold **" is not valid Markdown emphasis)."""
    core = text.strip()
    if not marker or not core:
        return text
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):]
    return f"{lead}{marker}{' '.join(core.split())}{marker}{trail}"


def _escape(text: str) -> str:
    """Escapes literal asterisks so they are not read as emphasis."""
    return text.replace("*", r"\*")


def _tidy(text: str) -> str:
    text = " ".join(text.split())
    # No space before punctuation that followed a closing marker: "**Encoder:** The"
    return re.sub(r"(\*+)\s+([,.;:)\]])", r"\1\2", text)


def _block_info(block: dict) -> Optional[_BlockInfo]:
    if block.get("type", 0) != 0:
        return None
    lines = block.get("lines", [])
    text = " ".join(
        "".join(span.get("text", "") for span in line.get("spans", [])) for line in lines
    )
    text = " ".join(text.split())
    if not text:
        return None

    sizes: Counter = Counter()
    colours: Counter = Counter()
    letters = bold_letters = 0
    for span in _spans(block):
        n = sum(c.isalpha() for c in span["text"])
        sizes[span["size"]] += len(span["text"].strip())
        colours[span_color(span)] += len(span["text"].strip())
        if not is_superscript(span):
            letters += n
            bold_letters += n if is_bold(span) else 0
    size = sizes.most_common(1)[0][0] if sizes else 0.0
    horizontal = all(abs(line.get("dir", (1, 0))[1]) < 1e-3 and line.get("dir", (1, 0))[0] > 0 for line in lines)
    colour = colours.most_common(1)[0][0] if colours else None
    return _BlockInfo(text, size, letters > 0 and bold_letters == letters, horizontal, colour)


def _spans(block: dict):
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("text", "").strip():
                yield span
