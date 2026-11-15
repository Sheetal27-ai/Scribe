"""
Caption-triggered figure detection for vector graphics.

LaTeX/matplotlib figures are often drawn with vector paths plus real text
labels instead of an embedded bitmap. PyMuPDF then returns those labels as
dozens of stray one-word text blocks and no image at all. This module finds
such figures by their caption ("Figure 3: ...", "Fig. 2."), works out the
graphic's region above the caption, and reports which text blocks and raster
images belong to it so the caller can render the region as one PNG and drop
the fragments.

Region estimation, per caption:
  1. Search window: horizontally the caption's column, vertically from the
     nearest body paragraph (or previous caption, or top margin) above the
     caption down to the caption's top edge.
  2. Graphic core: union of vector drawings and raster image placements that
     lie inside the window.
  3. Labels: text blocks inside the window that overlap or touch the core,
     grown repeatedly so label rows hanging off the core are picked up too.
     Section headings (bold or larger than body text) and full paragraphs
     are never treated as labels.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pymupdf

# "Figure 3:", "Figure 3.", "Fig. 3:", "FIGURE 3 -" ... at the start of a block.
# Must be at the start: body text that merely mentions "Figure 3" is ignored.
CAPTION_RE = re.compile(r"^\s*(?:figure|fig\.)\s*(\d+)\s*[:.\-–—]", re.IGNORECASE)

_BOLD_HINTS = ("bold", "black", "heavy", "semibold", "demi", "medi")

# Tunables (PDF points; 72pt = 1 inch).
_TOP_MARGIN = 36          # window never starts above this
_LABEL_GAP = 4            # how far a label may sit from the graphic
_PAD = 3                  # breathing room around the rendered crop
_MIN_FIGURE_SIZE = 30     # ignore "figures" smaller than this in either axis
_BODY_MIN_WORDS = 15      # a block with this many words is a paragraph, not a label


@dataclass
class FigureRegion:
    """A vector figure found above a caption."""
    number: str                              # "3" from "Figure 3:"
    rect: pymupdf.Rect                       # area to render
    caption_index: int                       # index of the caption text block
    label_indices: List[int] = field(default_factory=list)   # text blocks inside
    image_rects: List[pymupdf.Rect] = field(default_factory=list)  # rasters inside

    @property
    def has_labels(self) -> bool:
        return bool(self.label_indices)


def body_font_size(doc: pymupdf.Document, sample_pages: int = 10) -> float:
    """Most common font size by character count: the size of body text."""
    sizes: Counter = Counter()
    for page in doc.pages(0, min(sample_pages, len(doc))):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 10.0


def find_vector_figures(
    page: pymupdf.Page,
    text_rects: Sequence[pymupdf.Rect],
    texts: Sequence[str],
    image_rects: Sequence[pymupdf.Rect],
    body_size: float,
) -> List[FigureRegion]:
    """
    Finds figure regions above each caption on the page.

    Args:
        text_rects / texts: the page's text blocks (parallel lists).
        image_rects: placements of raster images already extracted.
        body_size: document body font size (from body_font_size()).

    Returns one FigureRegion per caption that has a graphic above it.
    A caption with nothing drawn above it (e.g. a figure on the previous
    page) is skipped.
    """
    captions = [i for i, t in enumerate(texts) if CAPTION_RE.match(t)]
    if not captions:
        return []

    try:
        drawings = _drawing_rects(page)
    except Exception as e:  # malformed vector content should not kill the page
        print(f"   [warn] page {page.number + 1}: cannot read drawings: {e}")
        drawings = []
    fonts = _block_fonts(page)
    mid_x = page.rect.x0 + page.rect.width / 2

    regions: List[FigureRegion] = []
    claimed: set = set()  # text blocks already given to a figure
    for ci in captions:
        cap = text_rects[ci]
        number = CAPTION_RE.match(texts[ci]).group(1)

        # 1. Search window ------------------------------------------------
        # Column: a caption sitting in one half belongs to that column.
        if cap.x1 <= mid_x + 5:
            x0, x1 = page.rect.x0, mid_x
        elif cap.x0 >= mid_x - 5:
            x0, x1 = mid_x, page.rect.x1
        else:
            x0, x1 = page.rect.x0, page.rect.x1

        top = page.rect.y0 + _TOP_MARGIN
        for j, r in enumerate(text_rects):
            if j == ci or r.y1 > cap.y0 + 1 or r.x1 <= x0 or r.x0 >= x1:
                continue
            is_boundary = (
                len(texts[j].split()) >= _BODY_MIN_WORDS  # body paragraph
                or j in captions                          # previous figure's caption
            )
            if is_boundary:
                top = max(top, r.y1)
        window = pymupdf.Rect(x0, top, x1, cap.y0)
        if window.height < _MIN_FIGURE_SIZE:
            continue

        # 2. Graphic core: drawings + raster images inside the window -------
        core: Optional[pymupdf.Rect] = None
        inside_images = []
        for r in drawings:
            if _within(r, window):
                core = _union(core, r)
        for r in image_rects:
            if _within(r, window):
                core = _union(core, r)
                inside_images.append(r)
        if core is None or core.width < _MIN_FIGURE_SIZE or core.height < _MIN_FIGURE_SIZE:
            continue

        # 3. Grow the core with adjacent labels until nothing new is added --
        labels: List[int] = []
        candidates = [
            j for j, r in enumerate(text_rects)
            if j != ci and j not in claimed and _within(r, window)
            and not _looks_like_text(texts[j], fonts.get(_key(r)), body_size)
        ]
        changed = True
        while changed:
            changed = False
            for j in candidates:
                if j in labels:
                    continue
                if _gap(text_rects[j], core) <= _LABEL_GAP:
                    labels.append(j)
                    core = _union(core, text_rects[j])
                    changed = True

        rect = pymupdf.Rect(core.x0 - _PAD, core.y0 - _PAD, core.x1 + _PAD, core.y1 + _PAD)
        rect &= page.rect
        rect.y1 = min(rect.y1, cap.y0)  # never bleed into the caption
        claimed.update(labels)
        regions.append(FigureRegion(number, rect, ci, labels, inside_images))
    return regions


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _drawing_rects(page: pymupdf.Page) -> List[pymupdf.Rect]:
    """Bounding boxes of vector paths, minus page-wide backgrounds/borders."""
    rects = []
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        # Lines have zero width or height but are still real figure content;
        # only a rect that is a single point (or inverted) is meaningless.
        if r.x1 < r.x0 or r.y1 < r.y0 or (r.width == 0 and r.height == 0):
            continue
        if r.width > page.rect.width * 0.9 and r.height > page.rect.height * 0.9:
            continue  # page background or frame
        rects.append(r)
    return rects


def _block_fonts(page: pymupdf.Page) -> Dict[Tuple[int, int, int, int], Tuple[float, bool]]:
    """Maps a text block's rounded bbox to (largest font size, any bold span)."""
    fonts = {}
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        spans = [s for line in block["lines"] for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        size = max(s["size"] for s in spans)
        bold = any(
            (s["flags"] & 16) or any(h in s["font"].lower() for h in _BOLD_HINTS)
            for s in spans
        )
        fonts[_key(pymupdf.Rect(block["bbox"]))] = (size, bold)
    return fonts


def _looks_like_text(text: str, font: Optional[Tuple[float, bool]], body_size: float) -> bool:
    """True for paragraphs and headings, which must never be swallowed."""
    if len(text.split()) >= _BODY_MIN_WORDS:
        return True
    if font is None:
        return False
    size, bold = font
    return bold or size > body_size + 0.5


def _key(r: pymupdf.Rect) -> Tuple[int, int, int, int]:
    return (round(r.x0), round(r.y0), round(r.x1), round(r.y1))


def _within(r: pymupdf.Rect, window: pymupdf.Rect, tol: float = 1.0) -> bool:
    return (r.x0 >= window.x0 - tol and r.x1 <= window.x1 + tol
            and r.y0 >= window.y0 - tol and r.y1 <= window.y1 + tol)


def _union(a: Optional[pymupdf.Rect], b: pymupdf.Rect) -> pymupdf.Rect:
    # Manual min/max: pymupdf's | operator ignores zero-height line rects.
    if a is None:
        return pymupdf.Rect(b)
    return pymupdf.Rect(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))


def _gap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Distance between two rects (0 if they touch or overlap)."""
    dx = max(b.x0 - a.x1, a.x0 - b.x1, 0)
    dy = max(b.y0 - a.y1, a.y0 - b.y1, 0)
    return max(dx, dy)
