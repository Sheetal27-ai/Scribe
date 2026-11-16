"""
Visual layout cues for Scribe: centred text blocks and horizontal rules.

Both are judged from geometry alone:
  * a block is centred when its left and right gaps to the page edges are
    (nearly) equal and it is clearly narrower than a full text line;
  * a horizontal rule is a long, flat vector line (or hairline rectangle)
    that is not part of a table grid or a figure.
"""
from __future__ import annotations

import re
from typing import List, Sequence

import pymupdf

# --- centring ----------------------------------------------------------- #
_CENTER_TOLERANCE = 0.025   # |left gap - right gap| allowed, as share of page width
_MIN_CENTER_TOLERANCE = 6   # ... but never stricter than this many points
_MAX_CENTERED_WIDTH = 0.85  # wider blocks are justified paragraphs, not centred

# --- rules ---------------------------------------------------------------- #
_RULE_MAX_THICKNESS = 5     # pt; covers thick title bars (NeurIPS: 4pt), not boxes
_RULE_MIN_LENGTH = 0.30     # share of the text width; shorter = table cmidrules etc.
_TABLE_MIN_RULES = 3        # this many rules with identical ends = a table grid
_SAME_END_TOLERANCE = 3     # pt, for "identical ends"
_MERGE_DISTANCE = 3         # pt; double rules collapse into one <hr>

_PAGE_NUMBER = re.compile(r"^\s*(?:\d{1,4}|[ivxlcdm]{1,6})\s*$", re.I)


# ---------------------------------------------------------------------- #
# Centring
# ---------------------------------------------------------------------- #
def is_centered(bbox: pymupdf.Rect, page_rect: pymupdf.Rect) -> bool:
    """
    True when the block sits in the horizontal middle of the page.

    Full-width justified paragraphs are also "centred" by geometry, so
    anything nearly as wide as the text column is excluded.
    """
    width = page_rect.width
    if width <= 0 or bbox.width <= 0:
        return False
    # Estimate the text line width without knowing the margins: a block's
    # smaller side gap is the margin if it touches the column edge; margins
    # are assumed to be at most 20% of the page width each.
    margin = min(bbox.x0 - page_rect.x0, page_rect.x1 - bbox.x1, width * 0.2)
    text_width = width - 2 * margin
    if bbox.width >= text_width * _MAX_CENTERED_WIDTH:
        return False
    left_gap = bbox.x0 - page_rect.x0
    right_gap = page_rect.x1 - bbox.x1
    tolerance = max(_MIN_CENTER_TOLERANCE, width * _CENTER_TOLERANCE)
    return abs(left_gap - right_gap) <= tolerance


def is_page_number(text: str, bbox: pymupdf.Rect, page_rect: pymupdf.Rect) -> bool:
    """A lone number (or roman numeral) in the top or bottom margin."""
    if not _PAGE_NUMBER.match(text):
        return False
    h = page_rect.height
    return bbox.y0 > page_rect.y0 + h * 0.9 or bbox.y1 < page_rect.y0 + h * 0.08


# ---------------------------------------------------------------------- #
# Horizontal rules
# ---------------------------------------------------------------------- #
def find_horizontal_rules(
    page: pymupdf.Page,
    text_width: float,
    exclude: Sequence[pymupdf.Rect] = (),
) -> List[pymupdf.Rect]:
    """
    Returns one thin Rect per horizontal divider on the page, top to bottom.

    Args:
        text_width: width of the text column(s); rules shorter than 30% of
                    it are ignored (table sub-rules, underlines, ticks).
        exclude:    areas whose rules must be ignored (rendered figures,
                    raster images).
    """
    try:
        drawings = page.get_drawings()
    except Exception as e:
        print(f"   [warn] page {page.number + 1}: cannot read drawings for rules: {e}")
        return []

    candidates: List[pymupdf.Rect] = []
    for d in drawings:
        for item in d.get("items", []):
            rect = _item_as_rule(item, d)
            if rect is None or rect.width < text_width * _RULE_MIN_LENGTH:
                continue
            if any(_overlaps(rect, ex) for ex in exclude):
                continue  # part of a figure
            candidates.append(rect)

    candidates = _drop_table_grids(candidates)
    return _merge_close(candidates)


def _item_as_rule(item: tuple, drawing: dict) -> pymupdf.Rect | None:
    """Turns a drawing item into a rule rect if it is a flat line/hairline box."""
    kind = item[0]
    if kind == "l":  # straight line: ("l", p1, p2)
        p1, p2 = item[1], item[2]
        if abs(p1.y - p2.y) > 0.5:
            return None  # not horizontal
        if drawing.get("color") is None:
            return None  # a fill-only path edge, never actually stroked
        half = max(drawing.get("width") or 0.5, 0.5) / 2
        y = (p1.y + p2.y) / 2
        return pymupdf.Rect(min(p1.x, p2.x), y - half, max(p1.x, p2.x), y + half)
    if kind == "re":  # rectangle: ("re", rect, orientation)
        r = pymupdf.Rect(item[1])
        if 0 < r.height <= _RULE_MAX_THICKNESS and (drawing.get("fill") or drawing.get("color")):
            return r
    return None


def _drop_table_grids(rules: List[pymupdf.Rect]) -> List[pymupdf.Rect]:
    """
    Booktabs-style tables draw 3+ rules with the same left and right ends.
    Those are table structure, not section dividers, so the whole group goes.
    """
    keep = []
    for r in rules:
        same_ends = sum(
            1 for o in rules
            if abs(o.x0 - r.x0) <= _SAME_END_TOLERANCE and abs(o.x1 - r.x1) <= _SAME_END_TOLERANCE
        )
        if same_ends < _TABLE_MIN_RULES:
            keep.append(r)
    return keep


def _merge_close(rules: List[pymupdf.Rect]) -> List[pymupdf.Rect]:
    """Collapses double/stacked rules into a single divider."""
    merged: List[pymupdf.Rect] = []
    for r in sorted(rules, key=lambda r: r.y0):
        if merged and r.y0 - merged[-1].y1 <= _MERGE_DISTANCE:
            m = merged[-1]
            merged[-1] = pymupdf.Rect(min(m.x0, r.x0), m.y0, max(m.x1, r.x1), max(m.y1, r.y1))
        else:
            merged.append(r)
    return merged


def _overlaps(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    return a.x0 < b.x1 and a.x1 > b.x0 and a.y0 < b.y1 and a.y1 > b.y0
