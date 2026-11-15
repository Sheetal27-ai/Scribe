"""
Layout-aware PDF extraction for Scribe.

Walks a PDF page by page, saves every embedded image to disk, and returns
text blocks and images merged into natural reading order (top-to-bottom,
column-by-column) so the Markdown output keeps figures next to the text
that surrounds them.

Uses PyMuPDF (`import pymupdf`; the legacy name `fitz` is the same library).
"""
from __future__ import annotations

import html
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pymupdf

# Block type codes returned by page.get_text("blocks")
_TEXT_BLOCK = 0
_IMAGE_BLOCK = 1

# Formats that Markdown viewers / browsers can render directly.
# Anything else (jpx, jbig2, tiff, CMYK data, ...) is re-encoded to PNG.
_WEB_SAFE_EXTS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

# Display width limits for <img width="..%">, as a percentage of page width.
_MIN_WIDTH_PCT = 15
_DEFAULT_WIDTH_PCT = 60


@dataclass
class TextElement:
    """A text block with its position on the page."""
    page_num: int
    bbox: pymupdf.Rect
    text: str

    def to_markdown(self) -> str:
        return self.text


@dataclass
class ImageElement:
    """One placement of a saved image on the page."""
    page_num: int
    bbox: pymupdf.Rect
    path: str          # path relative to the Markdown file, forward slashes
    caption: str
    width_pct: int     # displayed width as a percentage of the page width

    def to_markdown(self) -> str:
        # HTML instead of ![alt](path) so the figure keeps its size relative
        # to the page. Markdown renderers (GitHub, VS Code, Obsidian) pass
        # <img> through, and a percentage width scales with the viewer.
        return (
            f'<img src="{html.escape(self.path, quote=True)}" '
            f'width="{self.width_pct}%" '
            f'alt="{html.escape(self.caption, quote=True)}">'
        )


def _content_rect(text_elements: List["TextElement"], page_rect: pymupdf.Rect) -> pymupdf.Rect:
    """
    Horizontal extent of the page's text column(s), i.e. the page minus its
    margins. A Markdown viewer's 100% is its text column, so this is the
    right reference width. Falls back to the full page for image-only pages.
    """
    if not text_elements:
        return page_rect
    x0 = min(el.bbox.x0 for el in text_elements)
    x1 = max(el.bbox.x1 for el in text_elements)
    if x1 - x0 < page_rect.width * 0.3:  # a lone caption, not a real column
        return page_rect
    return pymupdf.Rect(x0, page_rect.y0, x1, page_rect.y1)


def _width_percent(bbox: pymupdf.Rect, content_rect: pymupdf.Rect) -> int:
    """
    Image width as a share of the text-column width, rounded to a multiple
    of 5 and clamped so small figures stay legible and nothing overflows.
    Falls back to a sensible default if the geometry is unusable.
    """
    if content_rect.width <= 0 or bbox.width <= 0:
        return _DEFAULT_WIDTH_PCT
    pct = bbox.width / content_rect.width * 100
    pct = int(round(pct / 5) * 5)
    return max(_MIN_WIDTH_PCT, min(100, pct))


PageElement = Union[TextElement, ImageElement]


class LayoutExtractor:
    """
    Extracts text and images from a PDF in reading order.

    Args:
        pdf_path:   PDF to read.
        output_dir: Directory the Markdown file will be written to. Images go
                    into `<output_dir>/<image_subdir>/` so the relative links
                    in the Markdown resolve correctly.
        image_subdir: Folder name for saved images (default "output_images").
        min_image_size: Images whose width or height (in pixels) is below
                    this are skipped. Filters out bullets, spacers, rules.
    """

    def __init__(
        self,
        pdf_path: str,
        output_dir: str,
        image_subdir: str = "output_images",
        min_image_size: int = 32,
    ):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.image_subdir = image_subdir
        self.image_dir = self.output_dir / image_subdir
        self.min_image_size = min_image_size

        # xref -> relative path. A logo repeated on every page is saved once.
        self._saved: Dict[int, Optional[str]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract_elements(self) -> List[List[PageElement]]:
        """Returns one reading-ordered list of elements per page."""
        pages: List[List[PageElement]] = []
        with pymupdf.open(self.pdf_path) as doc:
            for page in doc:
                text_elements = self._text_elements(page)
                content_rect = _content_rect(text_elements, page.rect)
                elements = text_elements + self._image_elements(doc, page, content_rect)
                pages.append(self._reading_order(elements, page.rect))
        return pages

    def extract_markdown(self) -> str:
        """
        Returns the whole document as one Markdown string: page markers, text
        blocks and image links in reading order, separated by blank lines.
        Image links are later protected from translation by TextShield.
        """
        parts: List[str] = []
        for page_num, elements in enumerate(self.extract_elements(), 1):
            if not elements:
                continue
            parts.append(f"--- Page {page_num} ---")
            parts.extend(el.to_markdown() for el in elements)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Text
    # ------------------------------------------------------------------ #
    def _text_elements(self, page: pymupdf.Page) -> List[TextElement]:
        elements = []
        # sort=False: we do our own ordering below.
        for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks"):
            if block_type != _TEXT_BLOCK:
                continue  # image blocks are handled via get_images()
            text = " ".join(text.split())  # unwrap hard line breaks inside a block
            if not text:
                continue  # empty / whitespace-only blocks
            elements.append(TextElement(page.number + 1, pymupdf.Rect(x0, y0, x1, y1), text))
        return elements

    # ------------------------------------------------------------------ #
    # Images
    # ------------------------------------------------------------------ #
    def _image_elements(
        self, doc: pymupdf.Document, page: pymupdf.Page, content_rect: pymupdf.Rect
    ) -> List[ImageElement]:
        elements = []
        page_num = page.number + 1
        index = 0
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception as e:  # malformed page content
                print(f"   [warn] page {page_num}: cannot locate image xref {xref}: {e}")
                continue
            # No rects = image is referenced by the page but never drawn.
            rects = [r for r in rects if not r.is_empty and r.intersects(page.rect)]
            if not rects:
                continue

            index += 1
            rel_path = self._save_image(doc, xref, page_num, index)
            if rel_path is None:
                index -= 1
                continue

            # One element per placement (the same image can be drawn twice).
            for rect in rects:
                caption = f"Figure (page {page_num}, image {index})"
                width_pct = _width_percent(rect, content_rect)
                elements.append(ImageElement(page_num, rect, rel_path, caption, width_pct))
        return elements

    def _save_image(self, doc: pymupdf.Document, xref: int, page_num: int, index: int) -> Optional[str]:
        """Writes the image to disk once and returns its Markdown-relative path."""
        if xref in self._saved:
            return self._saved[xref]

        rel_path = None
        try:
            info = doc.extract_image(xref)
            if not info or not info.get("image"):
                raise ValueError("no image data")
            if info["width"] < self.min_image_size or info["height"] < self.min_image_size:
                self._saved[xref] = None
                return None

            data: bytes = info["image"]
            ext = (info.get("ext") or "").lower().lstrip(".")
            smask = info.get("smask", 0)

            # Re-encode to PNG when: the image has a separate transparency mask
            # (raw bytes would lose it and show a black background), the format
            # is not web-renderable, or the extension is missing.
            if smask or ext not in _WEB_SAFE_EXTS:
                data, ext = self._to_png(doc, xref, smask), "png"

            self.image_dir.mkdir(parents=True, exist_ok=True)
            filename = f"page_{page_num}_img_{index}.{ext}"
            (self.image_dir / filename).write_bytes(data)
            rel_path = f"{self.image_subdir}/{filename}"
        except Exception as e:
            print(f"   [warn] page {page_num}: skipping image xref {xref}: {e}")

        self._saved[xref] = rel_path
        return rel_path

    @staticmethod
    def _to_png(doc: pymupdf.Document, xref: int, smask: int) -> bytes:
        pix = pymupdf.Pixmap(doc, xref)
        if smask:
            mask = pymupdf.Pixmap(doc, smask)
            if pix.alpha:  # drop any existing alpha before applying the mask
                pix = pymupdf.Pixmap(pix, 0)
            try:
                pix = pymupdf.Pixmap(pix, mask)
            except Exception:
                pass  # mask size mismatch: keep the opaque image
        if pix.colorspace and pix.colorspace.n > 3:  # CMYK etc. -> RGB
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        return pix.tobytes("png")

    # ------------------------------------------------------------------ #
    # Reading order
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reading_order(elements: List[PageElement], page_rect: pymupdf.Rect) -> List[PageElement]:
        """
        Sorts elements top-to-bottom, handling two-column layouts.

        An element counts as a column element if it sits entirely in the left
        or right half of the page. Full-width elements (titles, wide figures,
        abstracts) split the page into horizontal bands. Inside each band the
        whole left column is read before the right column; bands are read top
        to bottom. Single-column pages fall out naturally as pure y-sorting.
        """
        if not elements:
            return []

        mid_x = page_rect.x0 + page_rect.width / 2
        gutter = page_rect.width * 0.02  # tolerance for blocks touching the middle

        def column(el: PageElement) -> int:
            if el.bbox.x1 <= mid_x + gutter:
                return 0  # left
            if el.bbox.x0 >= mid_x - gutter:
                return 1  # right
            return -1     # spans both columns

        ordered: List[PageElement] = []
        band: List[PageElement] = []

        def flush_band():
            band.sort(key=lambda el: (column(el), el.bbox.y0, el.bbox.x0))
            ordered.extend(band)
            band.clear()

        for el in sorted(elements, key=lambda el: (el.bbox.y0, el.bbox.x0)):
            if column(el) == -1:
                flush_band()
                ordered.append(el)
            else:
                band.append(el)
        flush_band()
        return ordered
