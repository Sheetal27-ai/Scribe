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
from typing import Dict, List, Optional, Tuple, Union

import pymupdf

from src.parser.decorations import find_horizontal_rules, is_centered, is_page_number
from src.parser.figures import CAPTION_RE, find_vector_figures
from src.parser.styles import StyleProfile, block_to_markdown, build_profile

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
    text: str              # plain text: used for layout logic (captions etc.)
    markdown: str = ""     # styled text: headings, **bold**, *italic*, colour
    centered: bool = False # sits in the horizontal middle of the page

    def to_markdown(self) -> str:
        body = self.markdown or self.text
        if not self.centered:
            return body
        # Blank lines around the content end the HTML block, so Markdown
        # inside (headings, **bold**) is still parsed by CommonMark renderers.
        return f'<div style="text-align: center;">\n\n{body}\n\n</div>'


@dataclass
class RuleElement:
    """A horizontal divider line drawn on the page."""
    page_num: int
    bbox: pymupdf.Rect

    def to_markdown(self) -> str:
        return "<hr>"


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


def _rect_inside(inner: pymupdf.Rect, outer: pymupdf.Rect, tol: float = 2.0) -> bool:
    return (inner.x0 >= outer.x0 - tol and inner.y0 >= outer.y0 - tol
            and inner.x1 <= outer.x1 + tol and inner.y1 <= outer.y1 + tol)


PageElement = Union[TextElement, ImageElement, RuleElement]


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
        figure_dpi: Resolution for vector figures rendered from captions.
    """

    def __init__(
        self,
        pdf_path: str,
        output_dir: str,
        image_subdir: str = "output_images",
        min_image_size: int = 32,
        figure_dpi: int = 300,
    ):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.image_subdir = image_subdir
        self.image_dir = self.output_dir / image_subdir
        self.min_image_size = min_image_size
        self.figure_dpi = figure_dpi

        # xref -> relative path. A logo repeated on every page is saved once.
        self._saved: Dict[int, Optional[str]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract_elements(self) -> List[List[PageElement]]:
        """Returns one reading-ordered list of elements per page."""
        pages: List[List[PageElement]] = []
        with pymupdf.open(self.pdf_path) as doc:
            profile = build_profile(doc)
            body_size = profile.body_size
            for page in doc:
                text_elements = self._text_elements(page, profile)
                content_rect = _content_rect(text_elements, page.rect)
                image_elements = self._image_elements(doc, page, content_rect)
                figures, text_elements, image_elements = self._vector_figures(
                    page, text_elements, image_elements, content_rect, body_size
                )
                text_elements = self._apply_alignment(page, text_elements)
                rules = self._rules(page, text_elements, image_elements + figures)
                elements = text_elements + image_elements + figures + rules
                pages.append(self._reading_order(elements, page.rect))
        self._remove_unused_images(pages)
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
    def _text_elements(self, page: pymupdf.Page, profile: StyleProfile) -> List[TextElement]:
        """
        Reads text at span level so font size, weight and slant survive as
        Markdown headings and **bold** / *italic* runs.
        """
        try:
            blocks = page.get_text("dict")["blocks"]
        except Exception as e:
            print(f"   [warn] page {page.number + 1}: cannot read text: {e}")
            return []

        elements = []
        for block in blocks:
            if block.get("type") != _TEXT_BLOCK:
                continue  # image blocks are handled via get_images()
            try:
                text, markdown = block_to_markdown(block, profile)
            except Exception as e:  # odd span data: fall back to plain text
                print(f"   [warn] page {page.number + 1}: style analysis failed: {e}")
                text = " ".join(
                    s.get("text", "") for l in block.get("lines", []) for s in l.get("spans", [])
                )
                text = " ".join(text.split())
                markdown = ""
            if not text:
                continue  # empty / whitespace-only blocks
            elements.append(TextElement(page.number + 1, pymupdf.Rect(block["bbox"]), text, markdown))
        return elements

    # ------------------------------------------------------------------ #
    # Alignment and dividers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_alignment(page: pymupdf.Page, text_elements: List[TextElement]) -> List[TextElement]:
        """
        Drops page numbers and marks page-centred blocks (titles, author
        lines, display equations). Figure captions are left alone: the
        stylesheet already centres them via `img + p`, which a wrapper
        <div> would break.
        """
        kept = []
        for el in text_elements:
            if is_page_number(el.text, el.bbox, page.rect):
                continue  # "3" at the foot of the page is noise in Markdown
            el.centered = (
                not CAPTION_RE.match(el.text)
                and is_centered(el.bbox, page.rect)
            )
            kept.append(el)
        return kept

    @staticmethod
    def _rules(
        page: pymupdf.Page,
        text_elements: List[TextElement],
        pictures: List[ImageElement],
    ) -> List[RuleElement]:
        """Horizontal dividers, ignoring those inside figures and images."""
        # The widest text block is a full text line: the reference length.
        text_width = max((el.bbox.width for el in text_elements), default=page.rect.width)
        rects = find_horizontal_rules(page, text_width, exclude=[p.bbox for p in pictures])
        return [RuleElement(page.number + 1, r) for r in rects]

    # ------------------------------------------------------------------ #
    # Vector figures (caption-triggered region rendering)
    # ------------------------------------------------------------------ #
    def _vector_figures(
        self,
        page: pymupdf.Page,
        text_elements: List[TextElement],
        image_elements: List[ImageElement],
        content_rect: pymupdf.Rect,
        body_size: float,
    ) -> Tuple[List[ImageElement], List[TextElement], List[ImageElement]]:
        """
        Renders figures found above captions to PNG and removes the text
        fragments and raster pieces they replace.

        Returns (figure elements, remaining text, remaining raster images).
        """
        page_num = page.number + 1
        regions = find_vector_figures(
            page,
            [t.bbox for t in text_elements],
            [t.text for t in text_elements],
            [i.bbox for i in image_elements],
            body_size,
        )

        figures: List[ImageElement] = []
        drop_text: set = set()
        drop_images: set = set()
        for index, region in enumerate(regions, 1):
            # A plain embedded bitmap with no text labels around it (Figure 1
            # in most papers) is already extracted at full native resolution;
            # re-rendering it would only lose quality.
            if region.image_rects and not region.has_labels:
                continue

            filename = f"page_{page_num}_fig_{index}.png"
            try:
                pix = page.get_pixmap(clip=region.rect, dpi=self.figure_dpi)
                self.image_dir.mkdir(parents=True, exist_ok=True)
                pix.save(str(self.image_dir / filename))
            except Exception as e:
                # Leave the text blocks in place so nothing is lost.
                print(f"   [warn] page {page_num}: could not render Figure {region.number}: {e}")
                continue

            figures.append(ImageElement(
                page_num,
                region.rect,
                f"{self.image_subdir}/{filename}",
                f"Figure {region.number}",
                _width_percent(region.rect, content_rect),
            ))
            # Labels found by the detector, plus any other block that falls
            # entirely inside the rendered area: it is now part of the picture.
            drop_text.update(region.label_indices)
            drop_text.update(
                i for i, t in enumerate(text_elements)
                if i != region.caption_index and region.rect.contains(t.bbox)
            )
            # Raster pieces inside the rendered area are part of the picture too.
            drop_images.update(
                i for i, im in enumerate(image_elements)
                if _rect_inside(im.bbox, region.rect)
            )

        text_elements = [t for i, t in enumerate(text_elements) if i not in drop_text]
        image_elements = [im for i, im in enumerate(image_elements) if i not in drop_images]
        return figures, text_elements, image_elements

    def _remove_unused_images(self, pages: List[List[PageElement]]) -> None:
        """Deletes saved raster files that ended up inside a rendered figure."""
        used = {el.path for page in pages for el in page if isinstance(el, ImageElement)}
        for rel_path in set(self._saved.values()):
            if rel_path and rel_path not in used:
                try:
                    (self.output_dir / rel_path).unlink(missing_ok=True)
                except OSError:
                    pass

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
