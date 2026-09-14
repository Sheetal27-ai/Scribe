import pymupdf
import os

class PDFExtractor:
    """Extracts raw text page-by-page from selectable PDF files."""
    
    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self.pdf_path = pdf_path

    def extract_text(self) -> str:
        """Reads all pages and returns extracted text separated by page markers."""
        doc = pymupdf.open(self.pdf_path)
        extracted_pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                extracted_pages.append(f"--- Page {page_num + 1} ---\n\n{text.strip()}")

        doc.close()
        return "\n\n".join(extracted_pages)