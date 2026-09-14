from typing import List

class TextChunker:
    """Splits large documents into manageable paragraph blocks for API translation."""
    
    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max_chunk_size

    def chunk_text(self, text: str) -> List[str]:
        """Splits text by double newlines (paragraphs/page markers) into sized chunks."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len > self.max_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_length = para_len
            else:
                current_chunk.append(para)
                current_length += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks