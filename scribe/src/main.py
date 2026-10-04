import os
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from parser.pdf_reader import PDFExtractor
from parser.shield import TextShield
from parser.chunker import TextChunker
from translator.engine import TranslationEngine
from writer.text_writer import FileWriter

def run_v1_pipeline(pdf_path: str, target_lang: str = "Hindi"):
    print(f"=== Running Scribe Version 1 Pipeline ===")
    print(f"Input PDF: {pdf_path}")
    print(f"Target Language: {target_lang}\n")

    # 1. Extract text
    print("1. Extracting text from PDF...")
    extractor = PDFExtractor(pdf_path)
    raw_text = extractor.extract_text()
    print(f"   Extracted {len(raw_text)} characters.")

    # 2. Chunk text with larger window (4000 chars) to reduce total request count
    print("2. Chunking text...")
    chunker = TextChunker(max_chunk_size=12000)
    chunks = chunker.chunk_text(raw_text)
    print(f"   Created {len(chunks)} chunks.")

    # 3. Translate with rate-limiting pauses
    translator = TranslationEngine(target_lang=target_lang)
    translated_chunks = []

    print("\n3. Processing and Translating chunks...")
    for idx, chunk in enumerate(chunks, 1):
        print(f"   Translating chunk {idx}/{len(chunks)}...")
        
        shield = TextShield()
        shielded_text, placeholders = shield.shield(chunk)

        translated_shielded = translator.translate(shielded_text)

        final_chunk = shield.unshield(translated_shielded, placeholders)
        translated_chunks.append(final_chunk)

        # Pause 2 seconds between API requests to avoid tripping per-minute rate limits
        if idx < len(chunks):
            time.sleep(2)

    # 4. Save output
    full_translation = "\n\n".join(translated_chunks)
    output_path = os.path.join("output", "translated_paper.md")
    
    print("\n4. Saving final output...")
    FileWriter.save_markdown(full_translation, output_path)
    print("\n=== Pipeline Execution Completed Successfully! ===")

if __name__ == "__main__":
    sample_pdf = "test_files/sample.pdf"
    if os.path.exists(sample_pdf):
        run_v1_pipeline(sample_pdf, target_lang="Hindi")
    else:
        print(f"Error: Please place a sample PDF file at '{sample_pdf}'")