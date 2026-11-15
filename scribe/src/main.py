import sys
from pathlib import Path
# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import argparse
from src.parser.layout import LayoutExtractor
from src.parser.chunker import TextChunker
from src.parser.shield import TextShield
from src.translator.engine import TranslationEngine
from src.writer.text_writer import FileWriter

def parse_args():
    parser = argparse.ArgumentParser(description="Scribe: Academic PDF Translator preserving LaTeX math.")
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True, 
        help="Path to input PDF file (e.g., test_files/sample.pdf)"
    )
    parser.add_argument(
        "-l", "--lang", 
        type=str, 
        default="Hindi", 
        help="Target language for translation (default: Hindi)"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="output/translated_paper.md", 
        help="Destination path for output Markdown (default: output/translated_paper.md)"
    )
    parser.add_argument(
        "-c", "--chunk-size", 
        type=int, 
        default=12000, 
        help="Max character count per chunk (default: 12000)"
    )
    return parser.parse_args()

def run_v2_pipeline(input_path: str, target_lang: str, output_path: str, max_chunk_size: int):
    pdf_file = Path(input_path)
    if not pdf_file.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    print("=== Running Scribe Version 2 Pipeline ===")
    print(f"Input PDF: {input_path}")
    print(f"Target Language: {target_lang}")
    print(f"Output Target: {output_path}")

    # 1. Extraction
    print("\n1. Extracting text and images in reading order...")
    output_dir = os.path.dirname(output_path) or "."
    extractor = LayoutExtractor(str(pdf_file), output_dir=output_dir)
    raw_text = extractor.extract_markdown()
    print(f"   Extracted {len(raw_text)} characters.")
    print(f"   Saved images to '{extractor.image_dir}'.")

    # 2. Chunking
    print("\n2. Chunking text...")
    chunker = TextChunker(max_chunk_size=max_chunk_size)
    chunks = chunker.chunk_text(raw_text)
    print(f"   Created {len(chunks)} chunks.")

    # 3. Shielding & Translation
    print("\n3. Processing, Shielding, and Translating chunks...")
    shield = TextShield()
    translator = TranslationEngine(target_lang=target_lang)
    translated_chunks = []

    for i, chunk in enumerate(chunks, 1):
        print(f"   Translating chunk {i}/{len(chunks)}...")
        shielded_text, placeholders = shield.shield(chunk)
        translated_shielded = translator.translate(shielded_text)
        final_text = shield.unshield(translated_shielded, placeholders)
        translated_chunks.append(final_text)

    # 4. Writing Output
    print("\n4. Exporting translated document...")
    full_translation = "\n\n".join(translated_chunks)
    FileWriter.save_markdown(full_translation, output_path)
    print(f"\n=== Success! Output saved to '{output_path}' ===")

if __name__ == "__main__":
    args = parse_args()
    run_v2_pipeline(
        input_path=args.input,
        target_lang=args.lang,
        output_path=args.output,
        max_chunk_size=args.chunk_size
    )