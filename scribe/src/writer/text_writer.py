import os

class FileWriter:
    """Saves translated document content into structured files."""
    
    @staticmethod
    def save_markdown(content: str, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved translated document to: {output_path}")