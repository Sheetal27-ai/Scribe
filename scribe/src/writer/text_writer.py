import os
from pathlib import Path

# Reading styles kept in a real .css file so they are easy to edit.
STYLESHEET_PATH = Path(__file__).with_name("scribe.css")


def load_stylesheet() -> str:
    """Returns the CSS as a <style> block, or "" if the file is missing."""
    try:
        css = STYLESHEET_PATH.read_text(encoding="utf-8").strip()
    except OSError as e:
        print(f"   [warn] stylesheet not found ({e}); writing unstyled Markdown.")
        return ""
    # No blank lines inside the block: some Markdown renderers end an HTML
    # block at the first blank line and would print the rest as text.
    css = "\n".join(line for line in css.splitlines() if line.strip())
    return f"<style>\n{css}\n</style>"


class FileWriter:
    """Saves translated document content into structured files."""

    @staticmethod
    def save_markdown(content: str, output_path: str, styled: bool = True):
        """
        Writes the Markdown file. With styled=True a <style> block is put at
        the very top so previewers (VS Code, Typora, HTML exports) render the
        paper with Scribe's reading styles. It is added here, after
        translation, so the CSS never goes through the translator.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        style = load_stylesheet() if styled else ""
        with open(output_path, "w", encoding="utf-8") as f:
            if style:
                f.write(style + "\n\n")
            f.write(content.lstrip())
            if not content.endswith("\n"):
                f.write("\n")
        print(f"Saved translated document to: {output_path}")
