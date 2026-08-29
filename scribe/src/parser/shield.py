import re
from typing import Tuple, Dict

class TextShield:
    def __init__(self):
        # Regex patterns for display math ($$...$$), inline math ($...$), and code blocks
        self.patterns = [
            r'\$\$.*?\$\$',         # Display math $$ ... $$
            r'\$.*?\$',             # Inline math $ ... $
            r'```[\s\S]*?```',      # Multi-line code blocks
            r'`[^`\n]+`',           # Inline code
        ]
        self.combined_pattern = re.compile('|'.join(self.patterns), re.DOTALL)

    def shield(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replaces LaTeX and code snippets with unique placeholders."""
        maps: Dict[str, str] = {}
        counter = 0

        def replace_match(match: re.Match) -> str:
            nonlocal counter
            placeholder = f"___SHIELD_{counter}___"
            maps[placeholder] = match.group(0)
            counter += 1
            return placeholder

        shielded_text = self.combined_pattern.sub(replace_match, text)
        return shielded_text, maps

    def unshield(self, shielded_text: str, maps: Dict[str, str]) -> str:
        """Restores original LaTeX and code snippets from placeholders."""
        unshielded_text = shielded_text
        for placeholder, original in maps.items():
            unshielded_text = unshielded_text.replace(placeholder, original)
        return unshielded_text