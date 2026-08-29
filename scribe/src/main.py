from parser.shield import TextShield
from translator.engine import TranslationEngine

def run_v0_pipeline(sample_text: str, target_lang: str = "Spanish") -> str:
    print(f"=== Running Scribe Pipeline (Target Language: {target_lang}) ===")
    print("\n1. Original Input Text:")
    print(sample_text)
    
    # 1. Initialize shield engine and translator
    shield_engine = TextShield()
    translator = TranslationEngine(target_lang=target_lang)
    
    # 2. Shield LaTeX math & code blocks
    shielded_text, shield_map = shield_engine.shield(sample_text)
    print("\n2. Shielded Text (placeholders inserted):")
    print(shielded_text)
    
    # 3. Pass to Gemini API
    translated_shielded = translator.translate(shielded_text)
    print("\n3. Translated Output (with placeholders preserved):")
    print(translated_shielded)
    
    # 4. Unshield placeholders back to original LaTeX
    final_output = shield_engine.unshield(translated_shielded, shield_map)
    print("\n4. Final Restored Text:")
    print(final_output)
    
    return final_output

if __name__ == "__main__":
    # Test string with inline math ($...$), display math ($$...$$), and standard prose
    test_academic_text = (
        "In quantum mechanics, the time-dependent wave equation is defined as "
        "$$i\\hbar \\frac{\\partial}{\\partial t}\\Psi = \\hat{H}\\Psi$$. "
        "Furthermore, special relativity shows that total energy equals $E = mc^2$."
    )
    
    run_v0_pipeline(test_academic_text, target_lang="Hindi")

