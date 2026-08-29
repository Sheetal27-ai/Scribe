import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class TranslationEngine:
    def __init__(self, target_lang: str = "Hindi"):
        self.target_lang = target_lang
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables.")
        self.client = genai.Client(api_key=api_key)

    def translate(self, text: str) -> str:
        if not text.strip():
            return text

        prompt = (
            f"Translate the following text into {self.target_lang}.\n"
            f"CRITICAL: Do NOT translate, alter, or remove any placeholder tags formatted like ___SHIELD_X___.\n\n"
            f"Text:\n{text}"
        )

        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )
        return response.text.strip()