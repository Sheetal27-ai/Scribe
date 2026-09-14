import os
import time
from google import genai
from google.genai import types
from google.genai import errors
from dotenv import load_dotenv

load_dotenv()

class TranslationEngine:
    def __init__(self, target_lang: str = "Hindi"):
        self.target_lang = target_lang
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables.")
        self.client = genai.Client(api_key=api_key)

    def translate(self, text: str, max_retries: int = 3) -> str:
        if not text.strip():
            return text

        prompt = (
            f"Translate the following text into {self.target_lang}.\n"
            f"CRITICAL: Do NOT translate, alter, or remove any placeholder tags formatted like ___SHIELD_X___.\n\n"
            f"Text:\n{text}"
        )

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                )
                return response.text.strip()
            except errors.APIError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = (attempt + 1) * 10
                    print(f"\n   [Quota Exceeded] Rate limit hit. Waiting {wait_time}s before retry (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e

        raise RuntimeError("Failed to translate chunk after multiple retry attempts due to API quota limits.")