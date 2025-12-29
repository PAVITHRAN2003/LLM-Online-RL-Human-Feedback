import os
from google import genai

class ModelInterface:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)
        self.model_name = "models/gemini-2.5-flash"

    def generate(self, prompt, max_new_tokens=None):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
        )

        return response.text

    def logprob(self, prompt, completion):
        raise NotImplementedError(
            "Gemini does not expose token-level log probabilities"
        )
