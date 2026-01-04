"""Gemini API client wrapper."""

from google import genai
from google.genai import types

from .config import Config


class GeminiClient:
    def __init__(self, config: Config):
        self.config = config
        self._client = genai.Client(api_key=config.get_api_key())

    def generate_text(self, contents: list, model: str = None) -> str:
        model = model or self.config.model_text
        response = self._client.models.generate_content(model=model, contents=contents)
        return response.text

    def generate_image(self, contents: list) -> list[tuple[bytes, str]]:
        response = self._client.models.generate_content(
            model=self.config.model_image,
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        
        images = []
        for candidate in response.candidates or []:
            for part in (candidate.content.parts if candidate.content else []):
                if hasattr(part, "inline_data") and part.inline_data:
                    data = part.inline_data.data
                    if isinstance(data, str):
                        import base64
                        data = base64.b64decode(data)
                    images.append((data, part.inline_data.mime_type or "image/png"))
        return images

    def create_image_part(self, image_bytes: bytes, mime_type: str) -> types.Part:
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

