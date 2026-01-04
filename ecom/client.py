"""Gemini API client wrapper."""

from dataclasses import dataclass
from google import genai
from google.genai import types

from .config import Config


# Gemini API Pricing (USD per 1M tokens) - Updated Jan 2025
# Official pricing: https://ai.google.dev/pricing
#
# NOTE: Image generation costs are calculated via OUTPUT TOKENS, not per-image fee
# Each generated image consumes ~1,290-5,000 output tokens depending on resolution
# Cost per image ≈ (tokens / 1M) × output_text_rate
#
PRICING = {
    # Gemini 2.0 Flash (with native image generation - "Nano Banana")
    "gemini-2.0-flash": {
        "input_text": 0.10,      # $0.10 per 1M input tokens
        "input_image": 0.10,     # $0.10 per 1M tokens (images as tokens: ~258 tokens/image)
        "output_text": 0.40,     # $0.40 per 1M output tokens
        # Image output is charged via output tokens (~1,290+ tokens per image)
    },
    "gemini-2.0-flash-exp": {
        "input_text": 0.0,       # Free tier (experimental)
        "input_image": 0.0,
        "output_text": 0.0,
    },
    # Gemini 3 Pro Image Preview ("Nano Banana Pro")
    # This is the advanced image generation model
    "gemini-3-pro-image-preview": {
        "input_text": 0.10,      # Same as 2.0 Flash
        "input_image": 0.10,
        "output_text": 0.40,     # Images charged via output tokens
    },
    # Gemini 1.5 Flash
    "gemini-1.5-flash": {
        "input_text": 0.075,     # $0.075 per 1M input tokens
        "input_image": 0.075,
        "output_text": 0.30,     # $0.30 per 1M output tokens
    },
    # Gemini 1.5 Pro
    "gemini-1.5-pro": {
        "input_text": 1.25,      # $1.25 per 1M input tokens
        "input_image": 1.25,
        "output_text": 5.00,     # $5.00 per 1M output tokens
    },
    # Gemini 2.5 Pro
    "gemini-2.5-pro-preview-05-06": {
        "input_text": 1.25,      # $1.25 per 1M input tokens (<200k)
        "input_image": 1.25,
        "output_text": 10.00,    # $10.00 per 1M output tokens
    },
    # Imagen 4.0 (separate image generation API, fixed per-image pricing)
    "imagen-4.0-generate-001": {
        "input_text": 0.0,
        "input_image": 0.0,
        "output_text": 0.0,
        "per_image": 0.03,       # $0.03 per generated image
    },
    "imagen-4.0-ultra-generate-001": {
        "input_text": 0.0,
        "input_image": 0.0,
        "output_text": 0.0,
        "per_image": 0.06,       # $0.06 per generated image (ultra)
    },
    # Default fallback
    "default": {
        "input_text": 0.10,
        "input_image": 0.10,
        "output_text": 0.40,
    },
}


def get_pricing(model: str) -> dict:
    """Get pricing for a model, with fallback to default."""
    # Try exact match first
    if model in PRICING:
        return PRICING[model]
    # Try partial match
    for key in PRICING:
        if key in model or model in key:
            return PRICING[key]
    return PRICING["default"]


@dataclass
class TokenUsage:
    """Token usage statistics for a single API call."""
    # API reported totals
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Input breakdown (estimated)
    input_text_tokens: int = 0
    input_image_tokens: int = 0
    input_document_tokens: int = 0
    # Output breakdown
    output_text_tokens: int = 0
    output_image_count: int = 0
    # Model used (for pricing)
    model: str = ""
    
    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            input_text_tokens=self.input_text_tokens + other.input_text_tokens,
            input_image_tokens=self.input_image_tokens + other.input_image_tokens,
            input_document_tokens=self.input_document_tokens + other.input_document_tokens,
            output_text_tokens=self.output_text_tokens + other.output_text_tokens,
            output_image_count=self.output_image_count + other.output_image_count,
            model=self.model or other.model,
        )
    
    def calculate_cost(self, model: str = None) -> dict:
        """Calculate cost breakdown in USD.
        
        Note: For Gemini models with native image generation (e.g., gemini-2.0-flash,
        gemini-3-pro-image-preview), image output is charged via OUTPUT TOKENS.
        Each generated image consumes ~1,290-5,000 output tokens.
        
        For Imagen models, there's a fixed per-image fee.
        """
        pricing = get_pricing(model or self.model)
        
        # Input costs (per 1M tokens)
        input_text_cost = (self.prompt_tokens / 1_000_000) * pricing.get("input_text", 0)
        input_image_cost = (self.input_image_tokens / 1_000_000) * pricing.get("input_image", 0)
        input_doc_cost = (self.input_document_tokens / 1_000_000) * pricing.get("input_image", 0)
        
        # Output costs - ALL output (text + image) is charged via completion_tokens
        # For Gemini native image gen, generated images ARE output tokens
        output_text_cost = (self.completion_tokens / 1_000_000) * pricing.get("output_text", 0)
        
        # Imagen has fixed per-image pricing (separate from token-based)
        per_image_cost = self.output_image_count * pricing.get("per_image", 0)
        
        total_input = input_text_cost + input_image_cost + input_doc_cost
        total_output = output_text_cost + per_image_cost
        total = total_input + total_output
        
        return {
            "input": {
                "text": input_text_cost,
                "image": input_image_cost,
                "document": input_doc_cost,
                "subtotal": total_input,
            },
            "output": {
                "text_and_image": output_text_cost,  # Includes image gen tokens for Gemini
                "imagen_per_image": per_image_cost,  # Only for Imagen models
                "subtotal": total_output,
            },
            "total": total,
            "note": "Image generation cost is included in output tokens for Gemini models",
        }
    
    def to_dict(self, include_cost: bool = True, model: str = None) -> dict:
        result = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "input": {
                "text_tokens": self.input_text_tokens,
                "image_tokens": self.input_image_tokens,
                "document_tokens": self.input_document_tokens,
            },
            "output": {
                "text_tokens": self.output_text_tokens,
                "image_count": self.output_image_count,
            },
        }
        if include_cost:
            result["cost_usd"] = self.calculate_cost(model)
        return result


@dataclass
class TextResult:
    """Result from text generation including usage stats."""
    text: str
    usage: TokenUsage


@dataclass  
class ImageResult:
    """Result from image generation including usage stats."""
    images: list[tuple[bytes, str]]
    usage: TokenUsage


class GeminiClient:
    def __init__(self, config: Config):
        self.config = config
        self._client = genai.Client(api_key=config.get_api_key())

    def _extract_usage(self, response, content_info: dict = None) -> TokenUsage:
        """Extract token usage from API response."""
        usage = TokenUsage()
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            meta = response.usage_metadata
            usage.prompt_tokens = getattr(meta, 'prompt_token_count', 0) or 0
            usage.completion_tokens = getattr(meta, 'candidates_token_count', 0) or 0
            usage.total_tokens = getattr(meta, 'total_token_count', 0) or 0
        
        # Apply content breakdown if provided
        if content_info:
            usage.input_text_tokens = content_info.get('text_tokens', 0)
            usage.input_image_tokens = content_info.get('image_tokens', 0)
            usage.input_document_tokens = content_info.get('document_tokens', 0)
        
        return usage

    def generate_text(self, contents: list, model: str = None, content_info: dict = None) -> TextResult:
        model = model or self.config.model_text
        response = self._client.models.generate_content(model=model, contents=contents)
        usage = self._extract_usage(response, content_info)
        return TextResult(text=response.text, usage=usage)

    def generate_image(self, contents: list, content_info: dict = None) -> ImageResult:
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
        
        usage = self._extract_usage(response, content_info)
        return ImageResult(images=images, usage=usage)

    def create_image_part(self, image_bytes: bytes, mime_type: str) -> types.Part:
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    
    @staticmethod
    def estimate_text_tokens(text: str) -> int:
        """Rough estimate: ~4 chars per token for English, ~1.5 chars for Chinese."""
        # Simple heuristic based on character count
        return max(1, len(text) // 3)
    
    @staticmethod
    def estimate_image_tokens(image_bytes: bytes) -> int:
        """
        Gemini image token estimation.
        Images are typically 258 tokens per tile (768x768).
        Max tiles depends on resolution, but minimum is 258 tokens.
        """
        # Rough estimate based on file size - larger images = more tokens
        size_kb = len(image_bytes) / 1024
        # Base cost + scaled cost for larger images
        return 258 + int(size_kb / 100) * 258

