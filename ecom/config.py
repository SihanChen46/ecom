"""Configuration management."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

MODELS = {
    "gemini": "gemini-2.0-flash-exp",
    "gemini-3": "gemini-3-pro-image-preview",
    "imagen": "imagen-4.0-generate-001",
    "imagen-ultra": "imagen-4.0-ultra-generate-001",
}


@dataclass
class Config:
    model_text: str = "gemini-2.0-flash"
    model_image: str = "gemini-2.0-flash-exp"
    meta_prompt_file: str = "meta_prompt.txt"
    output_dir: str = "outputs"
    catalog_dir: str = "catalog"

    @staticmethod
    def get_api_key() -> str:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("Error: GEMINI_API_KEY not set")
        return key

    @classmethod
    def with_model(cls, model_name: str, **kwargs) -> "Config":
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        return cls(model_image=MODELS[model_name], **kwargs)

