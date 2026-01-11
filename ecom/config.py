"""Configuration management."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

MODELS = {
    "gemini": "gemini-2.0-flash-exp",
    "gemini-3": "gemini-3-pro-image-preview",
    "imagen": "imagen-4.0-generate-001",
    "imagen-ultra": "imagen-4.0-ultra-generate-001",
}

# Image generation modes
MODES = {
    "cover": "prompts/cover.txt",      # 主图 - 创意风格
    "preview": "prompts/preview.txt",  # 预览图 - 写实还原
    "top": "prompts/top.txt",          # 全套11张 - 电商详情页完整方案
    "adapt": "prompts/adapt.txt",      # 色彩适配 - 目标图+产品图合成
}


@dataclass
class Config:
    model_text: str = "gemini-3-flash-preview"  # Flash model for prompt generation
    model_image: str = "gemini-3-pro-image-preview"
    # model_image: str = "gemini-2.5-flash-image
    output_dir: str = "outputs"
    catalog_dir: str = "catalog"
    prompts_dir: str = "prompts"

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

    def get_prompt_file(self, mode: str) -> str:
        """Get the prompt file path for a given mode."""
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(MODES.keys())}")
        return MODES[mode]
