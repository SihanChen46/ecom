"""Prompt generation from product images."""

import re
from pathlib import Path

from ..client import GeminiClient
from ..utils import load_file, load_image


class PromptGenerator:
    def __init__(self, client: GeminiClient):
        self.client = client

    def generate(self, image_path: str, meta_prompt_path: str) -> tuple[str, list[dict]]:
        image_bytes, mime_type = load_image(image_path)
        meta_prompt = load_file(meta_prompt_path)
        
        image_part = self.client.create_image_part(image_bytes, mime_type)
        raw = self.client.generate_text([image_part, meta_prompt])
        
        return raw, self._extract_prompts(raw)

    def _extract_prompts(self, text: str) -> list[dict]:
        blocks = re.findall(r"```(?:\w+)?\s*([\s\S]*?)```", text)
        sections = re.split(r"(?:Image \d+|Detail \d+)\s*\[([^\]]+)\]", text)
        names = [s.strip() for i, s in enumerate(sections) if i % 2 == 1]

        prompts = []
        for i, block in enumerate(blocks):
            block = block.strip()
            if block and len(block) > 50:
                prompts.append({
                    "name": names[i] if i < len(names) else f"Prompt {i + 1}",
                    "prompt": block,
                })
        return prompts

