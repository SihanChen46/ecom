"""Prompt generation from product images and documents."""

import re
from pathlib import Path

from ..client import GeminiClient
from ..utils import load_file, load_image, load_document


class PromptGenerator:
    def __init__(self, client: GeminiClient):
        self.client = client

    def generate(
        self,
        image_path: str,
        meta_prompt_path: str,
        documents: list[Path] = None,
    ) -> tuple[str, list[dict]]:
        image_bytes, mime_type = load_image(image_path)
        meta_prompt = load_file(meta_prompt_path)

        # Build content parts
        contents = [self.client.create_image_part(image_bytes, mime_type)]

        # Add documents, track temp files for cleanup
        temp_files = []
        for doc in (documents or []):
            doc_bytes, doc_mime, temp_file = load_document(doc)
            contents.append(self.client.create_image_part(doc_bytes, doc_mime))
            if temp_file:
                temp_files.append(temp_file)

        contents.append(meta_prompt)

        try:
            raw = self.client.generate_text(contents)
        finally:
            # Cleanup temp files
            for tf in temp_files:
                try:
                    tf.unlink()
                except Exception:
                    pass

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
