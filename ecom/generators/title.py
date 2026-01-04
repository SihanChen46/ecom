"""SEO title generation for e-commerce listings."""

import json
import re
from pathlib import Path

from ..client import GeminiClient, TokenUsage
from ..utils import load_file, load_image, load_document


class TitleGenerator:
    def __init__(self, client: GeminiClient):
        self.client = client

    def generate(
        self,
        image_path: str,
        title_prompt_path: str,
        documents: list[Path] = None,
    ) -> tuple[dict, TokenUsage]:
        """Generate SEO titles from product image and documents."""
        image_bytes, mime_type = load_image(image_path)
        title_prompt = load_file(title_prompt_path)

        # Build content parts and estimate tokens
        contents = [self.client.create_image_part(image_bytes, mime_type)]
        
        # Estimate tokens by content type
        image_tokens = self.client.estimate_image_tokens(image_bytes)
        text_tokens = self.client.estimate_text_tokens(title_prompt)
        document_tokens = 0

        # Add documents, track temp files for cleanup
        temp_files = []
        for doc in (documents or []):
            doc_bytes, doc_mime, temp_file = load_document(doc)
            contents.append(self.client.create_image_part(doc_bytes, doc_mime))
            document_tokens += self.client.estimate_image_tokens(doc_bytes)
            if temp_file:
                temp_files.append(temp_file)

        contents.append(title_prompt)
        
        content_info = {
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "document_tokens": document_tokens,
        }

        try:
            result = self.client.generate_text(contents, content_info=content_info)
            # Set output stats
            result.usage.output_text_tokens = result.usage.completion_tokens
        finally:
            # Cleanup temp files
            for tf in temp_files:
                try:
                    tf.unlink()
                except Exception:
                    pass

        return self._parse_response(result.text), result.usage

    def _parse_response(self, text: str) -> dict:
        """Extract JSON from response."""
        # Try to find JSON block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try direct JSON parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Return raw text as fallback
        return {"raw_response": text, "parse_error": True}

