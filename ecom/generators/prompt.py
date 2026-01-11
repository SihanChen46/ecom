"""Prompt generation from product images and documents."""

import json
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
        documents: list[str] = None,
    ) -> tuple[str, list[dict]]:
        """
        Generate image prompts from product image and documents.
        
        Args:
            image_path: Path to main product image
            meta_prompt_path: Path to meta prompt file
            documents: Optional list of document paths
            
        Returns:
            (raw_analysis, prompts) tuple
        """
        image_bytes, mime_type = load_image(image_path)
        meta_prompt = load_file(meta_prompt_path)

        # Build content parts
        contents = [self.client.create_image_part(image_bytes, mime_type)]

        # Add documents, track temp files for cleanup
        temp_files = []
        for doc_path in (documents or []):
            doc_bytes, doc_mime, temp_file = load_document(Path(doc_path))
            contents.append(self.client.create_image_part(doc_bytes, doc_mime))
            if temp_file:
                temp_files.append(temp_file)

        contents.append(meta_prompt)

        try:
            result = self.client.generate_text(contents)
        finally:
            # Cleanup temp files
            for tf in temp_files:
                try:
                    tf.unlink()
                except Exception:
                    pass

        return result.text, self._extract_prompts(result.text)

    def _extract_prompts(self, text: str) -> list[dict]:
        """Extract JSON prompts from the generated analysis text."""
        # Find all JSON code blocks
        json_blocks = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        
        prompts = []
        for i, block in enumerate(json_blocks):
            try:
                # Parse JSON
                prompt_data = json.loads(block.strip())
                
                # Convert JSON to a descriptive prompt string for image generation
                prompt_text = self._json_to_prompt(prompt_data)
                
                # Extract a name from the prompt
                name = self._extract_name(prompt_data, i)
                
                prompts.append({
                    "name": name,
                    "prompt": prompt_text,
                    "raw_json": prompt_data,
                })
            except json.JSONDecodeError as e:
                print(f"      ⚠️  Failed to parse JSON block {i+1}: {e}")
                continue
        
        return prompts

    def _json_to_prompt(self, data: dict) -> str:
        """Convert structured JSON to a descriptive prompt string."""
        parts = []
        
        # Shot/Composition
        if data.get("shot"):
            parts.append(data["shot"])
        
        # Subject details
        subject = data.get("subject", {})
        if subject:
            subject_parts = []
            if subject.get("item"):
                subject_parts.append(subject["item"])
            if subject.get("colors"):
                subject_parts.append(f"colors: {subject['colors']}")
            if subject.get("materials"):
                subject_parts.append(f"made of {subject['materials']}")
            if subject.get("action"):
                subject_parts.append(subject["action"])
            if subject.get("condition"):
                subject_parts.append(subject["condition"])
            if subject_parts:
                parts.append(", ".join(subject_parts))
        
        # Environment
        if data.get("environment"):
            parts.append(f"Environment: {data['environment']}")
        
        # Camera settings
        camera = data.get("camera", {})
        if camera:
            cam_parts = []
            if camera.get("focal_length"):
                cam_parts.append(camera["focal_length"])
            if camera.get("aperture"):
                cam_parts.append(camera["aperture"])
            if camera.get("angle"):
                cam_parts.append(camera["angle"])
            if cam_parts:
                parts.append(f"Shot with {', '.join(cam_parts)}")
        
        # Lighting
        if data.get("lighting"):
            parts.append(f"Lighting: {data['lighting']}")
        
        # Color grade
        if data.get("color_grade"):
            parts.append(f"Color grade: {data['color_grade']}")
        
        # Style
        if data.get("style"):
            parts.append(f"Style: {data['style']}")
        
        # Quality
        if data.get("quality"):
            parts.append(data["quality"])
        
        # Negatives (what to avoid)
        if data.get("negatives"):
            parts.append(f"Avoid: {data['negatives']}")
        
        return ". ".join(parts)

    def _extract_name(self, data: dict, index: int) -> str:
        """Extract a descriptive name from the prompt data."""
        # Try to get name from style or subject item
        if data.get("style"):
            return data["style"][:40]
        if data.get("subject", {}).get("item"):
            return data["subject"]["item"][:40]
        return f"Prompt {index + 1}"
