"""All mode prompt generation - generates complete 11-shot deck for e-commerce detail pages."""

import re
from pathlib import Path

from ..client import GeminiClient
from ..utils import load_file, load_image, load_document


class AllPromptGenerator:
    """
    Specialized generator for "top" mode that extracts Nano Banana natural language prompts
    from the V4.2 visual strategy output.
    """
    
    # The 11 image types in order (V4.2)
    IMAGE_TYPES = [
        "封面图 (The Hero Shot)",
        "购物车预览图 (Cart Preview)",
        "量大管饱图 (Abundance Shot)",
        "尺寸对比图 (Size Comparison)",
        "场景对比图 (Scenario Comparison)",
        "沉浸式场景图 (Immersive Scenario)",
        "痛点/解决方案图 (Pain/Solution)",
        "核心卖点可视化 (USP Visualization)",
        "特写/细节图 (Close-up/Texture)",
        "规格/多合一展示 (The \"What you get\")",
        "使用步骤/傻瓜式指南 (The How-to)",
    ]

    def __init__(self, client: GeminiClient):
        self.client = client

    def generate(
        self,
        image_paths: list[str],
        meta_prompt_path: str,
        documents: list[str] = None,
    ) -> tuple[str, list[dict]]:
        """
        Generate the complete 11-shot deck prompts from product images and documents.
        
        Args:
            image_paths: List of product image paths (all will be sent to the API)
            meta_prompt_path: Path to all.txt meta prompt file
            documents: Optional list of document paths
            
        Returns:
            (raw_analysis, prompts) tuple where prompts are Nano Banana natural language
        """
        meta_prompt = load_file(meta_prompt_path)

        # Build content parts - add ALL images
        contents = []
        for img_path in image_paths:
            image_bytes, mime_type = load_image(img_path)
            contents.append(self.client.create_image_part(image_bytes, mime_type))
        
        print(f"      📷 Sending {len(image_paths)} images to API")

        # Add documents, track temp files for cleanup
        temp_files = []
        for doc_path in (documents or []):
            doc_bytes, doc_mime, temp_file = load_document(Path(doc_path))
            contents.append(self.client.create_image_part(doc_bytes, doc_mime))
            if temp_file:
                temp_files.append(temp_file)
        
        if documents:
            print(f"      📄 Sending {len(documents)} documents to API")

        # Add the meta prompt
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

        return result.text, self._extract_nano_banana_prompts(result.text)

    def _extract_nano_banana_prompts(self, text: str) -> list[dict]:
        """
        Extract Nano Banana natural language prompts from the generated analysis text.
        
        The V4.1 format looks like:
        ### 1. 封面图 (Hero Shot)
        * **👁️ 视觉战术思考 (Visual Logic):** ...
        * **🍌 Nano Banana Prompt:** A professional product photograph of...
          square image, 1:1 aspect ratio, centered composition.
        """
        prompts = []
        
        # Pattern to match each section: ### [number]. [name] (with optional ** prefix)
        section_pattern = r"\*{0,2}###\s*(\d+)\.\s*(.+?)(?=\*{0,2}###\s*\d+\.|$)"
        sections = re.findall(section_pattern, text, re.DOTALL)
        
        for idx_str, section_content in sections:
            idx = int(idx_str)
            
            # Extract the image type name (first line before any newline or bullet)
            name_match = re.match(r"([^\n*]+)", section_content.strip())
            name = name_match.group(1).strip() if name_match else f"Image {idx}"
            
            prompt_text = None
            
            # Method 1: Look for "🍌 Nano Banana Prompt:" marker
            nano_match = re.search(
                r"🍌\s*Nano\s*Banana\s*Prompt[：:]\s*(.+?)(?=\n\s*(?:\*\*|###|$)|\n\n\n)",
                section_content,
                re.DOTALL | re.IGNORECASE
            )
            if nano_match:
                prompt_text = self._clean_nano_banana_prompt(nano_match.group(1))
            
            # Method 2: Look for code blocks with natural language (fallback)
            if not prompt_text:
                code_blocks = re.findall(r"```(?:\w*)?\s*([\s\S]*?)```", section_content)
                for block in code_blocks:
                    block = block.strip()
                    # Accept blocks that look like natural language prompts
                    if len(block) > 50 and not block.startswith("{"):
                        prompt_text = self._clean_nano_banana_prompt(block)
                        break
            
            # Method 3: Look for any English paragraph after the Visual Logic section
            if not prompt_text:
                # Find text after Visual Logic that looks like a prompt
                after_logic = re.search(
                    r"Visual Logic[）\)][：:]\*?\*?\s*[^\n]+\n+\s*\*?\s*\*?[^*]*?[：:]\s*(.+?)(?=\n\s*###|\Z)",
                    section_content,
                    re.DOTALL
                )
                if after_logic:
                    candidate = after_logic.group(1).strip()
                    # Check if it's English and long enough
                    if len(candidate) > 50 and re.search(r"[a-zA-Z]{3,}", candidate):
                        prompt_text = self._clean_nano_banana_prompt(candidate)
            
            if prompt_text:
                prompts.append({
                    "index": idx,
                    "name": name,
                    "prompt": prompt_text,
                    "type": self._get_image_type(idx),
                })
            else:
                print(f"      ⚠️  No prompt found for section {idx}: {name[:30]}")
        
        # Sort by index to ensure correct order
        prompts.sort(key=lambda x: x["index"])
        
        return prompts

    def _clean_nano_banana_prompt(self, raw_prompt: str) -> str:
        """Clean and normalize a Nano Banana natural language prompt."""
        # Remove any markdown formatting
        prompt = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw_prompt)  # Remove bold
        prompt = re.sub(r"\*([^*]+)\*", r"\1", prompt)  # Remove italic
        
        # Clean up whitespace - join lines into a single paragraph
        lines = [line.strip() for line in prompt.split("\n") if line.strip()]
        prompt = " ".join(lines)
        
        # Remove bullet points if present
        prompt = re.sub(r"^\s*[\*\-•]\s*", "", prompt)
        
        # Ensure the prompt ends with the required 1:1 aspect ratio hint if not present
        if "1:1" not in prompt and "square" not in prompt.lower():
            prompt = prompt.rstrip(".") + ". Square image, 1:1 aspect ratio, centered composition."
        
        return prompt.strip()

    def _get_image_type(self, idx: int) -> str:
        """Get the image type name by index (1-based)."""
        if 1 <= idx <= len(self.IMAGE_TYPES):
            return self.IMAGE_TYPES[idx - 1]
        return f"Image {idx}"
