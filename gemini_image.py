#!/usr/bin/env python3
"""
Gemini Image Generation Script
Send a reference image with prompts to Gemini and generate new images.
"""

import json
import sys
import os
import mimetypes
import uuid
import base64
from pathlib import Path
from datetime import datetime

from google import genai
from google.genai import types


def get_api_key() -> str:
    """Get API key from environment variable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable is not set.")
        print("\nPlease set it:")
        print("  export GEMINI_API_KEY='your-api-key'")
        print("\nOr create a .env file and source it:")
        print("  echo 'export GEMINI_API_KEY=your-api-key' > .env")
        print("  source .env")
        sys.exit(1)
    return api_key


def generate_task_id() -> str:
    """Generate a unique task ID with timestamp and short UUID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"


def load_prompts(json_path: str) -> list[str]:
    """Load prompts from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Support both list format and dict with "prompts" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "prompts" in data:
        return data["prompts"]
    else:
        raise ValueError("JSON must be a list of prompts or a dict with 'prompts' key")


def load_image(image_path: str) -> tuple[bytes, str]:
    """Load image and return bytes with mime type."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Detect mime type
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/jpeg"
    
    with open(path, "rb") as f:
        image_bytes = f.read()
    
    return image_bytes, mime_type


def save_image_from_response(response, output_dir: Path, prompt_index: int) -> list[str]:
    """Extract and save images from Gemini response."""
    saved_files = []
    
    if not response.candidates:
        return saved_files
    
    for candidate in response.candidates:
        if not candidate.content or not candidate.content.parts:
            continue
            
        for part_idx, part in enumerate(candidate.content.parts):
            # Check if this part contains image data
            if hasattr(part, 'inline_data') and part.inline_data:
                inline_data = part.inline_data
                mime_type = inline_data.mime_type if hasattr(inline_data, 'mime_type') else "image/png"
                
                # Determine file extension
                ext_map = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/jpg": ".jpg",
                    "image/webp": ".webp",
                    "image/gif": ".gif",
                }
                ext = ext_map.get(mime_type, ".png")
                
                # Save the image
                filename = f"prompt_{prompt_index + 1}_image_{part_idx + 1}{ext}"
                filepath = output_dir / filename
                
                # Get the image data
                if hasattr(inline_data, 'data'):
                    image_data = inline_data.data
                    if isinstance(image_data, str):
                        # Base64 encoded
                        image_data = base64.b64decode(image_data)
                    
                    with open(filepath, "wb") as f:
                        f.write(image_data)
                    
                    saved_files.append(str(filepath))
                    print(f"  💾 Saved: {filepath}")
    
    return saved_files


def query_gemini(image_path: str, prompts: list[str], task_id: str) -> dict:
    """
    Send image with multiple prompts to Gemini and return responses.
    Saves generated images to task folder.
    """
    # Create output directory
    output_dir = Path("outputs") / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Create client
    client = genai.Client(api_key=get_api_key())
    
    # Load reference image
    image_bytes, mime_type = load_image(image_path)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    
    # Copy reference image to output dir
    ref_image_path = output_dir / f"reference{Path(image_path).suffix}"
    with open(ref_image_path, "wb") as f:
        f.write(image_bytes)
    print(f"📷 Reference image copied to: {ref_image_path}")
    
    results = {}
    
    for idx, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Prompt {idx + 1}: {prompt[:80]}...")
        print(f"{'='*60}")
        
        try:
            # Send request with image and prompt
            # Use gemini-2.0-flash-exp for image generation capability
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"]
                )
            )
            
            # Extract text response
            text_response = ""
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_response += part.text
            
            # Save any generated images
            saved_images = save_image_from_response(response, output_dir, idx)
            
            results[f"prompt_{idx + 1}"] = {
                "prompt": prompt,
                "text_response": text_response if text_response else None,
                "saved_images": saved_images,
                "success": True
            }
            
            if text_response:
                print(f"📝 Text response: {text_response[:200]}...")
            if saved_images:
                print(f"🖼️  Generated {len(saved_images)} image(s)")
            
        except Exception as e:
            results[f"prompt_{idx + 1}"] = {
                "prompt": prompt,
                "error": str(e),
                "success": False
            }
            print(f"❌ Error: {e}")
    
    # Save results summary to JSON
    summary_path = output_dir / "results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, indent=2, ensure_ascii=False, fp=f)
    print(f"\n📄 Results saved to: {summary_path}")
    
    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: python gemini_image.py <image_path> <prompts_json_path>")
        print("\nExample:")
        print("  python gemini_image.py ./my_image.jpg ./prompts.json")
        print("\nPrompts JSON format:")
        print('  ["Generate a variation of this image", "Create a marketing photo"]')
        sys.exit(1)
    
    image_path = sys.argv[1]
    prompts_json_path = sys.argv[2]
    
    # Generate unique task ID
    task_id = generate_task_id()
    
    print(f"🚀 Task ID: {task_id}")
    print(f"📷 Image: {image_path}")
    print(f"📋 Prompts file: {prompts_json_path}")
    
    # Load prompts
    prompts = load_prompts(prompts_json_path)
    print(f"📝 Loaded {len(prompts)} prompts")
    
    # Query Gemini
    results = query_gemini(image_path, prompts, task_id)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TASK SUMMARY")
    print("="*60)
    print(f"Task ID: {task_id}")
    
    success_count = sum(1 for r in results.values() if r.get("success"))
    total_images = sum(len(r.get("saved_images", [])) for r in results.values())
    
    print(f"Successful prompts: {success_count}/{len(prompts)}")
    print(f"Total images generated: {total_images}")
    print(f"Output folder: outputs/{task_id}/")
    
    return results


if __name__ == "__main__":
    main()
