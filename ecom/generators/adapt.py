"""Adapt mode - combine target image composition with product image colors."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..client import GeminiClient
from ..utils import load_file, load_image


ADAPT_INSTRUCTION = """生成一张新图片，要求如下：

## 核心任务
复制第一张图（目标图）的所有内容，但把颜色换成第二张图（产品图）的颜色。

## 严格规则

### 必须100%保持不变（全部来自目标图）：
- 所有物体的位置、角度、形状、细节
- 整体构图、光影、视角
- 背景和所有装饰元素的布局

### 只从产品图提取颜色，忽略其他一切：
- 只提取产品图中产品的主色调
- 忽略产品图的角度、光线、构图、背景等所有其他信息
- 产品图仅作为"色卡"使用

### 颜色应用：
- 将目标图中所有元素的颜色统一换成产品图的主色调同色系

## 输出要求
生成的图片应该是目标图的"换色版本"——除了颜色不同，其他一切都与目标图完全相同。正方形图片，1:1比例。"""


class AdaptGenerator:
    """
    Generator for 'adapt' mode that combines target image composition 
    with product image colors.
    """
    
    def __init__(self, client: GeminiClient, max_workers: int = 10):
        self.client = client
        self.max_workers = max_workers

    def generate(
        self,
        target_image: str,
        product_images: list[str],
        output_dir: Path,
    ) -> list[dict]:
        """
        Generate adapted images by combining target composition with product colors.
        
        Args:
            target_image: Path to target image (composition/style reference)
            product_images: List of product image paths (color/form reference)
            output_dir: Directory to save generated images
            
        Returns:
            List of result dicts with generated image paths
        """
        # Load target image
        target_bytes, target_mime = load_image(target_image)
        target_part = self.client.create_image_part(target_bytes, target_mime)
        
        total = len(product_images)
        results = [None] * total
        
        print(f"      🎨 Target image: {Path(target_image).name}")
        print(f"      🚀 Adapting {total} product images in parallel (max {self.max_workers} workers)...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._adapt_one, idx, target_part, product_path, output_dir
                ): idx
                for idx, product_path in enumerate(product_images)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    status = "✅" if result["images"] else "⚠️"
                    print(f"      [{idx+1}/{total}] {status} {result['product_name'][:50]}")
                except Exception as e:
                    results[idx] = {
                        "index": idx + 1,
                        "product_name": Path(product_images[idx]).name,
                        "product_path": product_images[idx],
                        "images": [],
                        "error": str(e)
                    }
                    print(f"      [{idx+1}/{total}] ❌ Error: {e}")
        
        return results

    def _adapt_one(
        self,
        idx: int,
        target_part,
        product_path: str,
        output_dir: Path,
    ) -> dict:
        """Adapt a single product image."""
        product_name = Path(product_path).name
        result = {
            "index": idx + 1,
            "product_name": product_name,
            "product_path": product_path,
            "images": [],
            "error": None
        }
        
        try:
            # Load product image
            product_bytes, product_mime = load_image(product_path)
            product_part = self.client.create_image_part(product_bytes, product_mime)
            
            # Generate adapted image: target (composition) + product (color) + instruction
            img_result = self.client.generate_image([
                target_part,
                product_part,
                ADAPT_INSTRUCTION
            ])
            
            if not img_result.images:
                result["error"] = "No images generated"
                return result
            
            for img_data, mime in img_result.images:
                ext = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }.get(mime, ".png")
                
                # Use product filename as base
                safe_name = re.sub(r"[^\w\-]", "_", Path(product_path).stem)[:40]
                filepath = output_dir / f"{idx + 1:02d}_adapt_{safe_name}{ext}"
                filepath.write_bytes(img_data)
                result["images"].append(str(filepath))
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
