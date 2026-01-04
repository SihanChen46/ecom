#!/usr/bin/env python3
"""比较图片相似度"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

CLIP_MODEL = "openai/clip-vit-base-patch32"


class CLIPEmbedder:
    def __init__(self):
        print("🔄 加载 CLIP 模型...")
        self.model = CLIPModel.from_pretrained(CLIP_MODEL)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"   设备: {self.device}\n")
    
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        embedding = features / features.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.linalg.norm(v1 - v2))


def get_images_from_dir(directory: str) -> list[str]:
    img_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    return sorted([
        str(f) for f in Path(directory).iterdir()
        if f.is_file() and f.suffix.lower() in img_extensions
    ])


def compare_images(image_paths: list[str], target: str = None):
    embedder = CLIPEmbedder()
    
    # 获取 embeddings
    print("📷 图片列表:")
    embeddings = {}
    for path in image_paths:
        name = Path(path).name
        img = Image.open(path).convert("RGB")
        embeddings[name] = embedder.get_embedding(img)
        marker = " ← target" if target and Path(target).name == name else ""
        print(f"   {name}{marker}")
    
    names = list(embeddings.keys())
    print("\n" + "=" * 80)
    
    if target:
        # Target 模式
        target_name = Path(target).name
        others = [n for n in names if n != target_name]
        
        print(f"📊 Target: {target_name}")
        print("=" * 80)
        print(f"{'vs':<45} {'cosine':>12} {'euclidean':>12}")
        print("-" * 80)
        
        results = [(n, cosine_similarity(embeddings[target_name], embeddings[n]),
                      euclidean_distance(embeddings[target_name], embeddings[n])) for n in others]
        results.sort(key=lambda x: x[1], reverse=True)
        
        for name, sim, dist in results:
            print(f"{name:<45} {sim:>12.4f} {dist:>12.4f}")
    
    else:
        # All 模式
        print("📊 All pairs")
        print("=" * 80)
        
        for i, name1 in enumerate(names):
            others = names[i+1:]
            if not others:
                continue
            
            print(f"\n┌─ {name1}")
            print(f"│  {'vs':<43} {'cosine':>12} {'euclidean':>12}")
            print("│  " + "-" * 69)
            
            results = [(n, cosine_similarity(embeddings[name1], embeddings[n]),
                          euclidean_distance(embeddings[name1], embeddings[n])) for n in others]
            results.sort(key=lambda x: x[1], reverse=True)
            
            for j, (name2, sim, dist) in enumerate(results):
                prefix = "└─" if j == len(results) - 1 else "├─"
                print(f"│  {prefix} {name2:<41} {sim:>12.4f} {dist:>12.4f}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="比较图片相似度")
    parser.add_argument("dir", help="图片文件夹路径")
    parser.add_argument("--target", "-t", help="target 模式：只比较该图片与其他图片")
    
    args = parser.parse_args()
    
    if not Path(args.dir).is_dir():
        sys.exit(f"Error: 目录不存在: {args.dir}")
    
    images = get_images_from_dir(args.dir)
    if len(images) < 2:
        sys.exit("Error: 目录内图片少于 2 张")
    
    target = None
    if args.target:
        target = str(Path(args.dir) / args.target) if not Path(args.target).exists() else args.target
        if not Path(target).exists():
            sys.exit(f"Error: target 不存在: {args.target}")
        if target not in images:
            images.append(target)
    
    compare_images(images, target=target)


if __name__ == "__main__":
    main()
