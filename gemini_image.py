#!/usr/bin/env python3
"""Generate e-commerce marketing images using Gemini AI.

Usage:
    # Generate cover images (主图) - default
    python gemini_image.py catalog/PRODUCT_ID/image.jpg

    # Generate preview images (预览图)
    python gemini_image.py --mode preview catalog/PRODUCT_ID/image.jpg

    # With documents
    python gemini_image.py --mode cover catalog/PRODUCT_ID/image.jpg catalog/PRODUCT_ID/doc.pdf
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ecom import Pipeline, Config
from ecom.config import MODELS, MODES

# 目标文件夹根目录
MANUAL_OUTPUTS_DIR = Path("/Users/sihanchen/Desktop/ecom/manual_outputs")

# 支持的文件扩展名
FILE_EXTENSIONS = r'\.(jpg|jpeg|png|gif|webp|bmp|pdf|docx?|xlsx?|pptx?|txt|md)'


def parse_file_paths(args: list[str]) -> list[str]:
    """
    智能解析文件路径，支持拖拽多个文件到终端。
    
    拖拽多个文件时，shell 会把 'path1''path2' 合并成一个字符串。
    这个函数通过识别文件扩展名来分割路径。
    """
    result = []
    for arg in args:
        # 检查是否包含多个路径（通过扩展名后紧跟 / 来判断）
        # 例如: /path/file.jpg/path/file2.png
        if re.search(FILE_EXTENSIONS + r'/', arg, re.IGNORECASE):
            # 在扩展名后面、下一个路径开始前插入分隔符
            # .jpg/Users/... -> .jpg|||/Users/...
            split_arg = re.sub(
                FILE_EXTENSIONS + r'(?=/)',
                r'\g<0>|||',
                arg,
                flags=re.IGNORECASE
            )
            paths = [p.strip() for p in split_arg.split('|||') if p.strip()]
            result.extend(paths)
        else:
            result.append(arg)
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        help="File paths - drag multiple files directly, they'll be auto-parsed"
    )
    parser.add_argument(
        "--mode",
        choices=MODES.keys(),
        default="cover",
        help="Generation mode: 'cover' (主图-创意风格) or 'preview' (预览图-写实还原). Default: cover"
    )
    parser.add_argument(
        "-m", "--model",
        choices=MODELS.keys(),
        default="gemini-3",
        help="Model to use for image generation (default: gemini-3)"
    )
    parser.add_argument(
        "--catalog",
        default="catalog",
        help="Base catalog directory (default: catalog)"
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't open the output folder in Finder after completion"
    )
    parser.add_argument(
        "-t", "--target",
        help="Target folder name - copy results to manual_outputs/<target>/<task_id>"
    )

    args = parser.parse_args()

    # 智能解析文件路径（支持拖拽多个文件）
    files = parse_file_paths(args.files)
    
    if len(files) != len(args.files):
        print(f"📁 Parsed {len(files)} files from input")

    # Create config
    config = Config.with_model(args.model, catalog_dir=args.catalog)
    
    # Run pipeline
    pipeline = Pipeline(config)
    result = pipeline.run(files, mode=args.mode)

    # Summary
    print(f"\n📊 Summary:")
    print(f"   Product ID: {result['product_id']}")
    print(f"   Mode: {result['mode']}")
    if 'prompts_count' in result:
        print(f"   Prompts generated: {result['prompts_count']}")
    print(f"   Images generated: {result['images_generated']}")
    
    success = sum(1 for r in result["results"] if r["images"])
    failed = len(result["results"]) - success
    if failed > 0:
        print(f"   ⚠️  Failed: {failed}")

    # Output directory
    output_dir = Path(config.output_dir) / result['product_id'] / result['mode'] / result['task_id']
    
    # Copy to target folder if specified
    final_dir = output_dir
    if args.target:
        target_dir = MANUAL_OUTPUTS_DIR / args.target / result['task_id']
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output_dir, target_dir)
        print(f"\n📋 Copied to: {target_dir}")
        final_dir = target_dir

    # Open output folder in Finder (macOS)
    if not args.no_open and final_dir.exists():
        print(f"📂 Opening folder in Finder...")
        subprocess.run(["open", str(final_dir)])


if __name__ == "__main__":
    main()
