# ecom

AI-powered product image generation for e-commerce.

## Setup

```bash
# Copy product catalog
cp -r "新格率免费体验精选品" catalog

# Install & configure
uv sync && source .venv/bin/activate
export GEMINI_API_KEY='...'
```

## Usage

```bash
python gemini_image.py -p PRODUCT_ID [-n NUM] [--model MODEL]
```

## Options

| Flag | Description |
|------|-------------|
| `-p` | Product ID (from `catalog/`) |
| `-i` | Direct image path |
| `-n` | Limit generated images |
| `--model` | `gemini` \| `gemini-3` \| `imagen` \| `imagen-ultra` |

## Pipeline

```mermaid
flowchart TD
    A[输入: 产品ID] --> B{查找产品目录}
    B --> C[LLM 选择主图]
    B --> D[LLM 选择文档]
    C --> E{prompts.json 存在?}
    D --> E
    E -->|否| F[阶段1: 图片+文档 → 生成 Prompts]
    E -->|是| G[加载缓存的 prompts]
    F --> H[阶段2: 生成图片]
    G --> H
    H --> I[保存结果]
```

## Supported Files

| Type | Extension | Note |
|------|-----------|------|
| Image | `.jpg` `.png` `.webp` `.gif` | LLM auto-selects main image |
| Document | `.pdf` `.txt` | Direct upload |
| Document | `.docx` `.doc` | Auto-convert to PDF |

## Output

```
outputs/{product_id}/
├── prompts.json           # Cached prompts
└── {timestamp}/
    ├── reference.jpg
    ├── results.json
    └── *.jpg
```

## Similarity Check

Compare image embeddings using CLIP model.

```bash
# All pairs
python test_similarity.py similarity_tests/my_folder/

# Target mode (compare one vs others)
python test_similarity.py similarity_tests/my_folder/ -t target.jpg
```

Test images go in `similarity_tests/` (gitignored).
