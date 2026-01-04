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
    C --> D{prompts.json 存在?}
    D -->|否| E[阶段1: 生成 Prompts]
    E --> F[保存 prompts.json]
    F --> G[阶段2: 生成图片]
    D -->|是| H[加载缓存的 prompts]
    H --> G
    G --> I[保存结果]
```

## Output

```
outputs/{product_id}/
├── prompts.json           # Cached prompts
└── {timestamp}/
    ├── reference.jpg
    ├── results.json
    └── *.jpg
```
