# Gemini Image Generator

Generate product marketing images using Google Gemini AI with reference images.

## Setup

1. **Install dependencies with uv:**

```bash
uv venv
uv sync
```

2. **Set your Gemini API key:**

```bash
export GEMINI_API_KEY='your-api-key-here'
```

Or add it to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
echo 'export GEMINI_API_KEY=your-api-key-here' >> ~/.zshrc
source ~/.zshrc
```

Get your API key from: https://aistudio.google.com/app/apikey

## Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the script
python gemini_image.py <image_path> <prompts_json_path>
```

### Example

```bash
python gemini_image.py ./product.jpg ./prompts.json
```

## Prompts Format

Create a `prompts.json` file with your image generation prompts:

```json
[
    "Commercial Product Photography, 8k resolution...",
    "Another prompt here...",
    "Third prompt..."
]
```

## Output

Each run creates a unique task folder in `outputs/`:

```
outputs/20260101_152503_3d541c36/
├── reference.png           # Your original reference image
├── prompt_1_image_1.png    # Generated image for prompt 1
├── prompt_2_image_1.png    # Generated image for prompt 2
├── prompt_3_image_1.png    # Generated image for prompt 3
└── results.json            # Task results summary
```
