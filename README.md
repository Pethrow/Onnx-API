# JoyTag + WD Tagger Endpoint (OpenAI/Ollama Compatible)

A standardized inference server for image interrogation/tagging using **JoyTag** and **SmilingWolf (WD)** models. Designed to work with **SwarmUI**, **SillyTavern**, and **OpenAI-compatible** clients.

## Features
- **OpenAI Compatible**: Use `/v1/chat/completions` with vision-capable clients (e.g. SwarmUI).
- **Ollama Compatible**: Use `/api/generate` or `/api/chat` with Ollama-friendly tools like SillyTavern.
- **Lazy Loading**: Models are downloaded and loaded into memory only when first requested, saving startup time and resources.
- **Multi-Model Support**:
    - `joytag` (Alpha CLIP based, great for booru tags)
    - `wd-eva02-large-tagger-v3` (SmilingWolf ONNX)
    - `wd-vit-large-tagger-v3` (SmilingWolf ONNX)
- **Management UI**: Built-in web interface for testing and configuration.
- **Configurable**: Adjustable confidence thresholds per model (including separate character thresholds).

## Installation
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   python server.py --device cuda
   ```

## API Usage

### OpenAI (Vision)
- **Endpoint**: `/v1/chat/completions`
- **Model**: `joytag` or `wd-eva02-large-tagger-v3`

### Ollama
- **Endpoint**: `/api/generate` or `/api/chat`
- **Model**: `joytag`

## Models
| Name | Type | Description |
| --- | --- | --- |
| `joytag` | PyTorch | Balanced, good for general captioning. |
| `wd-eva02-large-tagger-v3` | ONNX | High precision, large tag set. |
| `wd-vit-large-tagger-v3` | ONNX | VIT-based tagger. |

## Credits
- **JoyTag**: `fancyfeast`
- **SmilingWolf Taggers**: `SmilingWolf`
