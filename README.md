# JoyTag + WD Tagger Endpoint (OpenAI/Ollama Compatible)

A standardized inference server for image interrogation/tagging using **JoyTag** and **SmilingWolf (WD)** models. Designed to work with **SwarmUI**, **SillyTavern**, and **OpenAI-compatible** clients.

## Features
- **Models Supported**:
    - `joytag` (Alpha CLIP based, great for booru tags)
    - `wd-eva02-large-tagger-v3` (SmilingWolf ONNX)
    - `wd-vit-large-tagger-v3` (SmilingWolf ONNX)
- **Protocols**:
    - **OpenAI Vision**: `/v1/chat/completions` (Compatible with SwarmUI MagicPromptExtension)
    - **Ollama**: `/api/generate`, `/api/chat` (Drop-in replacement for Ollama vision)
- **Management UI**: Built-in web interface for testing and configuration.
- **Configurable**: Adjustable confidence thresholds per model (including separate character thresholds).

## Installation

1.  **Windows**: Double-click `install.bat`.
    - Creates a `venv` and installs PyTorch/Transformers/ONNXRuntime.
    - *Dependencies: `git`, `python 3.10+`*.

## Usage

### 1. Start the Server
- **GPU (Recommended)**: Run `start_gpu.bat`
- **CPU**: Run `start_cpu.bat`
- Server listens on: `http://127.0.0.1:5000`

### 2. Management UI
Open `http://127.0.0.1:5000` in your browser.
- **Dashboard**: Check loaded models.
- **Settings**: Adjust thresholds (Global vs Character).
- **Test Console**: Drag specific images to test tagging.

### 3. API Usage

#### OpenAI (SwarmUI / SillyTavern)
- **Base URL**: `http://127.0.0.1:5000/v1`
- **API Key**: `any-string`
- **Model**: `joytag` or `wd-eva02-large-tagger-v3`

#### Ollama
- **Base URL**: `http://127.0.0.1:5000`
- **Model**: `joytag`

## Models
| Model ID | Type | Description |
| :--- | :--- | :--- |
| `joytag` | PyTorch | Balanced, good for general captioning. |
| `wd-eva02-large-tagger-v3` | ONNX | High accuracy for Danbooru tags. |
| `wd-vit-large-tagger-v3` | ONNX | Vision Transformer based. |

## Credits
- **JoyTag**: `fancyfeast`
- **Taggers**: `SmilingWolf`
- **Server**: Antigravity
