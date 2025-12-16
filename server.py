import argparse
import base64
import io
import json
import logging
import os
from datetime import datetime
import sys
import time
import requests
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
import torch.amp.autocast_mode
import torchvision.transforms.functional as TVF
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
import pandas as pd
import cv2
import numpy as np

try:
    import onnxruntime as ort
    SMILINGWOLF_AVAILABLE = True
except ImportError:
    SMILINGWOLF_AVAILABLE = False
    ort = None

# Constants

# Constants
JOYTAG_REPO = "fancyfeast/joytag"
SMILINGWOLF_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
SMILINGWOLF_NAME = "wd-eva02-large-tagger-v3"
CONVNEXT_REPO = "SmilingWolf/wd-convnext-tagger-v3"
CONVNEXT_NAME = "wd-convnext-tagger-v3"

# Global variables
current_model_name = "joytag"
models = {}
device = "cuda" if torch.cuda.is_available() else "cpu"
threshold = 0.2

CONFIG_FILE = "config.json"

def load_config_file():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
    return {}

def save_config_file(config_data):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save config file: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load JoyTag by default
    models["joytag"] = JoyTagWrapper(device, threshold)
    
    # Load SmilingWolf Models
    try:
        logger.info(f"Loading {SMILINGWOLF_NAME}...")
        models[SMILINGWOLF_NAME] = SmilingWolfWrapper(device, threshold, SMILINGWOLF_REPO, SMILINGWOLF_NAME)
        logger.info(f"{SMILINGWOLF_NAME} loaded.")
        
        logger.info(f"Loading {CONVNEXT_NAME}...")
        models[CONVNEXT_NAME] = SmilingWolfWrapper(device, threshold, CONVNEXT_REPO, CONVNEXT_NAME)
        logger.info(f"{CONVNEXT_NAME} loaded.")
    except Exception as e:
        logger.error(f"Failed to load SmilingWolf models: {e}")
        # Non-fatal if at least one model loads, but we warned.
    
    # Apply persisted config
    saved_config = load_config_file()
    for model_name, model_config in saved_config.items():
        if model_name in models:
            logger.info(f"Applying saved config for {model_name}: {model_config}")
            models[model_name].update_config(model_config)

    yield
    models.clear()

app = FastAPI(title="Onnx API endpoint", lifespan=lifespan)

class ModelWrapper:
    def predict(self, image: Image.Image):
        raise NotImplementedError
    
    def get_config(self):
        return {"threshold": self.threshold}

    def update_config(self, config: Dict[str, Any]):
        if "threshold" in config:
            self.threshold = float(config["threshold"])
        if "character_threshold" in config:
            self.character_threshold = float(config["character_threshold"])

class SmilingWolfWrapper(ModelWrapper):
    def __init__(self, device, threshold, repo_id, model_name):
        self.device = device
        self.threshold = threshold
        self.character_threshold = threshold + 0.1 # Default higher for chars?
        self.repo_id = repo_id
        self.model_name = model_name
        self.image_size = 448
        self.session = None
        self.tags_df = None
        self.load()

    def load(self):
        if not SMILINGWOLF_AVAILABLE:
            raise ImportError("SmilingWolf dependencies (onnxruntime-gpu, pandas, opencv-python) are not installed.")
            
        logger.info(f"Downloading/Loading {self.model_name} from {self.repo_id}...")
        try:
            # Use unique directory per model to avoid overwriting model.onnx
            local_model_dir = Path(f"models/{self.model_name}")
            model_path = snapshot_download(
                repo_id=self.repo_id,
                local_dir=local_model_dir,
                local_dir_use_symlinks=False
            )
            
            onnx_path = Path(model_path) / "model.onnx"
            tags_path = Path(model_path) / "selected_tags.csv"

            if not onnx_path.exists():
                raise FileNotFoundError(f"model.onnx not found at {onnx_path}")
            if not tags_path.exists():
                raise FileNotFoundError(f"selected_tags.csv not found at {tags_path}")

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            self.tags_df = pd.read_csv(tags_path)
            logger.info("SmilingWolf model loaded.")
        except Exception as e:
            logger.error(f"Failed to load SmilingWolf: {e}")
            raise e

    def prepare_image(self, image: Image.Image) -> np.ndarray:
        # Convert to numpy (RGB)
        img = np.array(image.convert('RGB'))
        
        target_size = self.image_size
        
        # Pillow resize logic (pad to square) to match aspect ratio preservation
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))
        
        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
            
        img = np.array(padded_image).astype(np.float32)
        
        # RGB to BGR for CV2 style models (common in WD taggers)
        img = img[:, :, ::-1] 
        
        # Add batch dimension: (1, H, W, 3)
        img = np.expand_dims(img, 0)
        return img

    def predict(self, image: Image.Image, prompt: str = None):
        if not self.session: return "Model not loaded"
        
        img_input = self.prepare_image(image)
        
        # Check input name
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        # Preprocessing if model expects certain format (e.g. NCHW vs NHWC)
        # WD taggers generally expect NHWC (1, H, W, 3) or NCHW
        # Let's inspect shape
        input_shape = self.session.get_inputs()[0].shape
        
        # If NCHW (1, 3, 448, 448)
        if len(input_shape) == 4 and input_shape[1] == 3:
             img_input = img_input.transpose(0, 3, 1, 2)
        
        probs = self.session.run([output_name], {input_name: img_input})[0]
        
        # probs shape (1, num_tags)
        
        # NEW: Filter with category-specific thresholds
        # Usually category 0 = General, 4 = Character (in Danbooru based models)
        # SmilingWolf models usually have 'category' column in selected_tags.csv
        
        final_tags = []
        
        has_category = 'category' in self.tags_df.columns
        
        for i, prob in enumerate(probs[0]):
             if i >= len(self.tags_df): break
             
             score = float(prob)
             row = self.tags_df.iloc[i]
             tag_name = row['name']
             category = row['category'] if has_category else 0
             
             # Determine active threshold
             active_thresh = self.threshold
             if category == 4: # Character
                 active_thresh = self.character_threshold
            
             if score > active_thresh:
                 final_tags.append((tag_name, score))
                 
        # Sort by score descending
        final_tags.sort(key=lambda x: x[1], reverse=True)
        return ', '.join([t[0] for t in final_tags])

    def get_config(self):
        return {
            "threshold": self.threshold,
            "character_threshold": self.character_threshold
        }

    def update_config(self, config: Dict[str, Any]):
        if "threshold" in config:
            self.threshold = float(config["threshold"])
        if "character_threshold" in config:
            self.character_threshold = float(config["character_threshold"])

class JoyTagWrapper(ModelWrapper):
    def __init__(self, device: str, threshold: float):
        self.device = device
        self.threshold = threshold
        self.model = None
        self.top_tags = []
        self.load()

    def load(self):
        logger.info(f"Downloading/Loading JoyTag model from {JOYTAG_REPO}...")
        try:
            local_model_dir = Path("model")
            model_path = snapshot_download(
                repo_id=JOYTAG_REPO,
                local_dir=local_model_dir,
                local_dir_use_symlinks=False
            )
            
            # Ensure sys.path includes the model directory for importing Models.py
            if str(model_path) not in sys.path:
                sys.path.append(str(model_path))
            
            from Models import VisionModel
            
            self.model = VisionModel.load_model(model_path)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            with open(Path(model_path) / 'top_tags.txt', 'r', encoding='utf-8') as f:
                self.top_tags = [line.strip() for line in f.readlines() if line.strip()]
                
            logger.info("JoyTag model loaded.")
        except Exception as e:
            logger.error(f"Failed to load JoyTag: {e}")
            raise e

    def prepare_image(self, image: Image.Image, target_size: int) -> torch.Tensor:
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
        
        image_tensor = TVF.pil_to_tensor(padded_image) / 255.0
        image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        return image_tensor

    @torch.no_grad()
    def predict(self, image: Image.Image, prompt: str = None):
        image_tensor = self.prepare_image(image, self.model.image_size)
        batch = {'image': image_tensor.unsqueeze(0).to(self.device)}

        with torch.amp.autocast_mode.autocast(self.device, enabled=True):
            preds = self.model(batch)
            tag_preds = preds['tags'].sigmoid().cpu()
        
        scores = {self.top_tags[i]: tag_preds[0][i] for i in range(len(self.top_tags))}
        predicted_tags = [tag for tag, score in scores.items() if score > self.threshold]
        # Sort by score descending
        predicted_tags.sort(key=lambda x: scores[x], reverse=True)
        return ', '.join(predicted_tags)

class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContent], List[Dict[str, Any]]]
    images: Optional[List[str]] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None

# Ollama Models
class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: Optional[str] = ""
    images: Optional[List[str]] = None
    stream: bool = False
    options: Optional[Dict[str, Any]] = None

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    options: Optional[Dict[str, Any]] = None


def download_image(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")

def decode_base64_image(base64_string: str) -> Image.Image:
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64: {e}")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    requested_model = request.model
    
    # Debug Logging
    logger.info(f"Received chat completion request for model: {requested_model}")
    # Log structure of first user message to debug image extraction
    for msg in request.messages:
        if msg.role == "user":
            logger.info(f"User message content type: {type(msg.content)}")
            if isinstance(msg.content, list):
                logger.info(f"User message content list length: {len(msg.content)}")
                for i, item in enumerate(msg.content):
                    logger.info(f"Item {i} type: {type(item)}")
                    if isinstance(item, dict):
                        # Log keys but truncate values like huge base64
                        safe_item = {k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v) for k, v in item.items()}
                        logger.info(f"Item {i} dict: {safe_item}")
                    elif hasattr(item, 'model_dump'):
                         # It's a Pydantic model
                         safe_dump = item.model_dump()
                         if 'image_url' in safe_dump and safe_dump['image_url']:
                             if 'url' in safe_dump['image_url']:
                                 val = safe_dump['image_url']['url']
                                 safe_dump['image_url']['url'] = val[:50] + "..." if len(val) > 50 else val
                         logger.info(f"Item {i} model: {safe_dump}")
            else:
                 logger.info(f"User message content str: {msg.content[:100]}")
            break
    image = None
    for message in reversed(request.messages):
        if message.role == "user":
            content = message.content
            if isinstance(content, str):
                if content.startswith("http"):
                    image = download_image(content)
                    break
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, MessageContent):
                        item = item.model_dump()
                    
                    if isinstance(item, dict):
                        # CASE 1: Standard OpenAI "type": "image_url"
                        if item.get("type") == "image_url":
                            image_url_obj = item.get("image_url")
                            if image_url_obj and isinstance(image_url_obj, dict):
                                url = image_url_obj.get("url")
                                if url:
                                    if url.startswith("http"):
                                        image = download_image(url)
                                    elif "base64" in url or url.startswith("data:"):
                                        image = decode_base64_image(url)
                        
                        # CASE 2: Flat "image_url" string (MagicPrompt quirk?)
                        elif "image_url" in item and isinstance(item["image_url"], str):
                            url = item["image_url"]
                            if url.startswith("http"):
                                image = download_image(url)
                            elif "base64" in url or url.startswith("data:"):
                                image = decode_base64_image(url)
                                
                    if image: break
            if image: break
    if requested_model not in models:
        # Fallback to joytag if unknown, or error? 
        # For now, if it's not loaded, try to load or error.
        # But we only loaded joytag at startup. 
        if "joytag" in models:
            requested_model = "joytag"
        else:
            raise HTTPException(status_code=500, detail="No models loaded")

    image = None
    prompt_text = "Describe this image."
    
    for message in reversed(request.messages):
        if message.role == "user":
            content = message.content
            if isinstance(content, str):
                if content.startswith("http"):
                    image = download_image(content)
                elif "base64" in content or content.startswith("data:"):
                     image = decode_base64_image(content)
                else:
                     # Identify as prompt if it's not a URL/Image
                     if len(content) < 1000: # Heuristic
                         prompt_text = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, MessageContent):
                        item = item.model_dump()
                    
                    if isinstance(item, dict):
                        # Extract Text
                        if item.get("type") == "text":
                            if "text" in item:
                                prompt_text = item["text"]
                        
                        # Extract Image
                        if item.get("type") == "image_url":
                            image_url_obj = item.get("image_url")
                            if image_url_obj:
                                url = image_url_obj.get("url")
                                if url.startswith("http"):
                                    image = download_image(url)
                                elif "base64" in url or url.startswith("data:"):
                                    image = decode_base64_image(url)
                            
                        # MagicPrompt or other quirks
                        elif "image_url" in item and isinstance(item["image_url"], str):
                            # ... logic already here ...
                            url = item["image_url"]
                            if url.startswith("http"):
                                image = download_image(url)
                            elif "base64" in url or url.startswith("data:"):
                                image = decode_base64_image(url)
                                
            # Check for generic images list (Ollama style)
            if hasattr(message, 'images') and message.images:
                 img_str = message.images[0]
                 image = decode_base64_image(img_str)

            if image: break
    
    if not image:
        return {
            "id": "chatcmpl-error",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "No image found."}}],
        }

    tag_string = models[requested_model].predict(image, prompt=prompt_text)

    return {
        "id": "chatcmpl-joytag",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": tag_string
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user"
            } for name in models.keys()
        ]
    }

# --- Ollama Endpoints ---

@app.get("/api/tags")
async def ollama_tags():
    return {
        "models": [
            {
                "name": name,
                "model": name,
                "modified_at": datetime.utcnow().isoformat() + "Z",
                "size": 0,
                "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "vision",
                    "families": ["vision"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                }
            } for name in models.keys()
        ]
    }

@app.post("/api/generate")
async def ollama_generate(request: OllamaGenerateRequest):
    requested_model = request.model
    # Model selection logic (fallback to joytag if unknown/default)
    if requested_model not in models:
        requested_model = "joytag" if "joytag" in models else None
    
    if not requested_model or requested_model not in models:
         raise HTTPException(status_code=500, detail="Model not found")

    image = None
    if request.images and len(request.images) > 0:
        # Ollama passes base64 strings in 'images' list
        image = decode_base64_image(request.images[0])
    
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")

    # Use prompt from request or default
    tag_string = models[requested_model].predict(image, prompt=request.prompt or "Describe this image.")
    
    return {
        "model": requested_model,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "response": tag_string,
        "done": True,
        "context": [],
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0
    }

@app.post("/api/chat")
async def ollama_chat(request: OllamaChatRequest):
    # Reuse the same logic as OpenAI chat but return Ollama format
    requested_model = request.model
    if requested_model not in models:
        requested_model = "joytag" if "joytag" in models else None
    
    if not requested_model or requested_model not in models:
         raise HTTPException(status_code=500, detail="Model not found")

    image = None
    for message in reversed(request.messages):
         if message.role == "user":
            content = message.content
            # Handle list content (Ollama might send text/image separately? 
            # Actually standard Ollama client sends images list in generate, 
            # but in chat it follows OpenAI vision format mostly or has 'images' field in message)
            # Pydantic model 'Message' matches OpenAI structure.
            # But standard Ollama raw API for chat supports 'images' list in message object.
            # fastAPI Pydantic 'Message' we defined: content is Union[str, List...].
            # If Ollama sends 'images' key in message, our current Message model might drop it 
            # unless we add it to Message model?
            # Let's inspect Message model.
            pass

    # Quick fix: The 'Message' model we use is strict. 
    # But we can try to reuse the logic. 
    # If the user uses standard OpenAI-vision compatible schema for 'images', it works.
    # If they use Ollama 'images' key: we might miss it.
    
    # Let's improve image extraction for the shared Message model or just iterate dictionary if it was a dict
    # But request.messages is List[Message].
    
    # We will assume Ollama Chat request follows the same image embedding (base64 in content or similar)
    # OR we need to update Message definition to allow 'images'.
    
    # Let's rely on the shared logic for now, utilizing the fact that most clients adaption for vision 
    # using OpenAI format.
    
    # Re-using the logic from chat_completions is hard because it's inside the function.
    # Let's extract image finding logic? Or just duplicate it for safety.
    image = None
    prompt_text = "Describe this image."

    for message in reversed(request.messages):
        if message.role == "user":
            content = message.content
            
            # Check for Ollama-style 'images' list first (specific to Ollama mode)
            if message.images and len(message.images) > 0:
                 img_data = message.images[0]
                 if img_data.startswith("http"):
                     image = download_image(img_data)
                 else:
                     image = decode_base64_image(img_data)
            
            # Text extraction from content string
            if isinstance(content, str):
                if content.startswith("http"):
                    if not image: image = download_image(content)
                elif "base64" in content or content.startswith("data:"):
                     if not image: image = decode_base64_image(content)
                else:
                     if len(content) < 1000:
                         prompt_text = content

            # List content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, MessageContent):
                        item = item.model_dump()
                    
                    if isinstance(item, dict):
                        if item.get("type") == "text" and "text" in item:
                            prompt_text = item["text"]
                        
                        if not image:
                            if item.get("type") == "image_url":
                                image_url_obj = item.get("image_url")
                                if image_url_obj:
                                    url = image_url_obj.get("url")
                                    if url.startswith("http"):
                                        image = download_image(url)
                                    elif "base64" in url or url.startswith("data:"):
                                        image = decode_base64_image(url)
                        
                        # CASE 2: Flat "image_url" string (MagicPrompt quirk?)
                        elif "image_url" in item and isinstance(item["image_url"], str):
                            url = item["image_url"]
                            if url.startswith("http"):
                                image = download_image(url)
                            elif "base64" in url or url.startswith("data:"):
                                image = decode_base64_image(url)

                    if image: break
            if image: break
            
    if not image:
         # Try to check if request has top level images (not standard for chat but possible?)
         pass

    if not image:
        return {
            "model": requested_model,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "message": { "role": "assistant", "content": "No image found." },
            "done": True
        }

    tag_string = models[requested_model].predict(image)

    return {
        "model": requested_model,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "message": {
            "role": "assistant",
            "content": tag_string
        },
        "done": True,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0
    }

@app.get("/api/config")
async def get_config():
    configs = {}
    for name, model in models.items():
        configs[name] = model.get_config()
    return configs

class ConfigUpdate(BaseModel):
    threshold: Optional[float] = None
    character_threshold: Optional[float] = None

@app.post("/api/config/{model_name}")
async def update_config(model_name: str, config: ConfigUpdate):
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    update_data = config.model_dump(exclude_unset=True)
    models[model_name].update_config(update_data)
    
    # Persist change
    current_saved_config = load_config_file()
    if model_name not in current_saved_config:
        current_saved_config[model_name] = {}
    
    current_saved_config[model_name].update(update_data)
    save_config_file(current_saved_config)

    return {"status": "updated", "config": models[model_name].get_config()}

app.mount("/", StaticFiles(directory="static", html=True), name="static")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--threshold", type=float, default=0.2, help="Tagging threshold (default 0.2)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    else:
        device = args.device
    
    threshold = args.threshold

    uvicorn.run(app, host=args.host, port=args.port)
