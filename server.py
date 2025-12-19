import argparse
import base64
import io
import json
import logging
import os
import sys
import time
import requests
import uvicorn
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
import pandas as pd
import cv2
import numpy as np
import torch
import torch.amp.autocast_mode
import torchvision.transforms.functional as TVF

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    SMILINGWOLF_AVAILABLE = True
except ImportError:
    SMILINGWOLF_AVAILABLE = False
    ort = None

# Constants
JOYTAG_REPO = "fancyfeast/joytag"
SMILINGWOLF_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
SMILINGWOLF_NAME = "wd-eva02-large-tagger-v3"
VIT_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
VIT_NAME = "wd-vit-large-tagger-v3"

CONFIG_FILE = "config.json"

# Global states
models = {}
model_lock = asyncio.Lock()
device = "cuda" if torch.cuda.is_available() else "cpu"
threshold = 0.2

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

class ModelWrapper:
    def __init__(self, device, threshold):
        self.device = device
        self.threshold = threshold
        self.loaded = False

    async def ensure_loaded(self):
        if not self.loaded:
            async with model_lock:
                if not self.loaded:
                    # Run the blocking load in a thread to keep the event loop free
                    await asyncio.to_thread(self.load)
                    self.loaded = True

    def load(self):
        raise NotImplementedError

    def predict(self, image: Image.Image, **kwargs):
        raise NotImplementedError
    
    def get_config(self):
        return {"threshold": self.threshold}

    def update_config(self, config: Dict[str, Any]):
        if "threshold" in config:
            self.threshold = float(config["threshold"])

class SmilingWolfWrapper(ModelWrapper):
    def __init__(self, device, threshold, repo_id, model_name):
        super().__init__(device, threshold)
        self.character_threshold = threshold + 0.1
        self.repo_id = repo_id
        self.model_name = model_name
        self.image_size = 448
        self.session = None
        self.tags_df = None

    def load(self):
        if not SMILINGWOLF_AVAILABLE:
            raise ImportError("SmilingWolf dependencies (onnxruntime, pandas, opencv) are not installed.")
            
        logger.info(f"Downloading/Loading {self.model_name} from {self.repo_id}...")
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
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.tags_df = pd.read_csv(tags_path)
        logger.info(f"{self.model_name} loaded.")

    def prepare_image(self, image: Image.Image) -> np.ndarray:
        target_size = self.image_size
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))
        
        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
            
        img = np.array(padded_image).astype(np.float32)
        img = img[:, :, ::-1] # RGB to BGR
        return np.expand_dims(img, 0)

    def predict(self, image: Image.Image, threshold: float = None, character_threshold: float = None, **kwargs):
        if not self.session: return "Model not loaded"
        
        img_input = self.prepare_image(image)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        probs = self.session.run([output_name], {input_name: img_input})[0]
        
        final_tags = []
        base_threshold = threshold if threshold is not None else self.threshold
        base_char_threshold = character_threshold if character_threshold is not None else self.character_threshold
        
        for i, prob in enumerate(probs[0]):
             if i >= len(self.tags_df): break
             score = float(prob)
             row = self.tags_df.iloc[i]
             category = row.get('category', 0)
             active_thresh = base_char_threshold if category == 4 else base_threshold
            
             if score > active_thresh:
                 final_tags.append((row['name'], score))
                 
        final_tags.sort(key=lambda x: x[1], reverse=True)
        return final_tags

    def get_config(self):
        config = super().get_config()
        config["character_threshold"] = self.character_threshold
        return config

    def update_config(self, config: Dict[str, Any]):
        super().update_config(config)
        if "character_threshold" in config:
            self.character_threshold = float(config["character_threshold"])

class JoyTagWrapper(ModelWrapper):
    def __init__(self, device: str, threshold: float):
        super().__init__(device, threshold)
        self.model = None
        self.top_tags = []

    def load(self):
        logger.info(f"Downloading/Loading JoyTag model from {JOYTAG_REPO}...")
        local_model_dir = Path("model")
        model_path = snapshot_download(
            repo_id=JOYTAG_REPO,
            local_dir=local_model_dir,
            local_dir_use_symlinks=False
        )
        
        if str(model_path) not in sys.path:
            sys.path.append(str(model_path))
        
        from Models import VisionModel
        self.model = VisionModel.load_model(model_path)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        with open(Path(model_path) / 'top_tags.txt', 'r', encoding='utf-8') as f:
            self.top_tags = [line.strip() for line in f.readlines() if line.strip()]
        logger.info("JoyTag model loaded.")

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
    def predict(self, image: Image.Image, threshold: float = None, **kwargs):
        image_tensor = self.prepare_image(image, self.model.image_size)
        batch = {'image': image_tensor.unsqueeze(0).to(self.device)}

        with torch.amp.autocast_mode.autocast(self.device, enabled=True):
            preds = self.model(batch)
            tag_preds = preds['tags'].sigmoid().cpu()
        
        active_thresh = threshold if threshold is not None else self.threshold
        scores = {self.top_tags[i]: float(tag_preds[0][i]) for i in range(min(len(self.top_tags), tag_preds.shape[1]))}
        predicted_tags = [(tag, score) for tag, score in scores.items() if score > active_thresh]
        predicted_tags.sort(key=lambda x: x[1], reverse=True)
        return predicted_tags

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize wrapper objects but DON'T LOAD model weights yet
    models["joytag"] = JoyTagWrapper(device, threshold)
    models[SMILINGWOLF_NAME] = SmilingWolfWrapper(device, threshold, SMILINGWOLF_REPO, SMILINGWOLF_NAME)
    models[VIT_NAME] = SmilingWolfWrapper(device, threshold, VIT_REPO, VIT_NAME)
    
    # Apply persisted config
    saved_config = load_config_file()
    for model_name, model_config in saved_config.items():
        if model_name in models:
            models[model_name].update_config(model_config)

    yield
    models.clear()

app = FastAPI(title="Onnx API endpoint", lifespan=lifespan)

# Helper for image extraction
def get_image_from_messages(messages: List[Any]) -> tuple[Optional[Image.Image], str]:
    image = None
    prompt_text = "Describe this image."
    
    for message in reversed(messages):
        # Handle Pydantic or dict
        role = getattr(message, 'role', None) or message.get('role')
        if role != "user": continue
        
        images = getattr(message, 'images', None) or message.get('images')
        if images and len(images) > 0:
            image = decode_base64_image(images[0])
            if image: break

        content = getattr(message, 'content', None) or message.get('content')
        if isinstance(content, str):
            if content.startswith("http"):
                image = download_image(content)
            elif "base64" in content or content.startswith("data:"):
                image = decode_base64_image(content)
            else:
                if len(content) < 1000: prompt_text = content
        elif isinstance(content, list):
            for item in content:
                itype = item.get("type") if isinstance(item, dict) else getattr(item, 'type', None)
                if itype == "text":
                    prompt_text = item.get("text") if isinstance(item, dict) else getattr(item, 'text', None)
                elif itype == "image_url":
                    iurl_obj = item.get("image_url") if isinstance(item, dict) else getattr(item, 'image_url', None)
                    url = iurl_obj.get("url") if isinstance(iurl_obj, dict) else getattr(iurl_obj, 'url', None)
                    if url:
                        if url.startswith("http"): image = download_image(url)
                        else: image = decode_base64_image(url)
                elif isinstance(item, dict) and "image_url" in item and isinstance(item["image_url"], str):
                    url = item["image_url"]
                    if url.startswith("http"): image = download_image(url)
                    else: image = decode_base64_image(url)
                if image: break
        if image: break
    return image, prompt_text

def download_image(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None

def decode_base64_image(base64_string: str) -> Image.Image:
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        logger.error(f"Base64 decode failed: {e}")
        return None

async def get_model(model_name: str) -> Optional[ModelWrapper]:
    if model_name not in models:
        model_name = "joytag" if "joytag" in models else None
    
    if not model_name: return None
    
    wrapper = models[model_name]
    await wrapper.ensure_loaded()
    return wrapper

# Models for API
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

class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: Optional[str] = ""
    images: Optional[List[str]] = None
    stream: bool = False

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    model_wrapper = await get_model(request.model)
    if not model_wrapper:
        raise HTTPException(status_code=404, detail="Model not found")

    image, prompt = get_image_from_messages(request.messages)
    if not image:
        return JSONResponse(status_code=400, content={"error": "No image found in request"})

    prediction = model_wrapper.predict(image, prompt=prompt)
    tag_string = ', '.join([t[0] for t in prediction]) if isinstance(prediction, list) else str(prediction)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": tag_string}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": name, "object": "model", "created": int(time.time()), "owned_by": "user"} for name in models.keys()]}

@app.get("/api/tags")
async def ollama_tags():
    return {
        "models": [{
            "name": name, "model": name, "modified_at": datetime.utcnow().isoformat() + "Z", "size": 0, "digest": "sha256:...",
            "details": {"parent_model": "", "format": "gguf", "family": "vision", "families": ["vision"], "parameter_size": "7B", "quantization_level": "Q4_0"}
        } for name in models.keys()]
    }

@app.post("/api/generate")
async def ollama_generate(request: OllamaGenerateRequest):
    model_wrapper = await get_model(request.model)
    if not model_wrapper: raise HTTPException(status_code=404, detail="Model not found")

    image = decode_base64_image(request.images[0]) if request.images else None
    if not image: raise HTTPException(status_code=400, detail="No image provided")

    prediction = model_wrapper.predict(image, prompt=request.prompt)
    tag_string = ', '.join([t[0] for t in prediction]) if isinstance(prediction, list) else str(prediction)
    
    return {"model": request.model, "created_at": datetime.utcnow().isoformat() + "Z", "response": tag_string, "done": True}

@app.post("/api/chat")
async def ollama_chat(request: OllamaChatRequest):
    model_wrapper = await get_model(request.model)
    if not model_wrapper: raise HTTPException(status_code=404, detail="Model not found")

    image, _ = get_image_from_messages(request.messages)
    if not image: return {"model": request.model, "message": {"role": "assistant", "content": "No image found."}, "done": True}

    prediction = model_wrapper.predict(image)
    tag_string = ', '.join([t[0] for t in prediction]) if isinstance(prediction, list) else str(prediction)

    return {"model": request.model, "created_at": datetime.utcnow().isoformat() + "Z", "message": {"role": "assistant", "content": tag_string}, "done": True}

@app.get("/api/config")
async def get_all_config():
    return {name: m.get_config() for name, m in models.items()}

class ConfigUpdate(BaseModel):
    threshold: Optional[float] = None
    character_threshold: Optional[float] = None

@app.post("/api/config/{model_name}")
async def update_config(model_name: str, config: ConfigUpdate):
    if model_name not in models: raise HTTPException(status_code=404, detail="Model not found")
    
    update_data = config.model_dump(exclude_unset=True)
    models[model_name].update_config(update_data)
    
    current_saved_config = load_config_file()
    if model_name not in current_saved_config: current_saved_config[model_name] = {}
    current_saved_config[model_name].update(update_data)
    save_config_file(current_saved_config)

    return {"status": "updated", "config": models[model_name].get_config()}

class InterrogateRequest(BaseModel):
    model: str
    image: str
    threshold: Optional[float] = None
    character_threshold: Optional[float] = None

@app.post("/api/interrogate")
async def interrogate(request: InterrogateRequest):
    model_wrapper = await get_model(request.model)
    if not model_wrapper: raise HTTPException(status_code=404, detail="Model not found")
    
    image = decode_base64_image(request.image)
    if not image: raise HTTPException(status_code=400, detail="Invalid image")
    
    predictions = model_wrapper.predict(image, threshold=request.threshold, character_threshold=request.character_threshold)
    if isinstance(predictions, str): raise HTTPException(status_code=500, detail=predictions)
    
    return {"model": request.model, "tags": [{"name": t[0], "score": t[1]} for t in predictions]}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--threshold", type=float, default=0.2, help="Tagging threshold")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    else:
        device = args.device
    
    threshold = args.threshold
    uvicorn.run(app, host=args.host, port=args.port)
