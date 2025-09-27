# main.py (extend from Exercises 1–9)

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io
import torch

# Diffusers
from diffusers import StableDiffusionPipeline

app = FastAPI(title="Generative AI Exercises")

# ---- Hugging Face pipelines (reuse previous models) ----
generator = pipeline("text-generation", model="distilgpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# ---- Stable Diffusion setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
stable_diffusion = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
stable_diffusion = stable_diffusion.to(device)

# ---- Schemas ----
class GenerateImageRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

# ---- Routes ----
@app.post("/generate-image")
def generate_image(request: GenerateImageRequest):
    image = stable_diffusion(
        prompt=request.prompt,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale
    ).images[0]
    
    # Convert image to bytes for API response
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    
    return {
        "prompt": request.prompt,
        "image_bytes": img_byte_arr.hex()  # send as hex string; client can decode
    }
