#upload an image, return caption using nlpconnect/vit-gpt2-image-captioning hugging face model
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io
import numpy as np

app = FastAPI(title="Image Captioning Endpoint")

# Use the nlpconnect/vit-gpt2-image-captioning model for image captioning
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
class CaptionRequest(BaseModel):
    image_data: bytes
class CaptionResponse(BaseModel):
    caption: str
    model: str
# Convenience GET endpoint for quick testing in browser
@app.get("/caption-image")
async def caption_image_get():
    return {"message": "Use POST /caption-image with an image file to get a caption."}

@app.post("/caption-image", response_model=CaptionResponse)
async def caption_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Only JPEG and PNG are supported.")
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    # Reuse the pipeline to generate caption
    results = captioner(image)
    caption = results[0]["generated_text"]
    return CaptionResponse(
        caption=caption,
        model="nlpconnect/vit-gpt2-image-captioning"
    )
#run using: uvicorn caption-image:app --reload