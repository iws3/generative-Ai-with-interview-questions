# Import necessary libraries
from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
from PIL import Image
import io

# Create a FastAPI application instance
app = FastAPI()

# Load the image captioning pipeline from Hugging Face with the vit-gpt2-image-captioning model
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Define the root endpoint of the API
@app.get("/")
def read_root():
    """
    A simple endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the Image Captioning API!"}

# Define an endpoint for image captioning
@app.post("/caption-image")
async def caption_image(file: UploadFile = File(...)):
    """
    Generates a caption for an uploaded image using the vit-gpt2-image-captioning model.
    """
    # Read the contents of the uploaded image file
    contents = await file.read()
    # Open the image using PIL (Python Imaging Library)
    image = Image.open(io.BytesIO(contents))

    # Generate a caption for the image
    result = captioner(image)
    # Return the generated caption
    return {"caption": result[0]['generated_text']}