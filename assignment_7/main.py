# Import necessary libraries
from fastapi import FastAPI, File, UploadFile, Form
from transformers import pipeline
from PIL import Image
import io

# Create a FastAPI application instance
app = FastAPI()

# Load the Visual Question Answering pipeline from Hugging Face with the Salesforce/blip-vqa-base model
vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

# Define the root endpoint of the API
@app.get("/")
def read_root():
    """
    A simple endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the Visual Question Answering API!"}

# Define an endpoint for Visual Question Answering
@app.post("/qa-image-text")
async def qa_image_text(file: UploadFile = File(...), question: str = Form(...)):
    """
    Answers a question about an uploaded image using the Salesforce/blip-vqa-base model.
    """
    # Read the contents of the uploaded image file
    contents = await file.read()
    # Open the image using PIL (Python Imaging Library)
    image = Image.open(io.BytesIO(contents))

    # Use the VQA pipeline to answer the question about the image
    result = vqa_pipeline(image=image, question=question)
    # Return the answer
    return {"answer": result[0]['answer']}