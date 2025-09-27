# Import necessary libraries
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

# Create a FastAPI application instance
app = FastAPI()

# Load the summarization pipeline from Hugging Face with the bart-large-cnn model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define a Pydantic model for the request body
class TextToSummarize(BaseModel):
    text: str

# Define the root endpoint of the API
@app.get("/")
def read_root():
    """
    A simple endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the Text Summarizer API!"}

# Define an endpoint for text summarization
@app.post("/summarize")
def summarize_text(data: TextToSummarize):
    """
    Summarizes a long text using the bart-large-cnn model.
    """
    # Use the summarizer to create a summary of the text
    summary = summarizer(data.text, max_length=130, min_length=30, do_sample=False)
    # Return the generated summary
    return {"summary": summary[0]['summary_text']}