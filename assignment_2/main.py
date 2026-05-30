    """
    FastAPI application for text summarization using a Hugging Face BART model.

    This application provides a web API endpoint to summarize long texts.
    It leverages the 'transformers' library to utilize pre-trained models
    for sequence-to-sequence summarization.
    """

# Import necessary libraries for building the API and handling data.
from fastapi import FastAPI  # FastAPI framework for building web APIs.
from transformers import pipeline  # Hugging Face's pipeline for easy use of pre-trained models.
from pydantic import BaseModel  # Used for data validation and settings management.

# Create a FastAPI application instance.
# This is the main entry point for the web application.
app = FastAPI()

# Load the summarization pipeline from Hugging Face.
# The 'facebook/bart-large-cnn' model is a pre-trained sequence-to-sequence model
# optimized for summarization tasks. This step downloads and loads the model
# into memory, which can take some time and requires an internet connection.
# Changes to the model name here would load a different summarization model.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define a Pydantic model for the request body of the /summarize endpoint.
# This class specifies the expected structure and data types for incoming JSON payloads.
# If the incoming request body does not match this schema, FastAPI will automatically
# return a validation error.
class TextToSummarize(BaseModel):
    text: str  # The input text string that needs to be summarized.

# Define the root endpoint of the API.
# This is a GET request handler for the base URL ("/").
@app.get("/")
def read_root():
    """
    A simple endpoint that returns a welcome message for the API.
    This can be used to check if the API is running.
    """
    return {"message": "Welcome to the Generative AI Exercises!"}

# Define an endpoint for text summarization.
# This is a POST request handler for the "/summarize" URL.
# It expects a JSON payload conforming to the TextToSummarize Pydantic model.
@app.post("/summarize")
def summarize_text(data: TextToSummarize):
    """
    Summarizes a long text using the pre-trained 'facebook/bart-large-cnn' model.

    Args:
        data (TextToSummarize): The request body containing the text to be summarized.

    Returns:
        dict: A dictionary containing the generated summary under the key 'summary'.

    Potential Impact of Changes:
    - Changing 'max_length' or 'min_length' will alter the length of the generated summary.
    - Setting 'do_sample=True' would introduce more randomness into the summary generation.
    - If the 'summarizer' model is changed (e.g., to a different language model),
      the quality and style of the summaries will change accordingly.
    """
    # Use the loaded summarizer pipeline to generate a summary.
    # 'data.text' provides the input text from the request body.
    # 'max_length' and 'min_length' control the output summary length.
    # 'do_sample=False' ensures deterministic (non-random) summary generation.
    summary = summarizer(data.text, max_length=130, min_length=30, do_sample=False)
    # The pipeline returns a list of dictionaries; we extract the 'summary_text' from the first result.
    # Return the generated summary in a JSON-compatible dictionary.
    return {"summary": summary[0]['summary_text']}