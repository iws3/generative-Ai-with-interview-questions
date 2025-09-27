# Import necessary libraries
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

# Create a FastAPI application instance
app = FastAPI()

# Load the sentiment analysis pipeline from Hugging Face using an explicit model name
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define a Pydantic model for the request body
class TextToAnalyze(BaseModel):
    text: str

# Define the root endpoint of the API
@app.get("/")
def read_root():
    """
    A simple endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the Sentiment Analysis API!"}

# Define an endpoint for sentiment analysis
@app.post("/sentiment")
def analyze_sentiment(data: TextToAnalyze):
    """
    Analyzes the sentiment of a text using the distilbert-base-uncased-finetuned-sst-2-english model.
    """
    # Use the sentiment_analyzer to analyze the sentiment of the text
    result = sentiment_analyzer(data.text)
    # Return the sentiment analysis result
    return {"sentiment": result[0]}