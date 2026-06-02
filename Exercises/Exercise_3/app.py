from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Create FastAPI app
app = FastAPI(title="Sentiment Analysis API")


sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


# Request body
class SentimentRequest(BaseModel):
    text: str


@app.post("/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze the sentiment of a given text (positive/negative).
    """
    result = sentiment_model(request.text)[0]  
    return {
        "text": request.text,
        "label": result["label"],  
        "score": round(result["score"], 4)  
    }
