#detect postive/negative sentinment using distilbert-base-uncased-finetuned-sst-2-english hugging face model
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import List

app = FastAPI(title="Sentiment Analysis Endpoint")

# Use the distilbert-base-uncased-finetuned-sst-2-english model for sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
class SentimentRequest(BaseModel):
    texts: List[str]
class SentimentResponse(BaseModel):
    texts: List[str]
    sentiments: List[str]
    scores: List[float]
    model: str
# Convenience GET endpoint for quick testing in browser
@app.get("/sentiment-analysis", response_model=SentimentResponse)
async def sentiment_analysis_get(
    texts: List[str] = [
        "I love programming.",
        "I hate bugs.",
        "The weather is nice today."
    ]
):
    # Reuse the pipeline to analyze sentiments
    results = sentiment_analyzer(texts)
    sentiments = [r["label"] for r in results]
    scores = [r["score"] for r in results]
    return SentimentResponse(
        texts=texts,
        sentiments=sentiments,
        scores=scores,
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
#run using: uvicorn sentiiment_analysis:app --reload