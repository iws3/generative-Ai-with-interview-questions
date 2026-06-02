from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
#Load the sentimnt analysis model

classifier = pipeline("sentiment-analysis", model ="distilbert-base-uncased-finetuned-sst-2-english")

class SentimentRequest(BaseModel):
    text: str

@app.post("/sentiment")
def analyze_sentiment(request:SentimentRequest ):
    # Modal reruns a list of results for our unique input
    result = classifier(request.text)[0]

    return{
        "sentiment": result['label'],
        "confidence": round(result['score'], 4) # round to 4 decimal places
    } 
#Model is 269MB