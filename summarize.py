#summarize long text sequences-to-sequence using hugging face pipeline facebook/bart-large-cnn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Summarization Endpoint")

# Use the facebook/bart-large-cnn model for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30
    do_sample: bool = False
    num_return_sequences: int = 1

class SummarizeResponse(BaseModel):
    original_text: str
    summarized_texts: list[str]
    model: str

# Convenience GET endpoint for quick testing in browser
@app.get("/summarize", response_model=SummarizeResponse)
async def summarize_get(
    text: str = Query(
        "The quick brown fox jumps over the lazy dog. " * 10,
        description="Text to be summarized",
    ),
    max_length: int = Query(150, ge=10, le=1024)

):
    # Reuse the pipeline to summarize
    results = summarizer(
        text,
        max_length=max_length
    )
    texts = [r["summary_text"] for r in results]
    return SummarizeResponse(
        original_text=text,
        summarized_texts=texts,
        model="facebook/bart-large-cnn"
    )
