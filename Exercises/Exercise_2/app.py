from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI(title="Summarizer API")

# Load Hugging Face summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Request body schema
class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30
    temperature: float = 1.0  
    top_k: int = 50            
    top_p: float = 0.9         


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    """
    Summarize a long text using facebook/bart-large-cnn
    with controllable generation parameters.
    """
    try:
        summary = summarizer(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=True,                  
            temperature=request.temperature, 
            top_k=request.top_k,             
            top_p=request.top_p,             
        )
        return {
            "input_text": request.text,
            "summary_text": summary[0]["summary_text"]
        }
    except Exception as e:
        return {"error": str(e)}
