from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Hello LLM Endpoint")

# Use the small distilgpt2 model
generator = pipeline("text-generation", model="distilgpt2")

class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 50
    


class GenerateResponse(BaseModel):
    prompt: str
    generated_texts: list[str]
    model: str
    generation_settings: dict

# Convenience GET endpoint for quick testing in browser
@app.get("/hello-llm", response_model=GenerateResponse)
async def hello_llm_get(
    prompt: str = Query("Tell me a joke.", description="Prompt for the LLM"),
    max_length: int = Query(50, ge=1, le=1024)
):
    # Reuse the pipeline to generate
    results = generator(
        prompt,
        max_length=max_length
    )
    texts = [r["generated_text"] for r in results]
    return GenerateResponse(
        prompt=prompt,
        generated_texts=texts,
        model="distilgpt2",
        generation_settings={
            "max_length": max_length
        },
    )
