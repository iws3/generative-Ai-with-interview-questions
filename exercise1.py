from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# init FastAPI
app = FastAPI(title="Generative AI Exercises")

# load Hugging Face model (distilgpt2 is small enough to run on CPU)
generator = pipeline("text-generation", model="distilgpt2")

# request schema
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.get("/")
def root():
    return {"message": "FastAPI + Hugging Face is live 🚀"}

@app.post("/hello-llm")
def hello_llm(request: PromptRequest):
    """
    Generate text from a given prompt using distilgpt2.
    """
    output = generator(
        request.prompt,
        max_length=len(request.prompt.split()) + request.max_tokens,
        num_return_sequences=1
    )
    return {"prompt": request.prompt, "generated": output[0]["generated_text"]}
