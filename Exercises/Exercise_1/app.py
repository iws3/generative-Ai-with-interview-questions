from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


app = FastAPI(title="Hello LLM API")

# Load Hugging Face text generation pipeline
generator = pipeline("text-generation", model="distilgpt2")


class Prompt(BaseModel):
    text: str
    max_length: int = 50

@app.post("/hello-llm")
async def hello_llm(prompt: Prompt):
    """
    Generate text from a given prompt using distilgpt2.
    """
    output = generator(prompt.text, max_length=prompt.max_length, num_return_sequences=1)
    return {"prompt": prompt.text, "generated_text": output[0]["generated_text"]}
