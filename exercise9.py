# main.py (extend from Exercises 1–8)

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io

# LangChain
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

app = FastAPI(title="Generative AI Exercises")

# ---- Hugging Face pipelines ----
generator = pipeline("text-generation", model="distilgpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Wrap generator for LangChain
llm = HuggingFacePipeline(pipeline=generator)

# ---- Schemas ----
class ChatRequest(BaseModel):
    query: str
    file: UploadFile = None  # optional image

# ---- Simple intent detection ----
def detect_intent(query: str, file: UploadFile = None):
    if file:
        return "image"
    elif any(word in query.lower() for word in ["summarize", "summary"]):
        return "summarize"
    elif any(word in query.lower() for word in ["sentiment", "feeling", "opinion"]):
        return "sentiment"
    else:
        return "llm"

# ---- Routes ----
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    intent = detect_intent(request.query, getattr(request, "file", None))
    
    if intent == "llm":
        output = generator(request.query, max_length=100, num_return_sequences=1)
        answer = output[0]["generated_text"]
    elif intent == "summarize":
        output = summarizer(request.query, max_length=100, min_length=30, do_sample=False)
        answer = output[0]["summary_text"]
    elif intent == "sentiment":
        output = sentiment_analyzer(request.query)[0]
        answer = f"{output['label']} (score: {output['score']:.2f})"
    elif intent == "image" and request.file:
        contents = await request.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        caption = captioner(image)
        answer = caption[0]["generated_text"]
    else:
        answer = "Could not detect intent or missing input."

    return {"query": request.query, "intent": intent, "answer": answer}
