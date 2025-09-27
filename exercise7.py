# main.py (extend from Exercises 1–6)

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io

# LangChain + BLIP-2
from langchain.chains import LLMChain
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import Blip2Processor, Blip2ForConditionalGeneration

app = FastAPI(title="Generative AI Exercises")

# ---- Hugging Face pipelines (reuse previous models) ----
generator = pipeline("text-generation", model="distilgpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# ---- BLIP-2 setup ----
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

# ---- Schemas ----
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

class SummarizeRequest(BaseModel):
    text: str
    max_tokens: int = 130
    min_tokens: int = 30

class SentimentRequest(BaseModel):
    text: str

class RAGRequest(BaseModel):
    query: str

class QAImageTextRequest(BaseModel):
    question: str

# ---- Routes ----
@app.get("/")
def root():
    return {"message": "FastAPI + Hugging Face + LangChain is live 🚀"}

@app.post("/hello-llm")
def hello_llm(request: PromptRequest):
    output = generator(
        request.prompt,
        max_length=len(request.prompt.split()) + request.max_tokens,
        num_return_sequences=1
    )
    return {"prompt": request.prompt, "generated": output[0]["generated_text"]}

@app.post("/summarize")
def summarize(request: SummarizeRequest):
    summary = summarizer(
        request.text,
        max_length=request.max_tokens,
        min_length=request.min_tokens,
        do_sample=False
    )
    return {"summary": summary[0]["summary_text"]}

@app.post("/sentiment")
def sentiment(request: SentimentRequest):
    result = sentiment_analyzer(request.text)
    return {"text": request.text, "sentiment": result[0]}

@app.post("/caption-image")
async def caption_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    caption = captioner(image)
    return {"filename": file.filename, "caption": caption[0]["generated_text"]}

@app.post("/rag-query")
def rag_query(request: RAGRequest):
    # Placeholder: connect to Chroma/FAISS from previous exercises
    return {"query": request.query, "answer": "RAG answer placeholder"}

@app.post("/qa-image-text")
async def qa_image_text(file: UploadFile = File(...), question: str = ""):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt")
    out = blip2_model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return {"filename": file.filename, "question": question, "answer": answer}
