# main.py (extend from Exercises 1–7)

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.utilities import WikipediaAPIWrapper

app = FastAPI(title="Generative AI Exercises")

# ---- Hugging Face pipelines ----
generator = pipeline("text-generation", model="distilgpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Wrap generator for LangChain
llm = HuggingFacePipeline(pipeline=generator)

# Wikipedia wrapper
wiki = WikipediaAPIWrapper()

# ---- Schemas ----
class ResearcherRequest(BaseModel):
    topic: str
    summary_max_tokens: int = 100

# ---- Routes ----
@app.get("/")
def root():
    return {"message": "FastAPI + Hugging Face + LangChain is live 🚀"}

@app.post("/researcher")
def researcher(request: ResearcherRequest):
    # 1. Fetch Wikipedia content
    wiki_content = wiki.run(request.topic)
    
    # 2. Summarize content
    summary = summarizer(
        wiki_content,
        max_length=request.summary_max_tokens,
        min_length=30,
        do_sample=False
    )[0]["summary_text"]
    
    # 3. Sentiment analysis
    sentiment = sentiment_analyzer(summary)[0]
    
    return {
        "topic": request.topic,
        "summary": summary,
        "sentiment": sentiment
    }
