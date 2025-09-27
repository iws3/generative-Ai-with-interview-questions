# main.py (extend from Exercises 1–3)

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io

app = FastAPI(title="Generative AI Exercises")

# ---- Load models ----
generator = pipeline("text-generation", model="distilgpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

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

# ---- Routes ----
@app.get("/")
def root():
    return {"message": "FastAPI + Hugging Face is live 🚀"}

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


# **Interview Questions:**

# 1. How does ViT process images?
# answeR: ViT splits images into patches, processes them as sequences, and uses self-attention to capture relationships.
# 2. What role does GPT-2 play in captioning?
# answer: GPT-2 generates coherent text based on the visual features extracted by the vision encoder.
# 3. Why combine a vision encode=r with a language decoder?
# answer: Combining them allows the model to understand visual content and generate relevant textual descriptions.
# 4. What datasets are used for captioning?
# answer: Common datasets include MS COCO, Flickr8k, and Flickr30k.
# 5. What challenges exist in image captioning?
# answer: Challenges include understanding context, handling diverse objects, and generating natural language.
# 6. How do you evaluate captions (BLEU, CIDEr)?
# answer: BLEU measures n-gram overlap, while CIDEr evaluates consensus with multiple references.
# 7. What real-world apps use capti=oning?
# answer: Real-world applications include accessibility tools, content management, and social media platforms.
