# main.py (extend from Exercises 1 & 2)

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Generative AI Exercises")

# ---- Load models ----
generator = pipeline("text-generation", model="distilgpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

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

# **Interview Questions:**

# 1. What is transfer learning in NLP?
# answer: Transfer learning in NLP involves taking a pre-trained model (trained on a large corpus of text) and fine-tuning it on a specific task or dataset. This allows the model to leverage learned language representations, reducing the need for large amounts of task-specific data and improving performance.
# 2. Why use DistilBERT instead of BERT?
# answer: DistilBERT is a smaller, faster, and more efficient version of BERT that retains about 97% of BERT's performance while being 60% faster and having 40% fewer parameters. This makes it more suitable for deployment in resource-constrained environments or applications requiring lower latency.
# 3. What dataset is SST-2?
# answer: SST-2 (Stanford Sentiment Treebank) is a dataset used for sentiment analysis that contains movie reviews labeled as positive or negative. It is widely used for training and evaluating sentiment classification models.
# 4. What are embeddings in classification?
# answer: Embeddings are dense vector representations of words or phrases that capture semantic meaning and relationships. In classification tasks, embeddings serve as input features for machine learning models, allowing them to understand the context and nuances of the text data.
# 5. How do you evaluate classification performance?
# answer: Classification performance can be evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrix. The choice of metric depends on the specific task and the importance of false positives vs. false negatives.
# 6. What biases can exist in sentiment models?
# answer: Sentiment models can exhibit biases based on the training data, such as
# 7. How would you handle sarcasm in sentiment detection?
# answer: Sarcasm can be challenging for sentiment detection as it often involves saying the opposite of what is meant. To handle sarcasm, one could use more sophisticated models that consider context, tone, and user behavior. Additionally, incorporating datasets specifically labeled for sarcasm can help improve model performance in detecting sarcastic remarks.