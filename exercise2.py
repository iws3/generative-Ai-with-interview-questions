from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Generative AI Exercises")

# ---- Load models once on startup ----
generator = pipeline("text-generation", model="distilgpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ---- Schemas ----
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

class SummarizeRequest(BaseModel):
    text: str
    max_tokens: int = 130
    min_tokens: int = 40

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

# **Interview Questions:**

# 1. What is abstractive vs extractive summarization?
# answer: Abstractive summarization generates new phrases and sentences to capture the main ideas of the text, while extractive summarization selects and compiles key sentences or phrases directly from the original text.
# 2. Why is BART good for summarization?
# answer: BART is effective for summarization because it combines a bidirectional encoder (like BERT) with a left-to-right decoder (like GPT), allowing it to understand context and generate coherent summaries. Its pre-training on large text corpora helps it learn language patterns, making it adept at producing fluent and relevant summaries.
# 3. What are encoder-decoder architectures?
# answer: Encoder-decoder architectures consist of two main components: an encoder that processes the input data and encodes it into a fixed-size representation, and a decoder that takes this representation and generates the output sequence. This architecture is commonly used in tasks like machine translation and text summarization.
# 4. How does beam search affect summary quality?
# answer: Beam search improves summary quality by exploring multiple possible output sequences simultaneously, allowing the model to consider various options and select the most probable one. This leads to more coherent and contextually relevant summaries compared to greedy decoding, which only considers the most likely next word at each step.
# 5. What are hallucinations in summarization?
# answer: Hallucinations in summarization refer to instances where the model generates information that is not present in the original text, leading to inaccuracies or misleading content in the summary. This can occur when the model overgeneralizes or misinterprets the input data.
# 6. What evaluation metrics exist (ROUGE, BLEU)?
# answer: Evaluation metrics for summarization include ROUGE (Recall-Oriented Understudy for Gisting Evaluation), which measures the overlap of n-grams, word sequences, and word pairs between the generated summary and reference summaries. BLEU (Bilingual Evaluation Understudy) is another metric that evaluates the quality of text by comparing it to one or more reference texts, focusing on precision of n-grams.
# 7. How would you fine-tune BART on legal documents?
# answer: To fine-tune BART on legal documents, I would first gather a large dataset of legal texts and their corresponding summaries. Then, I would preprocess the data to ensure it is clean and formatted correctly. Next, I would use transfer learning to fine-tune the pre-trained BART model on this dataset, adjusting hyperparameters such as learning rate and batch size to optimize performance. Finally, I would evaluate the model using relevant metrics like ROUGE to ensure it generates accurate and coherent summaries of legal documents.
