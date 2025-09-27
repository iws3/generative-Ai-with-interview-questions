from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io

# LangChain + Chroma
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

app = FastAPI(title="Generative AI Exercises")

# ---- Hugging Face pipelines ----
generator = pipeline("text-generation", model="distilgpt2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Wrap distilgpt2 for LangChain
llm = HuggingFacePipeline(pipeline=generator)

# ---- RAG Setup (Chroma) ----
# Sample documents (in practice you’d load PDFs, Markdown, etc.)
docs = [
    "FastAPI is a modern, fast web framework for building APIs with Python.",
    "LangChain is a framework for developing applications powered by language models.",
    "Chroma is an open-source embedding database for building retrieval-augmented generation (RAG) systems."
]

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = splitter.create_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Chroma vector store
vectorstore = Chroma.from_documents(documents, embeddings, collection_name="exercise5")

# Build RAG chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# ---- Schemas ----
from fastapi import UploadFile, File

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
    answer = qa.run(request.query)
    return {"query": request.query, "answer": answer}


# **Interview Questions:**

# 1. What is RAG and why is it useful?
# answer: RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant documents with generative models to produce more accurate and contextually relevant responses. It is useful because it allows the model to access up-to-date information and reduces the need for extensive fine-tuning on specific datasets.
# 2. How do embeddings represent meaning?
# answer: Embeddings are dense vector representations of text that capture semantic meaning by placing similar concepts closer together in the vector space. They are generated using models trained on large corpora, allowing them to understand relationships between words and phrases based on context.
# 3. Why use Chroma as a vector DB?
# answer: Chroma is an open-source, efficient, and scalable vector database that allows for fast similarity searches and easy integration with various embedding models. It is designed to handle large-scale datasets and provides features like persistence and indexing, making it suitable for RAG applications.
# 4. What is cosine similarity in retrieval?
# answer: Cosine similarity is a metric used to measure the similarity between two non-zero vectors by calculating the cosine of the angle between them. In retrieval, it helps identify how similar a query vector is to document vectors in the embedding space, allowing for effective ranking of relevant documents.    
# 5. How do you update a knowledge base?
# answer: A knowledge base can be updated by adding new documents, re-embedding the content, and updating the vector store. This may involve re-indexing or incrementally adding new embeddings to ensure the retrieval system reflects the most current information.
# 6. What is the risk of injecting irrelevant documents?
# answer: Injecting irrelevant documents can lead to poor retrieval results, as the model may retrieve and generate responses based on unrelated or incorrect information. This can degrade the quality of answers and reduce user trust in the system.
# 7. How does RAG differ from fine-tuning?
# answer: RAG leverages external knowledge through retrieval, allowing the model to access a broader range of information without needing to modify the model's parameters. Fine-tuning, on the other hand, involves adjusting the model's weights on a specific dataset, which can be time-consuming and may lead to overfitting on that dataset. RAG provides more flexibility and adaptability to new information.
