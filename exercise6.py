# main.py (extend from Exercises 1–5)

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io

# LangChain + FAISS
from langchain.vectorstores import FAISS
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

# ---- RAG Setup (FAISS) ----
docs = [
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "LangChain integrates with FAISS for building scalable RAG pipelines.",
    "FAISS provides both exact and approximate nearest neighbor search methods."
]

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = splitter.create_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.from_documents(documents, embeddings)

qa_faiss = RetrievalQA.from_chain_type(llm=llm, retriever=faiss_index.as_retriever())

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
    answer = qa_faiss.run(request.query)
    return {"query": request.query, "answer": answer}

# **Interview Questions:**

# 1. What is FAISS, and why is it fast?
# answer: FAISS (Facebook AI Similarity Search) is a library developed by Facebook AI Research for efficient similarity search and clustering of dense vectors. It is designed to handle large-scale datasets and provides both exact and approximate nearest neighbor search methods. FAISS is fast because it uses optimized algorithms and data structures, such as inverted file systems (IVF) and hierarchical navigable small world graphs (HNSW), to reduce the search space and speed up the retrieval process. Additionally, FAISS leverages SIMD (Single Instruction, Multiple Data) instructions and multi-threading to further enhance performance.
# 2. What indexing methods does FAISS provide (IVF, HNSW)?
# answer: FAISS provides several indexing methods to optimize similarity search, including:
# - Inverted File System (IVF): This method partitions the dataset into clusters and creates an inverted index for efficient search within these clusters. It is particularly useful for large datasets.    
# 3. How does FAISS handle billions of vectors?
# answer: FAISS handles billions of vectors by using efficient indexing methods, such as IVF and HNSW, to reduce the search space and speed up retrieval. It also supports on-disk storage of indexes, allowing it to work with datasets that exceed available RAM. FAISS can be configured to use approximate nearest neighbor search, which trades off some accuracy for significant speed improvements, making it feasible to search through billions of vectors quickly. Additionally, FAISS is optimized for parallel processing and can leverage multiple CPU cores and SIMD instructions to further enhance performance.
# 4. Compare FAISS vs Chroma.
# answer: FAISS and Chroma are both libraries used for similarity search and vector storage, but they have different focuses and features:
# - FAISS: Developed by Facebook AI Research, FAISS is a highly optimized library for efficient similarity search and clustering of dense vectors. It provides various indexing methods, such as IVF and HNSW, and is designed to handle large-scale datasets with billions of vectors. FAISS is known for its speed and efficiency, making it suitable for applications requiring fast nearest neighbor searches.
# - Chroma: Chroma is an open-source vector database designed for storing and querying high-dimensional vectors. It focuses on ease of use, scalability, and integration with various machine learning frameworks. Chroma provides features like persistence, indexing, and support for multiple embedding models. While it may not be as optimized for speed as FAISS, Chroma offers a more user-friendly experience and is suitable for applications that require a simple and scalable vector storage solution.
# 5. What is approximate nearest neighbor (ANN) search?
# answer: Approximate Nearest Neighbor (ANN) search is a technique used to quickly find points in a high-dimensional space that are close to a given query point, without guaranteeing that the closest point is found. Instead of performing an exhaustive search, which can be computationally expensive and slow for large datasets, ANN algorithms use various heuristics and data structures to reduce the search space and speed up the retrieval process. This trade-off between accuracy and speed makes ANN search particularly useful for applications like recommendation systems, image retrieval, and natural language processing, where real-time performance is crucial.
# 6. How do you evaluate retrieval accuracy?
# answer: Retrieval accuracy can be evaluated using several metrics, depending on the specific application and requirements. Common metrics include:
# - Precision: The proportion of relevant documents retrieved out of the total documents retrieved. It measures the accuracy of the retrieval system in returning relevant results.
# - Recall: The proportion of relevant documents retrieved out of the total relevant documents available. It measures the system's ability to find all relevant results.
# - F1 Score: The harmonic mean of precision and recall, providing a single metric that balances both aspects of retrieval accuracy.
# - Mean Average Precision (MAP): The average precision across multiple queries, providing a comprehensive measure of retrieval performance.
# - Normalized Discounted Cumulative Gain (NDCG): A metric that takes into account the relevance of documents and their positions in the ranked list, rewarding systems that return highly relevant documents at the top of the list.
# 7. How would you deploy FAISS in production?
# answer: To deploy FAISS in production, you would typically follow these steps:
# - Set up a server environment: Choose a suitable server or cloud platform to host your FAISS instance, ensuring it has sufficient resources (CPU, RAM, storage) to handle your dataset and expected query load.
# - Install dependencies: Install FAISS and any other required libraries or frameworks, such as Python, NumPy, or Flask/FastAPI for building an API.
# - Prepare your dataset: Preprocess and embed your data into dense vectors using an appropriate embedding model. Store these vectors in a FAISS index using the desired indexing method (e.g., IVF, HNSW).
# - Build an API: Create a RESTful API using a web framework (e.g., Flask, FastAPI) to handle incoming search queries and return results from the FAISS index.
# - Optimize performance: Configure FAISS for optimal performance, such as enabling multi-threading, using approximate nearest neighbor search, and tuning index parameters based on your specific use case.# - Monitor and maintain: Implement monitoring to track the performance and health of your FAISS deployment, including query latency, error rates, and resource usage. Regularly update the index with new data and re-evaluate performance metrics to ensure continued accuracy and efficiency.
# - Scale as needed: Depending on the demand, consider scaling your deployment horizontally (adding more instances) or vertically (upgrading server resources) to handle increased query loads.
