#learn embeddings + RAG with chroma using all-MiniLM-L6-v2 hugging face model
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

app = FastAPI(title="RAG Query Endpoint")

# Use the all-MiniLM-L6-v2 model for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma vector store (in-memory for simplicity)
vector_store = Chroma(embedding_function=embedding_model)
# Use a text generation model for RAG (you can choose any suitable model)
rag_generator = pipeline("text-generation", model="distilgpt2")
class RAGRequest(BaseModel):
    query: str
    top_k: int = 3
    max_length: int = 100
class RAGResponse(BaseModel):
    query: str
    retrieved_docs: list[str]
    generated_texts: list[str]
    model: str
    generation_settings: dict
# Convenience GET endpoint for quick testing in browser
@app.get("/rag-query", response_model=RAGResponse)
async def rag_query_get(
    query: str = Query("What is the capital of France?", description="Query for RAG"),
    top_k: int = Query(3, ge=1, le=10),
    max_length: int = Query(100, ge=10, le=1024)
):
    # Retrieve relevant documents from the vector store
    docs = vector_store.similarity_search(query, k=top_k)
    retrieved_texts = [doc.page_content for doc in docs]
    # Combine retrieved documents into a single context
    context = " ".join(retrieved_texts)
    # Create a prompt for the generator
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    # Generate a response using the RAG generator
    results = rag_generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1
    )
    texts = [r["generated_text"] for r in results]
    return RAGResponse(
        query=query,
        retrieved_docs=retrieved_texts,
        generated_texts=texts,
        model="distilgpt2",
        generation_settings={
            "max_length": max_length,
            "top_k": top_k
        },
    )
