from faiss_rag import rag_faiss_query
from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/rag-faiss-query") 

def query_faiss_rag(request: QueryRequest):
    return rag_faiss_query(request.question)