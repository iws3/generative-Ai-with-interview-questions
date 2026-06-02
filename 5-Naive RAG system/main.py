from rag import rag_query
from pydantic import BaseModel
from fastapi import FastAPI

app= FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/rag-query")
def query_rag(request: QueryRequest):
    return rag_query(request.question)