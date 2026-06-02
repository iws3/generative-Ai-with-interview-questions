from research_chain import research_topic
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/researcher")
def research_endpoint(request: QueryRequest):
    return research_topic(request.question)  # FIXED: research_topic not research_endpoint