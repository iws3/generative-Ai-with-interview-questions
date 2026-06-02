from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

#Loading the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#Define input:  Now we need a longer text field
class SummaryRequest(BaseModel):
    text: str

@app.post('/summarize')
def summarize_text(request:SummaryRequest):
    #Calling the model. We must handle long text by truncating within the model's limits
    result = summarizer(
        request.text,
        max_length=130, #summary length
        min_length=30,
        do_sample=False #for summary to be deterministic
    )
    return{
        "summary": result[0]['summary_text']
    }
#Model is about 1.6GB