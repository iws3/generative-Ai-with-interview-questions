# Import necessary libraries
import wikipediaapi
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import SequentialChain, TransformChain
from transformers import pipeline

# Create a FastAPI application instance
app = FastAPI()

# Initialize Hugging Face pipelines for summarization and sentiment analysis
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize Wikipedia API wrapper for English language
wiki = wikipediaapi.Wikipedia('en')

# 1. Transform chain to fetch Wikipedia content
def get_wiki_content(inputs: dict) -> dict:
    """
    Fetches content from Wikipedia for a given topic.
    Limits the content length to 1024 characters for summarization.
    """
    topic = inputs["topic"]
    page = wiki.page(topic)
    if not page.exists():
        return {"content": f"Could not find a Wikipedia page for '{topic}'."}
    # Limit the content length for the summarizer
    return {"content": page.text[:1024]}

# Create a TransformChain for fetching Wikipedia content
wiki_chain = TransformChain(
    input_variables=["topic"],
    output_variables=["content"],
    transform=get_wiki_content
)

# 2. Summarization chain (using the pipeline directly in a transform)
def summarize_content(inputs: dict) -> dict:
    """
    Summarizes the provided text content using a pre-trained summarization pipeline.
    """
    content = inputs["content"]
    summary = summarizer_pipeline(content, max_length=100, min_length=30, do_sample=False)
    return {"summary": summary[0]['summary_text']}

# Create a TransformChain for summarizing content
symmary_chain = TransformChain(
    input_variables=["content"],
    output_variables=["summary"],
    transform=summarize_content
)

# 3. Sentiment analysis chain (using the pipeline directly in a transform)
def analyze_sentiment(inputs: dict) -> dict:
    """
    Analyzes the sentiment of the provided summary using a pre-trained sentiment analysis pipeline.
    """
    summary = inputs["summary"]
    sentiment = sentiment_analyzer_pipeline(summary)
    return {"sentiment_result": sentiment[0]}

# Create a TransformChain for sentiment analysis
sentiment_chain = TransformChain(
    input_variables=["summary"],
    output_variables=["sentiment_result"],
    transform=analyze_sentiment
)

# The overall sequential chain combines the Wikipedia fetch, summarization, and sentiment analysis chains
researcher_chain = SequentialChain(
    chains=[wiki_chain, symmary_chain, sentiment_chain],
    input_variables=["topic"],
    # The output of the last chain will be returned.
    output_variables=["summary", "sentiment_result"],
    verbose=True
)

# Define a Pydantic model for the topic request body
class Topic(BaseModel):
    topic: str

# Define an endpoint for the researcher functionality
@app.post("/researcher")
def research(topic: Topic):
    """
    Fetches Wikipedia content for a given topic, summarizes it, and analyzes its sentiment.
    """
    # Invoke the researcher chain with the provided topic
    result = researcher_chain.invoke({"topic": topic.topic})
    return result

# Define the root endpoint of the API
@app.get("/")
def read_root():
    """
    A simple endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the Researcher API!"}