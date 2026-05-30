# Import necessary libraries
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

# Create a FastAPI application instance
app = FastAPI()

# Load the text generation pipeline from Hugging Face with the distilgpt2 model
generator = pipeline('text-generation', model='distilgpt2')

# Define a Pydantic model for the request body
class Prompt(BaseModel):
    text: str

# Define the root endpoint of the API
@app.get("/")
def read_root():
    """
    A simple endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the Generative AI Exercises!"}

# Define an endpoint for text generation
@app.post("/hello-llm")
def generate_text(prompt: Prompt):
    """
    Generates text using the distilgpt2 model based on the provided prompt.
    """
    # Use the generator to create text, limiting the length to 50 tokens
    result = generator(prompt.text, max_length=50, num_return_sequences=1)
    # Return the generated text
    return {"generated_text": result[0]['generated_text']}

# RUN: curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"Hello, this is a test.\"}"http://localhost:8000/hello-llm