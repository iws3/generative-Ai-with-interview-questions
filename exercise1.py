from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# init FastAPI
app = FastAPI(title="Generative AI Exercises")

# load Hugging Face model (distilgpt2 is small enough to run on CPU)
generator = pipeline("text-generation", model="distilgpt2")

# request schema
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.get("/")
def root():
    return {"message": "FastAPI + Hugging Face is live 🚀"}

@app.post("/hello-llm")
def hello_llm(request: PromptRequest):
    """
    Generate text from a given prompt using distilgpt2.
    """
    output = generator(
        request.prompt,
        max_length=len(request.prompt.split()) + request.max_tokens,
        num_return_sequences=1
    )
    return {"prompt": request.prompt, "generated": output[0]["generated_text"]}
# answer interview questions
# **Interview Questions:**

# 1. What is a language model?
# answer: a language model is a statistical tool that predicts the next word in a sequence based on the words that came before it. It is trained on large datasets of text to learn patterns, grammar, and context, enabling it to generate coherent and contextually relevant text.
# 2. How does GPT-2 differ from GPT-3/4?
# answer: GPT-2 is smaller and less powerful than GPT-3/4, with fewer parameters and less training data. GPT-3/4 can generate more coherent and contextually relevant text, handle more complex tasks, and understand nuanced prompts better than GPT-2.
# 3. Why is `distilgpt2` considered lightweight?
# answer: `distilgpt2` is a distilled version of GPT-2, meaning it has been compressed to reduce its size and computational requirements while retaining much of the original model's performance. This makes it more efficient and faster to run, especially on hardware with limited resources.
# 4. What are tokens, and why do they matter in LLMs?
# answer: Tokens are the basic units of text that a language model processes, which can be words, subwords, or characters. They matter because LLMs have limits on the number of tokens they can handle in a single input or output, affecting the model's ability to understand and generate text effectively.
# 5. How do you handle prompt length limits?
# answer: To handle prompt length limits, you can truncate or summarize the input text to fit within the model's maximum token limit. Additionally, you can use techniques like sliding windows for longer texts or break down the input into smaller, manageable chunks.
# 6. Why expose models through an API instead of CLI?
# answer: Exposing models through an API allows for easier integration with various applications, enabling remote access and scalability. It also provides a more user-friendly interface for developers and users who may not be comfortable with command-line interfaces (CLI).
# 7. What’s the risk of directly exposing LLMs without moderation?
# answer: Directly exposing LLMs without moderation can lead to the generation of harmful, biased, or inappropriate content. LLMs may inadvertently produce offensive language, misinformation, or content that violates ethical guidelines, which can harm users and damage the reputation of the service provider.

