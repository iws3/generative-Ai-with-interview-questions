import os
import requests
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
#pretty print
from pprint import pprint
def environmental_variable():
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    # print("My Api keys loading...")
    # print(GOOGLE_API_KEY, OPENAI_API_KEY, GROQ_API_KEY)

def load_llm():
    from langchain_google_genai import GoogleGenerativeAI
    #loading keys
    environmental_variable()
    google_llm = GoogleGenerativeAI(
        #pass our configurations
        model = "gemini-2.5-flash",
        temperature = 0.9
    )
    return google_llm

def load_google_chat_model():
    from langchain_google_genai import ChatGoogleGenerativeAI
    environmental_variable()
    google_chat_model=ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature = 0.9
    )
    return google_chat_model