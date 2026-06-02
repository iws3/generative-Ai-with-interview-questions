from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline  #Easiest way to use

#Creating the app
app = FastAPI()

#Loading the model once the app starts
generator = pipeline('text-generation', model='distilgpt2')

#Defining wjat my API expects, which is a prompt string

class Request(BaseModel):
    prompt: str

# Creating the endpoint
@app.post('/hello-llm')
def generate_text(request: Request):
    #Calling the model
    result = generator(request.prompt, max_length=100,
    num_return_sequences=1, 
    temperature=0.7, repetition_penalty=1.5, do_sample=True)
    print(result)
    # Return the generated text
    return{
        "generated_text": result[0]['generated_text']
    }

    

