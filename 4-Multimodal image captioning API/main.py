from fastapi import FastAPI, File
from PIL import Image
from transformers import pipeline

app = FastAPI()
#Loading the model
captioner = pipeline("image-to-text" , model="nlpconnect/vit-gpt2-image-captioning")

app.post("/caption-image")
async def caption_image(file: UploadFile = File(...)):
    #reading the uploaded images
    image= Image.open(file.file)

    #Generating caption
    result=captioner(image)

    return{
        "caption": result[0]['generated_text']
    }
#Model is about 500MBs
