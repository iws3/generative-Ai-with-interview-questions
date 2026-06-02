from intelligent_chat import intelligent_chat
from fastapi import FastAPI, UploadFile,File

app = FastAPI()

@app.post('/chat')
async def chat_endpoint(
    query: str,
    file: UploadFile= File(None) # Optional imahe upload
):
    image_data= None
    if file:
        image_data = await file.read()
    
    return intelligent_chat(query,image_data)