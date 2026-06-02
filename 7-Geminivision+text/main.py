from multimodal_qa import analyze_image_with_question
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import multimodal_qa

app = FastAPI()

@app.post("/qa-image-text")
async def qa_image_text(
    question: str,
    file: UploadFile = File(...)
):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    answer = analyze_image_with_question(image, question)
    return{
        "question": question,
        "answer": answer,
        "model": "Gemini Vision"
    }