import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image


load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash')

def analyze_image_with_question(image: Image.Image, question : str):
    """Use Gemini to answer questions about images"""
    try:
        response = model.generate_content([question, image])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"
    
