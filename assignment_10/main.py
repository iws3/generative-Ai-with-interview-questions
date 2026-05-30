# Import necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
from starlette.responses import StreamingResponse

# Create a FastAPI application instance
app = FastAPI()

# Load the Stable Diffusion pipeline
# This will download the model, which is several gigabytes in size.
# It is highly recommended to run this on a machine with a GPU for performance.
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
# If a CUDA-enabled GPU is available, move the pipeline to the GPU for faster processing.
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

# Define a Pydantic model for the text prompt in the request body
class Prompt(BaseModel):
    text: str

# Define an endpoint for generating images
@app.post("/generate-image")
def generate_image(prompt: Prompt):
    """
    Generates an image from a text prompt using the Stable Diffusion model.
    Note: Image generation on a CPU will be very slow.
    """
    # Generate an image based on the provided text prompt.
    # The .images[0] extracts the first generated image.
    image = pipe(prompt.text).images[0]

    # Save the generated image to a byte buffer in PNG format.
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    # Reset the buffer's position to the beginning.
    buffer.seek(0)

    # Return the image as a streaming response with the appropriate media type.
    return StreamingResponse(buffer, media_type="image/png")

# Define the root endpoint of the API
@app.get("/")
def read_root():
    """
    A simple endpoint that returns a welcome message for the Image Generation API.
    """
    return {"message": "Welcome to the Image Generation API!"}