# Import necessary libraries
from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
import io
from PIL import Image

from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE as ROUTER_TEMPLATE

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from transformers import pipeline

# Create a FastAPI application instance
app = FastAPI()

# --- RAG Chain Setup ---
# Define documents for the RAG knowledge base
documents = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
]
# Initialize a text splitter to break down documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Create document objects
texts = text_splitter.create_documents(documents)
# Initialize Hugging Face embeddings for text vectorization
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Create a Chroma vector store from the documents and embeddings
vectorstore = Chroma.from_documents(texts, embeddings)
# Create a retriever to fetch relevant documents
retriever = vectorstore.as_retriever()
# Initialize a Hugging Face pipeline for the LLM used in RAG
llm_for_rag = HuggingFacePipeline(pipeline=pipeline('text-generation', model='distilgpt2', max_new_tokens=50))
# Create a RetrievalQA chain for RAG
rag_chain = RetrievalQA.from_chain_type(llm=llm_for_rag, chain_type="stuff", retriever=retriever)

# --- General Conversation Chain ---
# Initialize a Hugging Face pipeline for the LLM used in general conversation
llm_for_conv = HuggingFacePipeline(pipeline=pipeline('text-generation', model='distilgpt2', max_new_tokens=50))
# Create a general conversation chain
conversation_chain = ConversationChain(llm=llm_for_conv)

# --- Router Chain Setup ---
# Define information for different prompt routes
prompt_infos = [
    {
        "name": "eiffel_tower_expert",
        "description": "Good for answering questions about the Eiffel Tower",
        "chain": rag_chain
    },
    {
        "name": "general_conversation",
        "description": "Good for general conversation",
        "chain": conversation_chain
    }
]

# Initialize a Hugging Face pipeline for the router LLM
router_llm = HuggingFacePipeline(pipeline=pipeline('text2text-generation', model='t5-small', max_new_tokens=10))

# Format destinations for the router prompt
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = ROUTER_TEMPLATE.format(destinations=destinations_str)
# Create a PromptTemplate for the router
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
# Create the LLMRouterChain
router_chain = LLMRouterChain.from_llm(router_llm, router_prompt)

# Map destination names to their respective chains
destination_chains = {p['name']: p['chain'] for p in prompt_infos}
# Define a default chain if no specific route is matched
default_chain = conversation_chain

# Create the MultiPromptChain to combine the router and destination chains
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

# --- VQA (Image) Setup ---
# Initialize a Visual Question Answering pipeline
vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

# --- FastAPI Endpoint ---
@app.post("/chat")
async def chat(query: str = Form(...), file: Optional[UploadFile] = File(None)):
    """
    Intelligently routes user queries to different AI models based on input type (text or image).
    If an image is provided, it performs Visual Question Answering.
    Otherwise, it uses the MultiPromptChain to route text queries.
    """
    if file:
        # Process image query
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = vqa_pipeline(image=image, question=query)
        return {"answer": result[0]['answer'], "source": "image_qa"}
    else:
        # Process text query using the router chain
        result = chain.run(query)
        return {"answer": result}

@app.get("/")
def read_root():
    """
    A simple endpoint that returns a welcome message for the Intelligent Chat API.
    """
    return {"message": "Welcome to the Intelligent Chat API!"}