# Import necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create a FastAPI application instance
app = FastAPI()

# 1. Load and process the documents
# Define a list of documents that will form the knowledge base
documents = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
    "Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world.",
    "The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and is the tallest structure in Paris.",
    "Its base is square, measuring 125 metres (410 ft) on each side."
]
# Initialize a text splitter to break down long documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Create document objects from the raw text
texts = text_splitter.create_documents(documents)

# 2. Create embeddings
# Initialize Hugging Face embeddings to convert text into numerical vectors
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create a FAISS vector store and retriever
# Create a FAISS vector store from the documents and their embeddings
vectorstore = FAISS.from_documents(texts, embeddings)
# Create a retriever to search for relevant documents in the vector store
retriever = vectorstore.as_retriever()

# Define a Pydantic model for the query request body
class Query(BaseModel):
    question: str

# Define an endpoint for querying the RAG system
@app.post("/rag-faiss-query")
def rag_faiss_query(query: Query):
    """
    Queries the RAG system with a given question and returns the most relevant documents, using FAISS.
    """
    # Retrieve relevant documents based on the user's question
    docs = retriever.get_relevant_documents(query.question)
    # Return the content of the relevant documents
    return {"relevant_documents": [doc.page_content for doc in docs]}

# Define the root endpoint of the API
@app.get("/")
def read_root():
    """
    A simple endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the RAG with FAISS and LangChain API!"}