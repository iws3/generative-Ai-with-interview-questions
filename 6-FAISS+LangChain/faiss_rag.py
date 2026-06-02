from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline
from document_loader import load_documents_from_folder
from langchain.schema import Document

# Same embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Same LLM
llm_pipeline = hf_pipeline(
    "text-generation",
    model="distilgpt2", 
    max_new_tokens=50,
    temperature=0.1
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

def initialize_faiss_rag():
    """Initialize FAISS vector store"""
    documents = load_documents_from_folder("data")
    
    if not documents:
        documents = [Document(page_content="Add documents to /data folder")]
    
    # Key difference: FAISS instead of Chroma
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save FAISS index to disk (optional)
    vector_store.save_local("faiss_index")
    return vector_store.as_retriever()

# Initialize FAISS
faiss_retriever = initialize_faiss_rag()

def rag_faiss_query(question: str):
    """FAISS version of RAG query"""
    relevant_docs = faiss_retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in relevant_docs[:2]])
    
    # Simple prompt for the small model
    prompt = f"Based on this: {context}\nQuestion: {question}\nAnswer:"
    
    answer = llm.invoke(prompt)
    return {
        "question": question,
        "database": "FAISS",
        "contexts": [doc.page_content[:100] + "..." for doc in relevant_docs[:2]],
        "answer": answer
    }