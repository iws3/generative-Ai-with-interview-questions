from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline
from document_loader import load_documents_from_folder
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize LLM for answer generation
llm_pipeline = hf_pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=100,
    temperature=0.3
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

def initialize_rag():
    """Load documents and create vector store"""
    documents = load_documents_from_folder("data")
    
    if not documents:
        # Fallback to simple examples
        documents = [Document(page_content="No documents found in /data folder.")]
    
    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store.as_retriever()

# Initialize when app starts
retriever = initialize_rag()

def rag_query(question: str):
    # Use invoke() instead of get_relevant_documents()
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
    
    prompt = f"""Use ONLY the following context to answer the question. Do not use any other knowledge:

Context:
{context}

Question: {question}
If the answer is not in the context, say "I cannot find the answer in the provided documents

Answer:"""
    
    answer = llm.invoke(prompt)
    return {
        "question": question,
        "contexts": [doc.page_content for doc in relevant_docs[:2]],
        "answer": answer
    }