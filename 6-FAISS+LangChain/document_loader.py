import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

def load_documents_from_folder(folder_path: str = "data"):
    """Load all PDF and TXT files from the data folder"""
    all_docs = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Created /data folder. Add some PDF or TXT files.")
        return all_docs

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif filename.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            else:
                continue
                
            all_docs.extend(documents)
            print(f"Loaded {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    if not all_docs:
        # Fallback examples
        all_docs = [
            Document(page_content="Paul Biya has been the president of Cameroon since 1982."),
            Document(page_content="FastAPI is a modern Python web framework for building APIs."),
            Document(page_content="The capital of France is Paris.")
        ]
        print("Using example documents (add files to /data folder)")
    
    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    return text_splitter.split_documents(all_docs)