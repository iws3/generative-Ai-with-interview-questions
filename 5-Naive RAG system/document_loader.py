import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

def load_documents_from_folder(folder_path: str = "data"):
    """Load all PDF and TXT files from the data folder"""
    all_docs = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
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
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    # Split into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=300,  # Smaller to avoid the warning
        chunk_overlap=30
    )
    return text_splitter.split_documents(all_docs)