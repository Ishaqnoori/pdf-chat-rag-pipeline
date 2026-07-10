import os
import shutil
from langchain_community.vectorstores import FAISS
from utils.loaders import load_and_split_pdf
from utils.embeddings import get_embeddings_model

INDEX_DIR = "faiss_index"

def build_vector_store(pdf_path: str):
    """
    Builds a FAISS vector store from a PDF file and saves it to disk.
    
    Args:
        pdf_path (str): The path to the uploaded PDF file.
        
    Returns:
        FAISS: The initialized vector store.
    """
    # 1. Load and split the PDF
    chunks = load_and_split_pdf(pdf_path)
    
    # 2. Get the embeddings model
    embeddings = get_embeddings_model()
    
    # 3. Build the FAISS index
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 4. Save to disk (clear old index if exists)
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    
    vector_store.save_local(INDEX_DIR)
    
    return vector_store

def load_vector_store():
    """
    Loads an existing FAISS vector store from disk.
    
    Returns:
        FAISS: The loaded vector store, or None if it doesn't exist.
    """
    if not os.path.exists(INDEX_DIR):
        return None
        
    embeddings = get_embeddings_model()
    # Allow dangerous deserialization because we created the index locally
    vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vector_store

def get_retriever(k=4):
    """
    Gets the retriever interface from the loaded vector store.
    """
    vector_store = load_vector_store()
    if not vector_store:
        return None
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k})
