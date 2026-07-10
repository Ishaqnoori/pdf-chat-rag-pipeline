import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

def get_embeddings_model():
    """
    Initializes and returns the Google Generative AI embeddings model.
    """
    # Requires GOOGLE_API_KEY in environment
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings
