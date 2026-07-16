import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_llm():
    """
    Initializes and returns the Groq language model for generation.
    Includes basic retry/backoff handling via max_retries parameter.
    """
    # ChatGroq automatically looks for GROQ_API_KEY in the environment
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_retries=5  # Handles rate-limit 429 errors with backoff
    )
    return llm
