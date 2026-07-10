from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path: str, chunk_size: int = 800, chunk_overlap: int = 100):
    """
    Loads a PDF file and splits it into smaller chunks.
    
    Args:
        file_path (str): The path to the PDF file.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The overlap between chunks to maintain context.
        
    Returns:
        list: A list of Document objects representing the chunks.
    """
    # Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    return chunks
