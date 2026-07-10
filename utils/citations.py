def format_docs_with_citations(docs):
    """
    Formats a list of retrieved documents into a single string with citations.
    
    Args:
        docs (list): A list of LangChain Document objects.
        
    Returns:
        str: Formatted string containing the document content and source pages.
    """
    formatted_texts = []
    for i, doc in enumerate(docs):
        # Extract metadata (e.g., page number)
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        
        # Format the text with a citation
        text = f"--- Document {i+1} (Source: {source}, Page: {page}) ---\n{doc.page_content}"
        formatted_texts.append(text)
        
    return "\n\n".join(formatted_texts)

def extract_citations(docs):
    """
    Extracts citation metadata to be displayed in the UI.
    """
    citations = []
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        if page not in citations:
            citations.append(page)
    return citations
