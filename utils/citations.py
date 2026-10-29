def _display_page(doc):
    """
    PyPDFLoader stores 'page' as 0-indexed, so page 1 shows up as 0.
    Prefer 'page_label' (the printed page number) when available, else page + 1.
    """
    label = doc.metadata.get("page_label")
    if label not in (None, ""):
        return str(label)
    page = doc.metadata.get("page")
    if isinstance(page, int):
        return str(page + 1)
    return None


def format_docs_with_citations(docs):
    """
    Formats retrieved documents into one context string with source/page tags.
    """
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = _display_page(doc) or "Unknown"
        formatted.append(f"--- Document {i + 1} (Source: {source}, Page: {page}) ---\n{doc.page_content}")
    return "\n\n".join(formatted)


def extract_citations(docs):
    """
    Returns unique, human-readable page numbers in ascending order.
    """
    pages = {p for p in (_display_page(d) for d in docs) if p is not None}

    def sort_key(p):
        return (0, int(p), "") if p.isdigit() else (1, 0, p)

    return sorted(pages, key=sort_key)
