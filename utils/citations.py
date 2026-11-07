def _display_page(doc):
    """PyPDFLoader pages are 0-indexed; prefer the printed page label, else page + 1."""
    label = doc.metadata.get("page_label")
    if label not in (None, ""):
        return str(label)
    page = doc.metadata.get("page")
    return str(page + 1) if isinstance(page, int) else "Unknown"


def format_docs_with_citations(docs):
    """Joins retrieved chunks into one context string tagged with source and page."""
    return "\n\n".join(
        f"--- Document {i + 1} (Source: {d.metadata.get('source', 'Unknown')}, Page: {_display_page(d)}) ---\n"
        f"{d.page_content}"
        for i, d in enumerate(docs)
    )


def page_numbers(docs) -> list[str]:
    """Unique page labels of the retrieved chunks, in reading order."""
    pages = {_display_page(d) for d in docs} - {"Unknown"}
    return sorted(pages, key=lambda p: (0, int(p), "") if p.isdigit() else (1, 0, p))


def sources_label(docs) -> str:
    """'Pages 1, 2' for one PDF; 'invoice.pdf p. 1, 2; resume.pdf p. 3' when several PDFs are in the chat."""
    by_source: dict[str, list] = {}
    for d in docs:
        by_source.setdefault(d.metadata.get("source", "PDF"), []).append(d)
    if not by_source:
        return ""
    if len(by_source) == 1:
        pages = page_numbers(docs)
        return f"{'Page' if len(pages) == 1 else 'Pages'} {', '.join(pages)}" if pages else ""
    return "; ".join(f"{src} p. {', '.join(page_numbers(ds))}" for src, ds in by_source.items() if page_numbers(ds))
