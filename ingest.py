from rich import print

from rag.loader import load_documents
from rag.splitter import split_documents
from rag.index import build_index


def ingest_command(path: str):
    """
        Ingest documents into the vector database
    """
    # load documents
    print(f"[cyan]Loading documents from:[/cyan] {path}")
    documents = load_documents(path)
    print(f"[cyan]Loaded {len(documents)} documents[/cyan]")

    # split documents into chunks (nodes)
    print(f"[cyan]Splitting documents...[/cyan]")
    nodes = split_documents(documents)
    print(f"[cyan]Generated {len(nodes)} chunks[/cyan]")

    # build index
    print(f"[cyan]Building index...[/cyan]")
    index = build_index(nodes)

    print("[green]Ingestion complete![/green]")

    return index

