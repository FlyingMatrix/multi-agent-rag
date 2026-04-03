import typer
from rich import print

def query_command(question: str):
    """
        Query the RAG system
    """
    print(f"[green]Query:[/green] {question}")

    # TODO: implement query pipeline