from ingest import ingest_command
from query import query_command

import typer                                # a library for building CLI (Command Line Interface) apps easily
from rich import print                      # an enhanced print function for styled/colored terminal output


app = typer.Typer(help="RAG Agent CLI")     # "RAG Agent CLI" is the help description shown when running "python main.py --help"

# register commands
app.command(name="ingest")(ingest_command)  # register externally defined function "def ingest_command()" -> python main.py ingest
app.command(name="query")(query_command)    # python main.py query "What is AI?"


@app.callback()                             # define a root-level callback
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode")):
    """
        For global initialization, initing logging, loading settings, warming up models ...
        @app.callback(): run before any specific command (query or ingest)
    """
    if verbose:
        print("[yellow]Starting RAG system...[/yellow]")    


if __name__ == "__main__":
    app()                                   # call app() to start CLI

"""
    RAG System Architecture:

        User Query
            ↓
        Router
            ↓
        Reasoner
            ├── Planner
            │      ↓
            │   sub-queries
            ↓
        Retriever (multiple calls)
            ↓
        Reasoner (QA)
            ↓
        Critic (loop)
"""