from ingest import ingest_command
from query import query_command
import typer                                             # a library for building CLI (Command Line Interface) apps easily
from rich import print as rprint                         # an enhanced print function for styled/colored terminal output
import sys

app = typer.Typer(help="Multi-agent RAG - CLI & UI")     # "RAG Agent CLI" is the help description shown when running "python main.py --help"

# register commands
@app.command(name="ingest")                              # register externally defined function "def ingest_command()" -> python main.py ingest ./files
def ingest(path: str):
    ingest_command(path)
                
@app.command(name="query")                               # python main.py query "What is AI?"
def query(
    query_text: str, 
    ui: bool = typer.Option(False, "--ui", help="Enable clean output for Web UI")
):
    query_command(query_text, ui=ui)                


@app.callback()                                          # define a root-level callback
def main(
    verbose: bool = typer.Option(True, "--verbose", "-v", help="Enable verbose mode"),
    ui: bool = typer.Option(False, "--ui", help="Global UI flag")
):
    """
        For global initialization, initing logging, loading settings, warming up models ...
        @app.callback(): run before any specific command (query or ingest)
    """
    """
        If --ui is passed, we disable Rich styling and verbose logs 
        to ensure the Node.js bridge gets clean text.
    """
    if ui:
        verbose = False
    if verbose:
        rprint("[yellow]Starting RAG system...[/yellow]")  


if __name__ == "__main__":
    app()                                                # call app() to start CLI

"""  
    ✅ Complete Multi-Agent RAG Core Architecture:

        User Query
            ↓
        Router (entry control)
            ↓
        Planner (query decomposition)
            ↓
        Retriever (multiple calls)
            ↓
        Reasoner (grounded QA)
            ↓
        Critic (self-correction loop)
"""

