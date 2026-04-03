import typer                                # a library for building CLI (Command Line Interface) apps easily
from rich import print                      # an enhanced print function for styled/colored terminal output

from ingest import ingest_command
from query import query_command

app = typer.Typer(help="RAG Agent CLI")     # "RAG Agent CLI" is the help description shown when running "python main.py --help"

# register commands
app.command(name="ingest")(ingest_command)  # register externally defined function "def ingest_command()" -> "python main.py ingest"
app.command(name="query")(query_command)    # "python main.py query"

@app.callback()                             # define a root-level callback
def main():
    """
        RAG (Retrieval Augmented Generation) Agent CLI
    """
    pass

if __name__ == "__main__":
    app()                                   # call app() to start CLI

