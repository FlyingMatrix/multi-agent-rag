from agents.router import Router
from agents.reasoner import Reasoner
from rich.console import Console
import sys
import io

# Force stdout to use utf-8 and 'replace' characters it doesn't recognize
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

router = Router()
reasoner = Reasoner()
console = Console()     


def handle_query(query: str):
    route = router.route(query)

    if route == "rag":
        return reasoner.run(query)

    return ["I don't know."]


def query_command(query: str, ui: bool = False):
    if not ui:
        console.print(f"[green]Query: {query}[/green]")

    stream = handle_query(query)

    for token in stream:
        if ui:
            # Use sys.stdout.write for the cleanest possible stream
            sys.stdout.write(token)
            sys.stdout.flush()
        else:
            console.print(f"[cyan]{token}[/cyan]", end="")

    if ui:
        print()
    else:
        console.print()


"""
    Flow:    
        CLI -> query_command()
                -> handle_query()
                    -> Router
                    -> Reasoner
                        -> Planner (if added)
                        -> Retriever
                        -> Critic loop
                -> stream output
"""