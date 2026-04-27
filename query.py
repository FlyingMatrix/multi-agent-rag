from agents.router import Router
from agents.reasoner import Reasoner
from rich.console import Console


router = Router()
reasoner = Reasoner()
console = Console()     


def handle_query(query: str):
    route = router.route(query)

    if route == "rag":
        return reasoner.run(query)

    return ["I don't know."]


def query_command(query: str):
    console.print(f"[green]Query: {query}[/green]")

    stream = handle_query(query)
    for token in stream:
        console.print(f"[cyan]{token}[/cyan]", end="")

    console.print()     # print a final newline


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