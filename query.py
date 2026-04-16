from agents.router import Router
from agents.reasoner import Reasoner
from rich import print


router = Router()
reasoner = Reasoner()


def handle_query(query: str):
    route = router.route(query)

    if route == "rag":
        return reasoner.run(query)

    return ["I don't know."]


def query_command(query: str):

    print(f"Query: {query}", style="green")

    stream = handle_query(query)
    for token in stream:
        print(f"[blue]{token}[/blue]", end="")

    print()

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