class Router:
    """
    Router decides which pipeline should handle the query.
    For now:
        - "rag" -> go through retriever + reasoner
        - "fallback" -> return "I don't know"
    """

    def route(self, query: str) -> str:
        query = query.lower()

        # simple heuristics
        if not query.strip():
            return "fallback"

        # very basic rule: factual / info-seeking queries -> RAG
        keywords = [
            "what", "who", "when", "where", "why", "how",
            "list", "show", "find", "compare", "explain"
        ]

        if any(k in query for k in keywords):
            return "rag"

        return "rag"  # default to RAG for now
