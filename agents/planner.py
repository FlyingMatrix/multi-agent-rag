class Planner:
    """
    Planner decides how to solve the query:
        -> simple query (1-step)
        -> complex query (multi-step)
    Output:
        {
            "type": "simple" | "multi",
            "sub_queries": [...]
        }
    """

    def plan(self, query: str) -> dict:
        query_lower = query.lower()

        # detect multi-part queries
        multi_signals = [" and ", " compare ", " vs ", " difference ", "between"]

        if any(s in query_lower for s in multi_signals):
            # split query
            parts = [q.strip() for q in query.split(" and ") if q.strip()]

            if len(parts) > 1:
                return {
                    "type": "multi",
                    "sub_queries": parts
                }

        return {
            "type": "simple",
            "sub_queries": [query]
        }


"""
    TODO: upgrade Planner to LLM-powered decomposition (big performance jump)
"""