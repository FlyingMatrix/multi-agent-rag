import re
import json
import ollama
from typing import Dict, List

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
        query_clean = query.strip()
        
        # Keywords that should be REMOVED from the start of the query
        context_words = [r"^difference between\s+", r"^compare\s+", r"^between\s+"]
        for word in context_words:
            query_clean = re.sub(word, "", query_clean, flags=re.IGNORECASE)

        # Keywords that actually SEPARATE the topics
        split_signals = [" and ", " vs ", " , ", " or "]
        
        # Detect and Split
        pattern = "|".join(map(re.escape, split_signals))
        parts = [q.strip() for q in re.split(pattern, query_clean, flags=re.IGNORECASE) if q.strip()]

        if len(parts) > 1:
            return {"type": "multi", "sub_queries": parts}
            
        return {"type": "simple", "sub_queries": [query]}


"""
    TODO: 
        - upgrade Planner to LLM-powered decomposition (big performance jump)
        - adding "Context Injection":
            One common issue with decomposition is that sub-queries lose the original intent. 
            You can tweak your prompt to ensure the LLM adds context back into the sub-queries:
            - User: "Compare the battery life of the iPhone 15 and the S24."
            - Bad Decomposition: ["iPhone 15", "S24"] (The RAG searcher won't know what to look for).
            - Good Decomposition: ["Battery life specs of iPhone 15", "Battery life specs of Samsung Galaxy S24"].
"""
