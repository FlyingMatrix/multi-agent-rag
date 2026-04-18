import json
from typing import Dict
from settings import Settings
from llm import LLM


settings = Settings()
LLM_NAME = settings.llm_name


class Planner:
    """
        LLM-powered Planner using local LLMs.
        Decomposes complex queries into manageable sub-queries for RAG.
    """
    def __init__(self, model: str = LLM_NAME):
        self.model = model
        self.llm = LLM(model=self.model)        # Initialize the LLM instance
        self.system_prompt = (
            "You are a query decomposition assistant. Your task is to break down "
            "a user's request into a list of search queries for a RAG system.\n\n"
            "Rules:\n"
            "1. If the query is simple, return type 'simple'.\n"
            "2. If it requires multiple facts or comparison, return type 'multi'.\n"
            "3. Each sub_query must be a standalone sentence. You MUST inject "
            "the original context into each query (e.g., instead of 'its battery', "
            "use 'iPhone 15 battery life').\n"
            "4. Return ONLY a JSON object. Do not include any conversational text."
        )

    def plan(self, query: str) -> Dict:
        # Construct the instruction for the LLM
        self.prompt = (
            f"User Query: {query}\n\n"
            "Respond in this JSON format:\n"
            '{"type": "simple" | "multi", "sub_queries": ["query1", "query2"]}'
        )

        try:
            # Use the generate method from your LLM class
            # We pass temperature=0 through kwargs to keep it deterministic for consistency and precision in planning tasks
            content = self.llm.generate(
                prompt=self.prompt, 
                system_prompt=self.system_prompt,
                temperature=0 
            ).strip()

            # Some local models wrap JSON in code blocks (```json ... ```)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            plan_data = json.loads(content)

            # Basic validation to ensure the keys exist
            if "type" not in plan_data or "sub_queries" not in plan_data:
                raise ValueError("Missing keys in LLM response")

            return plan_data    # return a Dict
        
        except Exception as e:
            print(f"Error in Planner: {e}")
            # Fallback logic if the LLM output is malformed
            return {
                "type": "simple",
                "sub_queries": [query]
            }


"""
    TODO: 
        - upgrade Planner to LLM-powered decomposition (big performance jump)
"""
