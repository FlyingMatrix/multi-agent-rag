import json
import ollama
from typing import Dict
from settings import Settings

settings = Settings()
LLM_NAME = settings.llm_name

class Planner:
    """
        LLM-powered Planner using local LLMs.
        Decomposes complex queries into manageable sub-queries for RAG.
    """
    def __init__(self, model: str = LLM_NAME):
        self.model = model
        # System prompt defines the persona and the strict output format
        self.system_prompt = (
            "You are a query decomposition assistant. Your task is to break down "
            "a user's request into a list of search queries for a RAG system.\n\n"
            "Rules:\n"
            "1. If the query is simple, return type 'simple'.\n"
            "2. If it requires multiple facts or comparison, return type 'multi'.\n"
            "3. Each sub_query must be a standalone sentence (include context).\n"
            "4. Return ONLY a JSON object. Do not include any conversational text."
        )

    def plan(self, query: str) -> Dict:
        # Construct the instruction for the LLM
        prompt = (
            f"User Query: {query}\n\n"
            "Respond in this JSON format:\n"
            '{"type": "simple" | "multi", "sub_queries": ["query1", "query2"]}'
        )

        try:
            # Call the local Ollama instance
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                options={'temperature': 0}  # Keep it deterministic for consistency and precision in planning tasks
            )

            # Extract content and clean potential markdown formatting
            content = response['message']['content'].strip()

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
        - adding "Context Injection":
            One common issue with decomposition is that sub-queries lose the original intent. 
            You can tweak your prompt to ensure the LLM adds context back into the sub-queries:
            - User: "Compare the battery life of the iPhone 15 and the S24."
            - Bad Decomposition: ["iPhone 15", "S24"] (The RAG searcher won't know what to look for).
            - Good Decomposition: ["Battery life specs of iPhone 15", "Battery life specs of Samsung Galaxy S24"].
"""
