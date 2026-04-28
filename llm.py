import ollama
from settings import Settings
from typing import Iterator, List, Dict


settings = Settings()
LLM_NAME = settings.reasoner_llm    # default llm


class LLM:
    def __init__(self, model: str=LLM_NAME):
        self.model = model

    def _build_messages(self, prompt: str, system_prompt: str = None) -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        print(f"[DEBUG] Calling Ollama API with model: {self.model}")
        messages = self._build_messages(prompt, system_prompt)
        response = ollama.chat(
            model=self.model, 
            messages=messages, 
            options=kwargs
        )
        return response.get("message", {}).get("content", "")

    def stream(self, prompt: str, system_prompt: str = None, **kwargs) -> Iterator[str]:
        print(f"[DEBUG] Calling Ollama API with model: {self.model}")
        messages = self._build_messages(prompt, system_prompt)
        try:    
            response_stream = ollama.chat(
                model=self.model,
                messages=messages,
                options=kwargs,
                stream=True         # returns an Iterable
            )

            for chunk in response_stream:
                # Safely get content; if the chunk is just metadata, it might be empty
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        
        except Exception as e:
            yield f"\n[Error]: {str(e)}"

    def rewrite_query(self, query: str) -> str:
        prompt = (
            "Rewrite the query to improve retrieval quality.\n"
            "Keep it concise and precise.\n\n"
            f"Query: {query}\n"
            "Rewritten:"
        )
        return self.generate(prompt).strip()

