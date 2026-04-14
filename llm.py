from ollama import chat
from settings import Settings


settings = Settings()
LLM_NAME = settings.llm_name


class LLM:
    def __init__(self, model: str=LLM_NAME):
        self.model = model

    def generate(self, prompt: str) -> str:
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

