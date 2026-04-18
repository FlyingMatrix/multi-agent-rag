from ollama import chat
from settings import Settings


settings = Settings()
LLM_NAME = settings.llm_name


class LLM:
    def __init__(self, model: str=LLM_NAME):
        self.model = model

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        response = chat(
            model=self.model,
            messages=messages,
            options=kwargs  # Pass through temperature or other settings
        )
        return response["message"]["content"]

