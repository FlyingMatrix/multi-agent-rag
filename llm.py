from ollama import chat

class OllamaLLM:
    def __init__(self, model: str="llama3"):
        self.model = model

    def stream_generate(self, prompt: str):
        """
            Streaming generation using Ollama Python client
        """
        stream = chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

