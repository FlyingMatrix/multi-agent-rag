from agents.retriever import Retriever
from llm import OllamaLLM

class Reasoner:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = OllamaLLM(model="llama3")

    def build_prompt(self, query: str, contexts):
        context_text = "\n\n".join(
            [f"[{i}] {node.text}" for i, node in enumerate(contexts)]
        )

        prompt = f"""
            You are a helpful assistant.
            Use the context below to answer the question.

            Context:
            {context_text}

            Question:
            {query}

            Answer:
        """
        return prompt

    def run(self, query: str):
        contexts = self.retriever.retrieve(query)

        prompt = self.build_prompt(query, contexts)

        return self.llm.stream_generate(prompt)
    
