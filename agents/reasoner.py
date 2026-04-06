from agents.retriever import Retriever
from llm import OllamaLLM

class Reasoner:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = OllamaLLM(model="llama3")

    def build_prompt(self, query: str, contexts):
        """
            the contexts is the response from the retriever agent -> a list of NodeWithScore objects

            NodeWithScore(
                node=<Node>,
                score=<float>
            )
        """
        context_text = "\n\n".join(
            [f"[{i}] {nws.node.get_content()}" for i, nws in enumerate(contexts)]
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
        # get the top_k most similar nodes wrapped with scores
        contexts = self.retriever.retrieve(query)
        # use query and contexts to create prompt
        prompt = self.build_prompt(query, contexts)
        # pass the prompt to llm to generate results
        return self.llm.stream_generate(prompt)
    
