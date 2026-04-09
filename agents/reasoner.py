from agents.retriever import Retriever
from skill_registry import SkillRegistry
from llm import OllamaLLM

import tiktoken             # local tokenizer
import re


MAX_TOTAL_TOKENS = 8000     # safer than pushing limits
RESERVED_TOKENS = 1000      # instruction + question + formatting + answer buffer
MAX_CONTEXT_TOKENS = MAX_TOTAL_TOKENS - RESERVED_TOKENS
"""
    [ Instruction + Context + Question + Answer + Chat Formatting ] <= MAX_TOTAL_TOKENS
    
    Standard LLaMA 3 (local) supports ~8K tokens, so:
        ~1000 tokens -> instructions (system prompt) + question (user input) + chat formatting + answer (model output)
        ~6000-7000 tokens -> context
"""


def is_numeric_query(query: str) -> bool:
    query = query.lower()

    # keywords that suggest structured / numeric lookup
    keywords = [
        "how many", "how much", "number", "total", "sum",
        "average", "mean", "max", "min", "minimum", "maximum",
        "count", "percentage", "ratio", "compare", "difference",
        "highest", "lowest", "increase", "decrease"
    ]

    if any(k in query for k in keywords):
        return True

    # contains digits -> often numeric intent
    if re.search(r"\d", query):
        return True

    return False


class Reasoner:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = OllamaLLM(model="llama3")
        self.tokenizer = tiktoken.encoding_for_model(model_name="gpt-3.5-turbo")
        self.skill_registry = SkillRegistry()

    @staticmethod
    def fallback():
        yield "I don't know."

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def boost_score(self, nws, is_numeric):
        """
            boost table chunks if query suggests numeric lookup
        """
        score = nws.score
        if is_numeric and nws.node.metadata.get("type") == "table":
            score += 0.1  
        return score

    def build_prompt(self, query: str, contexts):
        """
            the contexts is the response from the retriever agent -> a list of NodeWithScore objects

            NodeWithScore(
                node=<Node>,
                score=<float>
            )
        """
        context_text = ""
        current_tokens = 0

        if not contexts:
            return self.fallback
        
        # before loop, prioritize important chunks based on scores to ensure that most relevant info is included first
        contexts = sorted(contexts, key=lambda x: self.boost_score(x, is_numeric_query(query), reverse=True))

        for i, nws in enumerate(contexts):
            chunk = f"[{i}]\n(type={nws.node.metadata.get('type', 'text')} | score={nws.score:.2f})\n{nws.node.get_content()}"

            chunk_tokens = self.count_tokens(chunk)

            if current_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
                # adaptive truncate the chunk instead of dropping a chunk entirely
                remaining_tokens = MAX_CONTEXT_TOKENS - current_tokens
                truncated_chunk = self.tokenizer.decode(
                    self.tokenizer.encode(chunk)[:remaining_tokens]
                )
                context_text += truncated_chunk
                break

            context_text += chunk + "\n\n"
            current_tokens += chunk_tokens

        # build prompt with self.skill_registry.render()
        prompt = self.skill_registry.render(
            "rag_context_qa",
            context=context_text,
            query=query
        )

        return prompt

    def run(self, query: str):
        """
            pipeline: rewrite_query -> retrieve -> build_prompt -> generate
        """
        # rewrite query to improve retrieval quality 
        original_query = query
        rewritten_query = self.llm.rewrite_query(query)

        # get the top_k most similar nodes wrapped with scores, retrieval optimized
        contexts = self.retriever.retrieve(rewritten_query)

        # use original_query and contexts to create prompt, answer grounded in original question
        prompt = self.build_prompt(original_query, contexts)

        # pass the prompt to llm to generate results
        results = self.llm.stream_generate(prompt)

        return results
    

"""
    TODO: 
        - Context compression -> before building prompt:
            Summarize long chunks
            Keep only relevant sentences
"""

