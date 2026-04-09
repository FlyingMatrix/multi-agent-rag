from agents.retriever import Retriever
from llm import OllamaLLM
import textwrap             # for removing any common leading whitespace from every line in text
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

        # build prompt with hallucination control
        prompt = textwrap.dedent(f"""\
        Persona:
        You are a precise and reliable assistant for question answering using retrieved context.

        Instructions:
        Answer the question using ONLY the provided context. Do not use prior knowledge.

        Context types:
        - text: unstructured paragraphs
        - table: structured data with rows and columns

        Guidelines for using context:
        - Base your answer strictly on the provided context.
        - Do not infer beyond what is explicitly stated.
        - Only include information that directly answers the question.
        - If the answer is uncertain, weakly supported, or the context is empty, say "I don't know."

        Tables:
        - Read tables carefully by matching rows and columns.
        - Extract exact values; do not approximate.
        - If multiple rows are relevant, include all necessary values.
        - Perform simple calculations only if all required values are explicitly present.

        Conflicts:
        - If multiple sources provide conflicting information, mention the conflict and cite all relevant sources.
        - Do not attempt to resolve conflicts unless the context clearly indicates which is correct.

        Citations:
        - Each context chunk is labeled with an index like [0], [1], etc.
        - Track which sources support the answer.

        Answer style:
        - Be concise, factual, and direct.
        - Do not include explanations, reasoning steps, or extra commentary.
        - Do not repeat the question.
        - If the question has multiple parts, answer all parts if possible.

        Output format:
        Answer: <final answer>
        Sources: [0][1]

        Context:
        {context_text}

        Question:
        {query}
        """)

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

