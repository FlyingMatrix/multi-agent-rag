from agents.retriever import Retriever
from skill_registry import SkillRegistry
from llm import OllamaLLM

import tiktoken             # local tokenizer
import re
import json


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


def parse_json(text: str) -> dict:
    """
        Safely parse JSON text and fail gracefully if it's invalid.
    """
    try:
        return json.loads(text)     # json.loads() converts a JSON string into a Python dictionary
    except json.JSONDecodeError:
        return {
            "verdict": "incorrect",
            "issues": ["invalid json output"],
            "corrected_answer": "I don't know."
        }
    

def is_unknown(text: str) -> bool:
    text = text.lower()
    return (
        "don't know" in text or
        "do not know" in text or
        "not sure" in text or
        "not available" in text or
        "not provided" in text or
        "context does not" in text
    )
    

def stream_text(text: str):
    """
        Streaming UX
    """
    for token in text.split():
        yield token + " "


class Reasoner:
    def __init__(self, max_retries: int = 2):
        self.retriever = Retriever()
        self.skill_registry = SkillRegistry()
        self.llm = OllamaLLM(model="llama3")
        self.tokenizer = tiktoken.encoding_for_model(model_name="gpt-3.5-turbo")
        self.max_retries = max_retries

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

    def build_prompt(self, query: str, contexts, feedback: str = ""):
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
            return ("Answer: I don't know.\nSources: []", "")   # (prompt, context_text)
        
        # before loop, prioritize important chunks based on scores to ensure that most relevant info is included first
        is_numeric = is_numeric_query(query)
        contexts = sorted(
            contexts,
            key=lambda x: self.boost_score(x, is_numeric),
            reverse=True
        )

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

        # inject feedback for retries
        if feedback:
            prompt += f"""

            CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT:
            You MUST fix the following issues:

            {feedback}

            Do not repeat these mistakes.
            """

        return prompt, context_text
    
    def run_critic(self, context_text: str, query: str, answer: str):
        prompt = self.skill_registry.render(
            "rag_context_critic",
            context=context_text,
            query=query,
            answer=answer
        )
        return self.llm.generate(prompt)

    def run(self, query: str):
        """
            pipeline: rewrite_query -> retrieve -> QA + Critic loop
        """
        # rewrite query to improve retrieval quality 
        original_query = query
        rewritten_query = self.llm.rewrite_query(query)

        # get the top_k most similar nodes wrapped with scores, retrieval optimized
        contexts = self.retriever.retrieve(rewritten_query)

        feedback = ""

        # QA + Critic loop
        for attempt in range(self.max_retries + 1):
            # Step 1: QA generation
            prompt, context_text = self.build_prompt(original_query, contexts, feedback)
            answer = self.llm.generate(prompt)

            # Step 2: Critic evaluation
            if not is_unknown(answer):
                critic_raw = self.run_critic(context_text, original_query, answer)  # critic_raw -> JSON
                critic = parse_json(critic_raw)                                     # critic -> dictionary

                verdict = critic.get("verdict", "incorrect")
                issues = critic.get("issues", [])
                if not issues:
                    issues = ["Answer marked incorrect but no issues provided."]
                corrected_answer = critic.get("corrected_answer", "I don't know.")

                # logging
                print(f"[Attempt {attempt+1}], Verdict: {verdict}, Issues: {issues}")

                # if correct -> stream answer
                if verdict == "correct":
                    return stream_text(answer)

                # if last attempt -> stream corrected answer
                if attempt == self.max_retries:
                    return stream_text(corrected_answer)
                
                # prepare feedback for next iteration
                feedback = "\n".join(f"- {issue}" for issue in issues)
            else:
                if attempt == self.max_retries:
                    return stream_text("I don't know.")
                feedback = "- The answer was 'I don't know'. Try to find a supported answer if possible."
                continue
   

"""
    TODO: 
        - Context compression -> before building prompt:
            Summarize long chunks
            Keep only relevant sentences
"""

