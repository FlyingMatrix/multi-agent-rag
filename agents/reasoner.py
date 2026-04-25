from agents.retriever import Retriever
from agents.planner import Planner
from skill_registry import SkillRegistry
from llm import LLM
from typing import Iterable
from dataclasses import dataclass
from settings import Settings
from rich import print

import tiktoken             # local tokenizer
import re
import json


settings = Settings()

ENCODING_NAME = settings.encoding_name

REASONER_LLM = settings.reasoner_llm
REASONER_MAX_CONTEXT_TOKENS = settings.reasoner_max_context_tokens

CRITIC_LLM = settings.critic_llm


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
        Advanced JSON extractor: handles markdown blocks and model chatter.
    """
    try:
        # 1. Try a clean parse first
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # 2. Try to find JSON inside code blocks or curly braces
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        # 3. Final Fallback
        return {
            "verdict": "incorrect",
            "issues": ["The model failed to provide a valid JSON structure."],
            "corrected_answer": "I don't know."
        }
    

def should_critic(text: str) -> bool:
    text = text.lower()
    weak_signals = [
        "don't know",
        "do not know",
        "not sure",
        "not available",
        "not provided",
        "context does not",
        "cannot find",
        "does not contain",
        "no information"
    ]
    return not any(w in text for w in weak_signals)
    

def stream_text(text: str):
    """
        Streaming UX
    """
    for token in re.findall(r"\S+|\n", text):
        yield token + " "


class Reasoner:
    def __init__(self, max_retries: int = 2):
        self.retriever = Retriever()
        self.planner = Planner()
        self.skill_registry = SkillRegistry()
        self.reasoner_llm = LLM(model=REASONER_LLM)
        self.critic_llm = LLM(model=CRITIC_LLM)
        self.tokenizer = tiktoken.get_encoding(ENCODING_NAME)   # generic subword tokenization scheme working reasonably well across models
        self.max_retries = max_retries

        print(f"[magenta]Initialize {REASONER_LLM} as the Reasoner LLM[/magenta]")
        print(f"[magenta]Initialize {CRITIC_LLM} as the Critic LLM[/magenta]")

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
        is_numeric = is_numeric_query(query)

        if not contexts:
            return ("Answer: I don't know.\nSources: []", "")   # (prompt, context_text)
        
        # before loop, prioritize important chunks based on scores to ensure that most relevant info is included first
        contexts = sorted(
            contexts,
            key=lambda x: self.boost_score(x, is_numeric),
            reverse=True
        )

        for i, nws in enumerate(contexts):
            if current_tokens >= REASONER_MAX_CONTEXT_TOKENS:
                break

            chunk = f"[{i}]\n(type={nws.node.metadata.get('type', 'text')} | score={nws.score:.2f})\n{nws.node.get_content()}"

            chunk_tokens = self.count_tokens(chunk)

            if current_tokens + chunk_tokens > REASONER_MAX_CONTEXT_TOKENS:
                # adaptive truncate the chunk instead of dropping a chunk entirely
                remaining_tokens = REASONER_MAX_CONTEXT_TOKENS - current_tokens
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

            CRITICAL INSTRUCTIONS (HIGHEST PRIORITY):
            You made mistakes in the previous attempt.

            You MUST fix ALL of the following issues:

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
        return self.critic_llm.generate(prompt)

    def run(self, query: str) -> Iterable[str]:
        """
            pipeline: 
                plan query -> rewrite (sub)query -> retrieve -> build prompt -> generate answers -> critic evaluation loop

            adaptive query retrieval strategy:
                - simple -> fast path:
                    1 query -> 1 retrieval -> contexts -> global sorting
                - complex -> multi-step retrieval:
                    N sub-queries -> multi-step retrieval -> merged -> deduplicated -> global sorting -> richer contexts
        """
        original_query = query
        # use planner to decide how to solve the query (if split into sub_query is necessary)
        plan = self.planner.plan(original_query)

        # ---- simple query (fast path) ----
        if plan["type"] == "simple":
            # rewrite query to improve retrieval quality 
            rewritten = self.reasoner_llm.rewrite_query(original_query)
            all_contexts = self.retriever.retrieve(rewritten)

        # ---- multi query ----
        else:
            all_contexts = []

            # for each sub_query, get the top_k most similar nodes wrapped with scores, retrieval optimized
            for sub_query in plan["sub_queries"]:
                # rewrite sub_query to improve retrieval quality 
                rewritten = self.reasoner_llm.rewrite_query(sub_query)
                contexts = self.retriever.retrieve(rewritten)
                all_contexts.extend(contexts)

            # deduplicate all_contexts by node id to avoid wasting tokens and hurting ranking
            seen = set()
            unique_contexts = []

            for nws in all_contexts:
                node_id = nws.node.node_id
                if node_id not in seen:
                    seen.add(node_id)
                    unique_contexts.append(nws)

            all_contexts = unique_contexts
        
        # sort all contexts globally
        all_contexts = sorted(all_contexts, key=lambda x: x.score, reverse=True)

        # empty retrieval edge case
        if not all_contexts:
            return stream_text("I don't know.")
        
        feedback = ""

        # QA + Critic loop
        for attempt in range(self.max_retries + 1):
            # Step 1: QA generation
            prompt, context_text = self.build_prompt(original_query, all_contexts, feedback)
            # answer = self.reasoner_llm.generate(prompt)
            answer_stream = self.reasoner_llm.stream(prompt)
            answer = ""
            for token in answer_stream:
                answer += token

            # Step 2: Critic evaluation
            if should_critic(answer):
                critic_raw = self.run_critic(context_text, original_query, answer)  # critic_raw -> JSON
                critic = parse_json(critic_raw)                                     # critic -> dictionary

                verdict = critic.get("verdict", "incorrect")
                issues = critic.get("issues", [])
                if not issues:
                    issues = "Answer marked incorrect but no issues provided."
                corrected_answer = critic.get("corrected_answer", "I don't know.")

                # logging
                print(f"[yellow]Attempt: {attempt+1}, Verdict: {verdict}, Issues: {issues}[/yellow]")

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
        - Auto-adjust chunk size based on query
        - Reranker after retrieval
        - Code and formula format
        - Print out detailed sources info (e.g.: index, file, etc.)
"""
