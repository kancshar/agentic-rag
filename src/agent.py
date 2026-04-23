"""
agent.py — Routing agent that decides whether a question can be answered
           from the local docs or requires a web search.

Decision flow:
  1. Retrieve top-k chunks from FAISS  (via Retriever)
  2. Ask the LLM: "Can you answer the question from these chunks? YES or NO"
  3a. YES → answer using RAG (the retrieved chunks)
  3b. NO  → invoke web_search tool → answer using web results

This is the classic "retrieval grader" pattern in agentic RAG.
"""

import os
import pathlib

from dotenv import load_dotenv
from groq import Groq

from retriever import Retriever
from web_search import web_search
from generator import generate

# ── Env ────────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GROQ_MODEL = "llama-3.1-8b-instant"

# L2 distance threshold for all-MiniLM-L6-v2 embeddings.
# If the best (lowest) score is above this, the docs are clearly off-topic
# and we skip the LLM grader entirely and go straight to web search.
# Typical ranges: <0.5 excellent, 0.5-1.0 decent, 1.0-1.5 weak, >1.5 unrelated.
SCORE_THRESHOLD = 1.2

# ── Relevance-grading prompt ───────────────────────────────────────────────────
GRADER_SYSTEM = """\
You are a relevance grader. Your job is to decide whether the retrieved
context contains enough information to answer the user's question.

Rules:
- Reply with exactly one word: YES or NO.
- YES  → the context clearly addresses the question.
- NO   → the context is off-topic, vague, or insufficient.
- Do NOT explain your answer.
"""


def _grade_relevance(query: str, chunks: list[dict]) -> bool:
    """
    Ask the LLM whether the retrieved chunks are sufficient to answer the query.

    Returns True if the LLM says YES (use docs), False if NO (use web).
    """
    if not chunks:
        return False

    context_preview = "\n\n".join(
        f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(chunks[:3])
    )
    user_msg = (
        f"Question: {query}\n\n"
        f"Retrieved context:\n{context_preview}\n\n"
        "Can the question be answered from this context? Reply YES or NO."
    )

    api_key = os.getenv("GROQ_API_KEY", "")
    client  = Groq(api_key=api_key)
    resp    = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=5,
        temperature=0.0,
    )

    verdict = resp.choices[0].message.content.strip().upper()
    return verdict.startswith("YES")


class AgentPipeline:
    """
    Agentic RAG pipeline with a web-search fallback tool.

    ask() returns:
        {
          answer:  str,
          sources: list[str],
          chunks:  list[dict],
          source:  "docs" | "web",   ← where the answer came from
        }
    """

    def __init__(self, docs_set: str, top_k: int = 5):
        self.retriever = Retriever(docs_set, top_k=top_k)

    def ask(self, query: str) -> dict:
        # Step 1 — retrieve from local docs
        chunks = self.retriever.retrieve(query)

        # Step 2 — fast-path: if best FAISS score is too high, docs are off-topic
        best_score = min((c["score"] for c in chunks), default=float("inf"))
        if best_score > SCORE_THRESHOLD:
            use_docs = False
            route_reason = f"score {best_score:.3f} > threshold {SCORE_THRESHOLD} → web"
        else:
            # Step 2b — LLM grader for ambiguous cases
            use_docs     = _grade_relevance(query, chunks)
            route_reason = f"score {best_score:.3f} ≤ threshold, LLM grader → {'docs' if use_docs else 'web'}"

        if use_docs:
            # Step 3a — answer from docs
            result            = generate(query, chunks)
            result["chunks"]  = chunks
            result["source"]  = "docs"
        else:
            # Step 3b — invoke web search tool
            web_chunks        = web_search(query)
            result            = generate(query, web_chunks)
            result["chunks"]  = web_chunks
            result["source"]  = "web"

        result["route_reason"] = route_reason
        return result
