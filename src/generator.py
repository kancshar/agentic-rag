"""
generator.py — Phase 3: Build prompt → Call Groq → Return answer + sources

Reads GROQ_API_KEY from .env
"""

import os
import pathlib

from dotenv import load_dotenv
from groq import Groq

# ── Load env ───────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GROQ_MODEL  = "llama-3.1-8b-instant"
MAX_CONTEXT = 4          # max chunks included in the prompt
MAX_TOKENS  = 512


SYSTEM_PROMPT = """\
You are a helpful assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find enough information to answer."
Always be concise and accurate. Do not make up information.
Cite the source when useful.
"""


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks[:MAX_CONTEXT], 1):
        parts.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")
    return "\n\n".join(parts)


def generate(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Build a RAG prompt and call Groq.

    Args:
        query:            The user's question.
        retrieved_chunks: List of {text, source, score} dicts from the retriever.

    Returns:
        {answer: str, sources: list[str]}
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "":
        raise ValueError(
            "GROQ_API_KEY is not set. Add your key to the .env file."
        )

    context_block = build_context_block(retrieved_chunks)

    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}"
    )

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.2,
    )

    answer  = response.choices[0].message.content.strip()
    sources = list(dict.fromkeys(c["source"] for c in retrieved_chunks[:MAX_CONTEXT]))

    return {"answer": answer, "sources": sources}
