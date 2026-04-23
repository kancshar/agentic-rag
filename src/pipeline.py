"""
pipeline.py — Orchestrates Retriever + Generator into a single ask() call.

Usage (standalone test):
    python src/pipeline.py --docs requests_docs --query "how do I set a timeout?"
"""

import argparse

from retriever import Retriever
from generator import generate


class RAGPipeline:
    """
    Thin orchestration layer:
      query → Retriever.retrieve() → generate() → {answer, sources, chunks}
    """

    def __init__(self, docs_set: str, top_k: int = 5):
        self.retriever = Retriever(docs_set, top_k=top_k)

    def ask(self, query: str) -> dict:
        """
        Run the full RAG pipeline.

        Returns:
            {
              answer:  str,
              sources: list[str],        # deduplicated source filenames
              chunks:  list[dict],       # raw retrieved chunks for display
            }
        """
        chunks = self.retriever.retrieve(query)
        result = generate(query, chunks)
        result["chunks"] = chunks
        return result


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the full RAG pipeline")
    parser.add_argument("--docs",  required=True, help="Docs set subfolder name")
    parser.add_argument("--query", required=True, help="Question to ask")
    args = parser.parse_args()

    pipeline = RAGPipeline(args.docs)
    result   = pipeline.ask(args.query)

    print("\n── Answer ────────────────────────────────────────────")
    print(result["answer"])
    print("\n── Sources ───────────────────────────────────────────")
    for s in result["sources"]:
        print(f"  • {s}")
    print("\n── Retrieved Chunks ──────────────────────────────────")
    for i, c in enumerate(result["chunks"], 1):
        print(f"[{i}] {c['source']}  (score: {c['score']:.4f})")
        print(f"    {c['text'][:200]} ...")
        print()
