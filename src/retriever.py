"""
retriever.py — Phase 2: Load FAISS index → Embed query → Top-k search

Usage (standalone test):
    python src/retriever.py --docs requests_docs --query "how to send a POST request"
"""

import argparse
import json
import pathlib

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "index"

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 5


class Retriever:
    """
    Loads a FAISS index for a given docs set and answers vector-similarity
    queries against it.
    """

    def __init__(self, docs_set: str, top_k: int = TOP_K):
        self.docs_set = docs_set
        self.top_k    = top_k

        index_path    = INDEX_DIR / docs_set / "faiss_index.bin"
        metadata_path = INDEX_DIR / docs_set / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"No FAISS index found for '{docs_set}'. "
                f"Run: python src/ingest.py --docs {docs_set}"
            )

        self.index    = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata: list[dict] = json.load(f)

        self.model = SentenceTransformer(EMBED_MODEL)
        print(f"Retriever ready — '{docs_set}' ({self.index.ntotal} vectors)")

    def retrieve(self, query: str) -> list[dict]:
        """
        Embed the query and return the top-k most similar chunks.

        Returns a list of dicts:
            {text, source, score}
        where score is the L2 distance (lower = more similar).
        """
        query_vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)

        distances, indices = self.index.search(query_vec, self.top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:           # FAISS returns -1 when fewer results exist
                continue
            chunk = self.metadata[idx]
            results.append({
                "text":   chunk["text"],
                "source": chunk["source"],
                "score":  float(dist),
            })

        return results


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the retriever")
    parser.add_argument("--docs",  required=True, help="Docs set subfolder name")
    parser.add_argument("--query", required=True, help="Query string")
    args = parser.parse_args()

    retriever = Retriever(args.docs)
    results   = retriever.retrieve(args.query)

    print(f"\nTop-{TOP_K} results for: \"{args.query}\"\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] source={r['source']}  score={r['score']:.4f}")
        print(f"    {r['text'][:200]} ...")
        print()
