"""
ingest.py — Phase 1: Load → Parse → Chunk → Embed → Save FAISS Index

Usage:
    python src/ingest.py --docs requests_docs

Reads all .html / .txt / .pdf files from  data/<docs_set>/
Saves FAISS index + metadata to            index/<docs_set>/
"""

import argparse
import json
import os
import pathlib
import re

import faiss
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
INDEX_DIR  = BASE_DIR / "index"

EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500   # characters
CHUNK_OVERLAP = 50    # characters


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text_html(filepath: pathlib.Path) -> str:
    """Parse HTML and return visible text, stripping nav/script/style."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def extract_text_txt(filepath: pathlib.Path) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _clean_pdf_text(text: str) -> str:
    """
    Clean raw pdfminer output:
    - Re-join words broken across lines with a hyphen (de-hyphenation)
    - Collapse runs of whitespace/newlines into single spaces
    - Strip leading/trailing whitespace
    """
    # Re-join hyphenated line-breaks: "connec-\ntion" → "connection"
    text = re.sub(r"-\s*\n\s*", "", text)
    # Replace remaining newlines / tabs with a space
    text = re.sub(r"[\n\t\r]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_pdf_pages(filepath: pathlib.Path) -> list[tuple[int, str]]:
    """
    Extract text from each PDF page individually.

    Returns a list of (page_number, cleaned_text) tuples (1-indexed).
    Pages with no extractable text are skipped.
    """
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    import io

    pages: list[tuple[int, str]] = []
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()

    with open(filepath, "rb") as fh:
        for page_num, page in enumerate(PDFPage.get_pages(fh, check_extractable=True), start=1):
            output = io.StringIO()
            device = TextConverter(rsrcmgr, output, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            interpreter.process_page(page)
            device.close()
            raw = output.getvalue()
            output.close()

            cleaned = _clean_pdf_text(raw)
            if cleaned:
                pages.append((page_num, cleaned))

    return pages


def extract_text(filepath: pathlib.Path) -> str:
    """Single-string extraction for .html and .txt (used by chunk_text)."""
    suffix = filepath.suffix.lower()
    if suffix == ".html":
        return extract_text_html(filepath)
    elif suffix == ".txt":
        return extract_text_txt(filepath)
    else:
        return ""


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Sliding-window character chunking.
    Returns a list of dicts: {text, source}
    """
    chunks = []
    text = " ".join(text.split())   # normalize whitespace
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source})
        start += chunk_size - overlap
    return chunks


# ── Main ───────────────────────────────────────────────────────────────────────

def ingest(docs_set: str) -> None:
    docs_path  = DATA_DIR / docs_set
    index_path = INDEX_DIR / docs_set

    if not docs_path.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_path}")

    index_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect all supported files ────────────────────────────────────────
    supported = {".html", ".txt", ".pdf"}
    files = [f for f in docs_path.rglob("*") if f.suffix.lower() in supported]

    if not files:
        raise ValueError(f"No .html / .txt / .pdf files found in {docs_path}")

    print(f"Found {len(files)} file(s) in '{docs_set}'")

    # ── 2. Extract text + chunk ───────────────────────────────────────────────
    all_chunks: list[dict] = []
    for filepath in files:
        print(f"  Processing: {filepath.name}")
        if filepath.suffix.lower() == ".pdf":
            # Extract page-by-page so chunks carry a page reference
            pages = extract_pdf_pages(filepath)
            for page_num, page_text in pages:
                source_label = f"{filepath.name}#p{page_num}"
                chunks = chunk_text(page_text, source=source_label)
                all_chunks.extend(chunks)
        else:
            text = extract_text(filepath)
            chunks = chunk_text(text, source=filepath.name)
            all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    print(f"Loading embedding model '{EMBED_MODEL}' ...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in all_chunks]
    print("Embedding chunks (this may take a moment) ...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # ── 4. Build FAISS index ──────────────────────────────────────────────────
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    faiss_path    = index_path / "faiss_index.bin"
    metadata_path = index_path / "metadata.json"

    faiss.write_index(index, str(faiss_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved index    → {faiss_path}")
    print(f"Saved metadata → {metadata_path}")
    print("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a docs set into FAISS")
    parser.add_argument("--docs", required=True, help="Subfolder name inside data/")
    args = parser.parse_args()
    ingest(args.docs)
