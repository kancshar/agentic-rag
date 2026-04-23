"""
app.py — Streamlit UI for the RAG Q&A Bot

Run:
    streamlit run app.py
"""

import pathlib
import sys
import os

import streamlit as st

# ── Secrets: Streamlit Cloud injects via st.secrets; local dev uses .env ───────
# We push secrets into os.environ early so every module (generator, agent)
# can read them with os.getenv() regardless of environment.
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ── Make src/ importable ───────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from agent import AgentPipeline    # noqa: E402

DATA_DIR  = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Docs Q&A Bot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Docs Q&A Bot")
st.caption("Agentic RAG · FAISS · Groq · DuckDuckGo fallback")


# ── Sidebar — docs set selector ────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # Discover available docs sets by scanning data/
    if DATA_DIR.exists():
        docs_sets = sorted(
            [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
        )
    else:
        docs_sets = []

    if not docs_sets:
        st.warning("No docs sets found.\nCreate a folder inside `data/` and add files.")
        st.stop()

    selected_docs = st.selectbox("Select Docs Set", docs_sets)

    # Check whether index exists for this docs set
    index_exists = (INDEX_DIR / selected_docs / "faiss_index.bin").exists()

    if not index_exists:
        st.error(
            f"No index found for **{selected_docs}**.\n\n"
            f"Run ingestion first:\n```\npython src/ingest.py --docs {selected_docs}\n```"
        )
        st.stop()

    st.success(f"Index loaded: **{selected_docs}**")
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=5)


# ── Load pipeline (cached per docs_set + top_k) ───────────────────────────────
@st.cache_resource(show_spinner="Loading index and embedding model ...")
def load_pipeline(docs_set: str, k: int) -> AgentPipeline:
    return AgentPipeline(docs_set, top_k=k)


pipeline = load_pipeline(selected_docs, top_k)


# ── Chat history ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "chunks" in msg:
            from_source    = msg.get("source", "docs")
            expander_label = "🌐 Web Sources" if from_source == "web" else "📄 Doc Sources"
            with st.expander(expander_label, expanded=False):
                for i, chunk in enumerate(msg["chunks"], 1):
                    if from_source == "web":
                        st.markdown(f"**[{i}]** [{chunk['source']}]({chunk['source']})")
                    else:
                        st.markdown(f"**[{i}] {chunk['source']}** &nbsp; `score: {chunk['score']:.4f}`")
                    st.caption(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))


# ── Input ──────────────────────────────────────────────────────────────────────
query = st.chat_input(f"Ask a question about {selected_docs} ...")

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking ..."):
            try:
                result      = pipeline.ask(query)
                answer      = result["answer"]
                chunks      = result["chunks"]
                from_source = result.get("source", "docs")

                # Badge: show where the answer came from
                if from_source == "web":
                    st.info("🌐 **Answer sourced from web search** — question was outside the docs.")
                else:
                    st.success("📄 **Answer sourced from docs**")
                st.caption(f"🔍 Routing: {result.get('route_reason', '')}")

                st.markdown(answer)

                expander_label = "🌐 Web Sources" if from_source == "web" else "📄 Doc Sources"
                with st.expander(expander_label, expanded=False):
                    for i, chunk in enumerate(chunks, 1):
                        if from_source == "web":
                            st.markdown(f"**[{i}]** [{chunk['source']}]({chunk['source']})")
                        else:
                            st.markdown(f"**[{i}] {chunk['source']}** &nbsp; `score: {chunk['score']:.4f}`")
                        st.caption(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "chunks":  chunks,
                    "source":  from_source,
                })

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")
