"""
retriever.py - Hybrid retriever: BM25 (keyword) + Chroma MMR (semantic)

Do tarike se use hota hai:
    hybrid_search(query, workspace_id) -> List[Document]   # structured, error par exception raise karta hai
    retrieve (LangChain tool)          -> str              # purana API, text return karta hai

Debug:
    python retriever.py "paper ka title kya hai?" ws1
"""
from __future__ import annotations

import logging
import os
import pickle
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("retriever")

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION = os.getenv("RAG_COLLECTION", "enterprise_docs")
TOP_N = int(os.getenv("RAG_TOP_N", "6"))
BM25_K = int(os.getenv("RAG_BM25_K", "12"))
VEC_K = int(os.getenv("RAG_VEC_K", "10"))
FETCH_K = int(os.getenv("RAG_FETCH_K", "35"))
USE_RERANK = os.getenv("RAG_USE_RERANK", "0") == "1"


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
def _find_path(rel: str) -> Path:
    bases = [os.getenv("RAG_BASE_DIR"), Path(__file__).resolve().parent, Path.cwd()]
    tried: List[str] = []
    for b in bases:
        if not b:
            continue
        p = Path(b) / rel
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(f"'{rel}' nahi mila. Dhoondha: {' | '.join(tried)}")


# ─────────────────────────────────────────────────────────────
# Hinglish query cleaning
# ─────────────────────────────────────────────────────────────
_HINGLISH_MARKERS = {
    "kya", "hai", "hain", "ka", "ki", "ke", "ko", "se", "me", "mein", "kaun", "kaunsa", "kaunse",
    "kitna", "kitne", "kitni", "kaise", "kyun", "kahan", "batao", "bataiye", "hisaab", "karta",
    "hota", "hua", "hue", "tha", "wale", "wala", "hote", "hoti", "karti", "karte",
}
_HINGLISH_STOP = _HINGLISH_MARKERS | {
    "is", "ye", "yeh", "wo", "woh", "aur", "ya", "par", "pe", "to", "toh", "do", "de", "ek", "iska",
    "uska", "jo", "jab", "tab", "bhi", "hi", "nahi", "liye", "gaya", "gaye", "bata", "batana",
    "samjhao", "dikhao", "kab", "jahan", "kis", "kisi", "kuch", "sab", "saare", "poora", "pura",
    "unme", "isme", "usme", "iske", "uske", "unka", "inka", "the",
}
_STRIP_CHARS = "?.,;:!\"'"


def clean_query(query: str) -> str:
    """Normalize and remove Hinglish conversational filler words while preserving mathematical terms."""
    q_norm = unicodedata.normalize("NFKC", query)
    q_norm = q_norm.replace("−", "-").replace("–", "-").replace("—", "-")
    tokens = [t.strip(_STRIP_CHARS) for t in q_norm.split()]
    tokens = [t for t in tokens if t]
    if not any(t.lower() in _HINGLISH_MARKERS for t in tokens):
        return q_norm
    kept = [t for t in tokens if t.lower() not in _HINGLISH_STOP]
    return " ".join(kept) if kept else q_norm


# ─────────────────────────────────────────────────────────────
# Cached heavy objects
# ─────────────────────────────────────────────────────────────
_embeddings: Any = None
_vector_stores: Dict[str, Any] = {}
_bm25_cache: Dict[str, Tuple[float, Any]] = {}


def _get_embeddings() -> Any:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_vector_store() -> Any:
    path = str(_find_path("test_db_store"))
    col = os.getenv("RAG_COLLECTION", COLLECTION)
    key = f"{path}_{col}"
    if key not in _vector_stores:
        _vector_stores[key] = Chroma(
            persist_directory=path,
            embedding_function=_get_embeddings(),
            collection_name=col,
        )
    return _vector_stores[key]


def _tokenize(text: str) -> List[str]:
    norm_text = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"\w+", norm_text)


def _get_bm25(workspace_id: str) -> Any:
    pkl = _find_path(f"workspace_docs/{workspace_id}_docs.pkl")
    mtime = pkl.stat().st_mtime
    cached = _bm25_cache.get(workspace_id)
    if cached and cached[0] == mtime:
        return cached[1]
    with pkl.open("rb") as f:
        documents = pickle.load(f)
    if not documents:
        raise ValueError(f"{pkl} me koi document nahi hai")
    bm25 = BM25Retriever.from_documents(documents, preprocess_func=_tokenize)
    bm25.k = BM25_K
    _bm25_cache[workspace_id] = (mtime, bm25)
    return bm25


# ─────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────
def hybrid_search(query: str, workspace_id: str, top_n: Optional[int] = None) -> List[Document]:
    """BM25 + MMR vector -> ensemble. Error par exception raise karta hai."""
    top_n = top_n or TOP_N
    q = clean_query(query)

    bm25 = _get_bm25(workspace_id)
    vector_retriever = _get_vector_store().as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": VEC_K,
            "fetch_k": FETCH_K,
            "lambda_mult": 0.7,
            "filter": {"workspace_id": workspace_id},
        },
    )
    hybrid = EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.5, 0.5])

    if USE_RERANK:
        try:
            from langchain_classic.retrievers import ContextualCompressionRetriever
            from langchain_classic.retrievers.document_compressors import FlashrankRerank
            reranker = FlashrankRerank(model="rank-T5-flan", top_n=top_n)
            compression = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=hybrid
            )
            docs = compression.invoke(q)
        except Exception:
            logger.exception("Rerank fail hua, fallback to hybrid results")
            docs = hybrid.invoke(q)[:top_n]
    else:
        docs = hybrid.invoke(q)[:top_n]

    logger.info("query=%r cleaned=%r -> %d chunks", query, q, len(docs))
    return docs


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


@tool
def retrieve(query: str, workspace_id: str) -> str:
    """
    Creates a Hybrid Retriever combining BM25 (Keyword Search) and Chroma (Vector/Semantic Search).

    Args:
        query (str): The search question or keyword.
        workspace_id (str): The workspace ID for multi-tenant isolation.
    """
    try:
        return format_docs(hybrid_search(query, workspace_id))
    except Exception as e:  # noqa: BLE001
        logger.exception("retrieve tool failed")
        return f"[ERROR] Hybrid retriever failed: {e}"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    q = sys.argv[1] if len(sys.argv) > 1 else "paper ka title kya hai?"
    ws = sys.argv[2] if len(sys.argv) > 2 else "ws1"
    print(f"query   : {q}\ncleaned : {clean_query(q)}\n")
    for i, d in enumerate(hybrid_search(q, ws), 1):
        meta = d.metadata or {}
        print(f"--- {i} | source={meta.get('source')} page={meta.get('page')} ---")
        print(d.page_content[:300].replace("\n", " "), "\n")
