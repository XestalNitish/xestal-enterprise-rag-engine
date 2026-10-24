from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from retriever import hybrid_search, retrieve

from local_llm_service import (
    DEFAULT_PORT,
    get_local_chat_llm,
    is_server_alive,
    start_local_server,
    strip_thinking,
)

logger = logging.getLogger("rag_agent")

MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "3800"))
NO_ANSWER_TEXT = "Context mein ye information nahi hai."

SYSTEM_PROMPT = (
    "You are a helpful and concise RAG assistant. Read the provided context carefully and answer the question using ONLY the given context.\n"
    f"- If the information is completely missing from the context, reply exactly: '{NO_ANSWER_TEXT}'\n"
    "- If the context contains relevant information, answer clearly and concisely in 1-2 sentences, making sure all key terms, entities, and conditions from the context are mentioned.\n"
    "- Match the language style of the question (e.g. Hinglish or English)."
)


class RAGState(TypedDict, total=False):
    question: str
    workspace_id: str
    context: str
    chunks: List[Dict[str, str]]
    sources: List[str]
    answer: str
    status: str  # "ok" | "empty" | "error"
    error: str


_active_llm: Any = None
_local_fallback_llm: Any = None


def set_active_llm(llm: Any) -> None:
    """Set the globally active live LLM for Code Dev Agent executions."""
    global _active_llm
    _active_llm = llm


def get_dev_llm(
    provider: str = "local",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1
) -> Any:
    """
    Factory to instantiate live LLM client based on requested provider or environment.
    Defaults to local offline llama-server for fast, zero-cost, reliable inference.
    """
    global _active_llm
    if _active_llm is not None and provider == "current":
        return _active_llm

    # Explicit local request or default
    if provider in ("local", "auto"):
        return get_local_chat_llm(temperature=temperature, auto_start=True)

    # 1. Groq Cloud API
    if provider == "groq":
        key = api_key or os.getenv("GROQ_API_KEY")
        if key:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=key,
                    model_name=model or "llama-3.3-70b-versatile",
                    temperature=temperature,
                    request_timeout=60,
                    max_retries=1,
                )
            except Exception as e:
                logger.warning(f"Groq initialization notice: {e}")

    # 2. Google Gemini API (OpenAI Compatible endpoint)
    if provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key and not key.startswith("#"):
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    api_key=key,
                    model_name=model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                    temperature=temperature,
                    request_timeout=60,
                    max_retries=1,
                )
            except Exception as e:
                logger.warning(f"Gemini initialization notice: {e}")

    # 3. OpenAI API
    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if key:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    api_key=key,
                    model_name=model or "gpt-4o-mini",
                    temperature=temperature,
                    request_timeout=60,
                    max_retries=1,
                )
            except Exception as e:
                logger.warning(f"OpenAI initialization notice: {e}")

    # 4. Local Spark-X2.5 LLM (llama-server) fallback
    return get_local_chat_llm(temperature=temperature, auto_start=True)


# Default initialization - local offline server
llm_base = get_dev_llm(os.getenv("RAG_PROVIDER", "local"))
_active_llm = llm_base


def normalize_chunks(result: Any) -> List[Dict[str, str]]:
    """Normalize retrieve output (str / list of str / dict / Document) -> [{'text', 'source'}]."""
    if result is None:
        return []
    if isinstance(result, (str, dict)) or hasattr(result, "page_content"):
        result = [result]

    chunks: List[Dict[str, str]] = []
    for r in result:
        if isinstance(r, str):
            text, source = r, "unknown"
        elif isinstance(r, dict):
            text = r.get("text") or r.get("page_content") or r.get("content") or ""
            meta = r.get("metadata") or {}
            source = r.get("source") or meta.get("source") or "unknown"
            if meta.get("page") is not None:
                source = f"{source} (p.{meta['page']})"
        else:  # LangChain Document
            text = getattr(r, "page_content", str(r))
            meta = getattr(r, "metadata", None) or {}
            source = meta.get("source", "unknown")
            if meta.get("page") is not None:
                source = f"{source} (p.{meta['page']})"
        text = str(text).strip()
        if text:
            chunks.append({"text": text, "source": str(source)})
    return chunks


def build_context(chunks: List[Dict[str, str]], limit: int = MAX_CONTEXT_CHARS) -> str:
    parts = [f"[{i}] ({c['source']})\n{c['text']}" for i, c in enumerate(chunks, 1)]
    return "\n\n".join(parts)[:limit]


def retrieve_node(state: RAGState) -> RAGState:
    try:
        raw_result = hybrid_search(state["question"], state["workspace_id"])
        chunks = normalize_chunks(raw_result)
    except Exception as e:
        logger.exception("retrieve_node failed")
        return {"status": "error", "error": f"Retriever failed: {e}", "chunks": [], "sources": [], "context": ""}

    if not chunks:
        return {"status": "empty", "chunks": [], "sources": [], "context": ""}

    context_str = build_context(chunks)
    return {
        "status": "ok",
        "chunks": chunks,
        "sources": sorted({c["source"] for c in chunks}),
        "context": context_str,
    }


def generate_node(state: RAGState) -> RAGState:
    global _active_llm, _local_fallback_llm
    try:
        prompt = f"Context:\n{state.get('context', '')}\n\nQuestion: {state['question']}\n\n"
        logger.info("Generator received context")
        
        resp = None
        try:
            resp = _active_llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        except Exception as e:
            err_msg = str(e).lower()
            if any(term in err_msg for term in ["429", "quota", "resource_exhausted", "limit exceeded"]):
                logger.warning("Cloud LLM rate/quota limit reached; failing over to local Spark-X2.5 LLM...")
                if _local_fallback_llm is None:
                    _local_fallback_llm = get_local_chat_llm(temperature=0.1, auto_start=True)
                resp = _local_fallback_llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
            else:
                raise e

        answer = strip_thinking(getattr(resp, "content", str(resp)))
        return {"answer": answer}
    except Exception as e:
        logger.exception("generate failed")
        return {"status": "error", "error": f"LLM failed: {e}", "answer": f"LLM error: {e}"}


def no_answer_node(state: RAGState) -> RAGState:
    if state.get("status") == "error":
        return {"answer": f"Retrieval error: {state.get('error', 'unknown')}"}
    return {"answer": NO_ANSWER_TEXT}


def route_after_retrieve(state: RAGState) -> str:
    return "generate" if state.get("status") == "ok" else "no_answer"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("no_answer", no_answer_node)

    g.add_edge(START, "retrieve")
    g.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "no_answer": "no_answer"},
    )
    g.add_edge("generate", END)
    g.add_edge("no_answer", END)
    return g.compile()


rag_app = build_graph()


def ask(question: str, workspace_id: str) -> Dict[str, Any]:
    """Returns {"answer", "sources", "chunks", "status", "error"}."""
    out = rag_app.invoke({"question": question, "workspace_id": workspace_id})
    return {
        "answer": out.get("answer", ""),
        "sources": out.get("sources", []),
        "chunks": out.get("chunks", []),
        "status": out.get("status", ""),
        "error": out.get("error"),
    }


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    q = sys.argv[1] if len(sys.argv) > 1 else "Abstract ke hisaab se ye paper kis coordination problem ko model karta hai?"
    ws = sys.argv[2] if len(sys.argv) > 2 else "ws1"

    res = ask(q, ws)
    print("\nSTATUS :", res["status"])
    print("ANSWER :", res["answer"])
    print("SOURCES:", res["sources"])
    if res["error"]:
        print("ERROR  :", res["error"])
    for i, c in enumerate(res["chunks"], 1):
        print(f"\n--- chunk {i} ({c['source']}) ---\n{c['text'][:300]}")
