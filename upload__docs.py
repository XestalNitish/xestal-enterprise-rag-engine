import os
from typing import Any, Dict, List, Literal, Optional, TypedDict
from langgraph.graph import END, StateGraph
from tools import docx_reader, pdf_reader, video_reader, web_reader, youtube
from database import chroma_store, smart_chunk_document


class UploadState(TypedDict, total=False):
    file_path: str
    file_type: Literal["docs", "pdf", "link", "youtube_link", "local_video"]
    workspace_id: str
    source: Optional[str]
    loader: Any
    structured_chunks: List[Dict[str, Any]]
    status: str
    db_store: Any
    collection_name: str


LOADERS = {
    "docs": docx_reader,
    "pdf": pdf_reader,
    "link": web_reader,
    "local_video": video_reader,
    "youtube_link": youtube,
}


def upload_documents(state: UploadState):
    print("--- STEP 1: LOADING DOCUMENT ---")
    try:
        tool = LOADERS.get(state["file_type"])
        if tool is None:
            raise ValueError(f"Unsupported file type: {state['file_type']}")
        loader = tool.invoke(state["file_path"])
        return {"loader": loader, "status": "loaded"}
    except Exception as e:
        return {"status": f"error: loading failed: {e}"}


def chunking(state: UploadState):
    print("--- STEP 2: CHUNKING DOCUMENT WITH METADATA ---")
    source = state.get("source") or state["file_path"]
    try:
        data = smart_chunk_document.invoke({
            "text_content": state["loader"],
            "source_name": source,
            "workspace_id": state["workspace_id"],
        })
        return {"structured_chunks": data, "source": source, "status": "chunked"}
    except Exception as e:
        return {"status": f"error: chunking failed: {e}"}


def store_chunks(state: UploadState):
    print("--- STEP 3: STORING IN CHROMADB ---")
    try:
        col_name = state.get("collection_name") or os.getenv("RAG_COLLECTION", "enterprise_docs")
        store = chroma_store.invoke({
            "structured_chunks": state["structured_chunks"],
            "workspace_id": state["workspace_id"],
            "collection_name": col_name,
        })
        return {"db_store": store, "status": "done"}
    except Exception as e:
        return {"status": f"error: storing failed: {e}"}


def check(next_node):
    def router(state: UploadState):
        return END if str(state.get("status", "")).startswith("error") else next_node
    return router


graph = StateGraph(UploadState)
graph.add_node("load", upload_documents)
graph.add_node("chunk", chunking)
graph.add_node("store", store_chunks)
graph.set_entry_point("load")
graph.add_conditional_edges("load", check("chunk"))
graph.add_conditional_edges("chunk", check("store"))
graph.add_edge("store", END)
app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({
        "file_path": "2610.30074v1.pdf",
        "file_type": "pdf",
        "workspace_id": "ws1",
        "collection_name": os.getenv("RAG_COLLECTION", "enterprise_docs"),
    })
    print("Upload Result Status:", result.get("status"))
