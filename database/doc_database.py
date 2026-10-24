import os
import pickle
from typing import Any, List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=_EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


@tool
def chroma_store(
    structured_chunks: list,
    workspace_id: str,
    collection_name: Optional[str] = None
) -> Any:
    """
    Stores structured chunks into ChromaDB along with their metadata.
    """
    try:
        col_name = collection_name or os.getenv("RAG_COLLECTION", "enterprise_docs")
        embedding = get_embeddings()
        documents = []
        for chunk in structured_chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"],
            )
            documents.append(doc)

        docs_folder = "./workspace_docs"
        os.makedirs(docs_folder, exist_ok=True)
        file_path = os.path.join(docs_folder, f"{workspace_id}_docs.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(documents, f)

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory="./test_db_store",
            collection_name=col_name,
        )
        print(f"Successfully stored {len(documents)} chunks in ChromaDB for workspace: {workspace_id}")
        return vector_store, documents
    except Exception as e:
        print(f"[ERROR] Storing in ChromaDB failed: {str(e)}")
        return None
