import os
import re
import unicodedata
from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

@tool
def smart_chunk_document(
    text_content: str,
    source_name: str,
    workspace_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """Splits raw document text into intelligent semantic chunks with enterprise metadata.
    Args:
        text_content: Full text extracted from pdf_reader, docx_reader, or web_reader.
        source_name: Name or path of the source file/URL.
        workspace_id: Enterprise workspace or project ID for multi-tenant isolation.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks to retain context continuity.
    Returns:
        A list of dictionaries containing chunk text and rich metadata.
    """
    try:
        # Normalize mathematical symbols and whitespace
        normalized_content = unicodedata.normalize("NFKC", text_content)
        normalized_content = normalized_content.replace("−", "-").replace("–", "-").replace("—", "-")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        raw_chunks = splitter.split_text(normalized_content)
        structured_chunks = []
        current_page = 1

        for idx, chunk in enumerate(raw_chunks):
            # Track page transitions: e.g. "--- Page 3 ---"
            page_matches = re.findall(r"---\s*Page\s*(\d+)\s*---", chunk)
            if page_matches:
                current_page = int(page_matches[-1])

            meta = {
                "source": source_name,
                "workspace_id": workspace_id,
                "chunk_index": idx,
                "total_chunks": len(raw_chunks),
                "page": current_page,
            }

            chunk_record = {
                "chunk_id": f"{workspace_id}_{abs(hash(source_name))}_{idx}",
                "text": chunk.strip(),
                "metadata": meta,
            }
            structured_chunks.append(chunk_record)

        return structured_chunks
    except Exception as e:
        print(f"[ERROR] Chunking failed for {source_name}:{str(e)}")
        return []
