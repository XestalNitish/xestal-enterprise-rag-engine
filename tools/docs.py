import os
from typing import Dict, List, Tuple

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.tools import tool

try:
    from tools._safe import safe_local_path
except ImportError:  # running the file directly / flat layout
    from _safe import safe_local_path

ERR = "[ERROR]"
NO_RESULTS = "[NO_RESULTS]"
TEXT_EXTS = (".txt", ".md", ".csv")


def _docx_blocks(path: str) -> List[Tuple]:
    """Walk the document body in real order. Yields ("heading", level, text),
    ("para", text) or ("table", markdown)."""
    document = docx.Document(path)
    blocks: List[Tuple] = []
    for element in document.element.body:
        if element.tag.endswith("}p"):
            para = Paragraph(element, document)
            text = para.text.strip()
            if not text:
                continue
            style = para.style.name if para.style is not None else ""
            if style == "Title":
                blocks.append(("heading", 1, text))
            elif style.startswith("Heading"):
                try:
                    level = int(style.split()[-1])
                except ValueError:
                    level = 3
                blocks.append(("heading", level, text))
            else:
                blocks.append(("para", text))
        elif element.tag.endswith("}tbl"):
            table = Table(element, document)
            rows = []
            for r_idx, row in enumerate(table.rows):
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
                if r_idx == 0:
                    rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            if rows:
                blocks.append(("table", "\n".join(rows)))
    return blocks


def load_docx_sections(source: str) -> List[Dict]:
    """Structured loader for ingestion: one dict per heading section.

    Returns [{"text": "...", "metadata": {"source", "heading_path"}}] where heading_path is
    like "Policy Handbook > Leave Types", so chunks keep their place in the document.
    """
    path = safe_local_path(source)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found at path: {source}")

    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTS:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [{"text": content, "metadata": {"source": source, "heading_path": ""}}] if content.strip() else []
    if ext != ".docx":
        raise ValueError(f"Unsupported file extension '{ext}'. Use .docx, .txt, .md or .csv.")

    sections: List[Dict] = []
    stack: List[str] = []
    buf: List[str] = []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append({"text": text,
                             "metadata": {"source": source, "heading_path": " > ".join(stack)}})
        buf.clear()

    for block in _docx_blocks(path):
        if block[0] == "heading":
            flush()
            level, title = block[1], block[2]
            stack[:] = stack[: level - 1] + [title]
        else:
            buf.append(block[1])
    flush()
    return sections


@tool
def docx_reader(source: str, max_chars: int = 50000) -> str:
    """Robust parser for Microsoft Word (.docx) and plain text (.txt, .md, .csv) files.
    Preserves headings, paragraphs, and tables in Markdown format.

    Args:
        source: Local file path to a .docx, .txt, .md or .csv file.
        max_chars: Maximum characters to return.
    """
    try:
        path = safe_local_path(source)
        if not os.path.isfile(path):
            return f"{ERR} File not found at path: {source}"

        ext = os.path.splitext(path)[1].lower()

        if ext in TEXT_EXTS:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if not content.strip():
                return f"{NO_RESULTS} The text file is empty."
            return f"Source: {source}\nType: Plain Text\n\n{content[:max_chars]}"

        if ext == ".docx":
            out = []
            for block in _docx_blocks(path):
                if block[0] == "heading":
                    out.append(f"\n{'#' * block[1]} {block[2]}\n")
                elif block[0] == "table":
                    out.append("\n" + block[1] + "\n")
                else:
                    out.append(block[1])
            full_text = "\n".join(out)
            if not full_text.strip():
                return f"{NO_RESULTS} No readable text or tables found in the .docx file."
            return f"Source: {source}\nType: Microsoft Word (.docx)\n\n{full_text[:max_chars]}"

        return f"{ERR} Unsupported file extension '{ext}'. Use .docx, .txt, .md or .csv."

    except PermissionError as e:
        return f"{ERR} {e}"
    except Exception as e:
        return f"{ERR} docx_reader error: {type(e).__name__}: {str(e)[:150]}"
