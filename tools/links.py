import json
import re
import urllib.parse
from typing import Dict, Tuple

from bs4 import BeautifulSoup
from langchain_core.tools import tool

try:
    from tools._safe import safe_get
except ImportError:
    from _safe import safe_get
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

ERR = "[ERROR]"
NO_RESULTS = "[NO_RESULTS]"


def _extract_text(html: str, url: str) -> Tuple[str, str]:
    """Return (title, clean_text). Trafilatura first, BeautifulSoup as fallback."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    text = ""
    try:
        import trafilatura
        text = trafilatura.extract(
            html, url=url, include_tables=True, include_formatting=True, include_links=False
        ) or ""
    except ImportError:
        pass
    if len(text.strip()) < 200:
        for tag in soup(["script", "style", "nav", "footer", "noscript", "aside", "form", "header", "iframe"]):
            tag.decompose()
        main = (soup.find("main") or soup.find("article")
                or soup.find("div", class_=re.compile("content|body|main", re.I)))
        target = main if main else soup
        lines = [ln.strip() for ln in target.get_text(separator="\n").splitlines() if ln.strip()]
        fallback = "\n".join(lines)
        if len(fallback) > len(text):
            text = fallback
    return title, text


def load_web_page(url: str, max_chars: int = 15000) -> Dict:
    """Structured loader for ingestion: {"text": ..., "metadata": {...}}. Raises on failure."""
    resp = safe_get(url, headers=BROWSER_HEADERS, timeout=20)
    if resp.status_code != 200:
        raise IOError(f"HTTP {resp.status_code} for URL: {url}")
    ctype = resp.headers.get("Content-Type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        raise ValueError("This URL points to a PDF file. Use pdf_reader instead.")
    title, text = _extract_text(resp.text, url)
    return {"text": text[:max_chars], "metadata": {"source": url, "title": title, "type": "web"}}


@tool
def web_reader(url: str, max_chars: int = 15000) -> str:
    """Robust web page reader to extract clean, full text content from documentation,
    blogs, articles, and GitHub READMEs with fallbacks.

    Args:
        url: Full URL to read.
        max_chars: Max characters to return (default 15,000).
    """
    try:
        page = load_web_page(url, max_chars=max_chars)
        if not page["text"].strip():
            return f"{NO_RESULTS} No readable text found at {url}. The page might require JavaScript rendering (SPA)."
        return f"Source: {url}\n\n{page['text']}"
    except PermissionError as e:
        return f"{ERR} {e}"
    except Exception as e:
        # requests.exceptions.Timeout etc. also land here with a clear type name
        return f"{ERR} web_reader({url}): {type(e).__name__}: {str(e)[:150]}"


@tool
def browser_agent(url: str, extract_type: str = "summary") -> str:
    """Inspect a web page structure: title, meta description, headings, and outbound links.

    Args:
        url: Page URL.
        extract_type: 'summary' (title, description, headings), 'links', or 'full_structure' (both).
    """
    try:
        resp = safe_get(url, headers=BROWSER_HEADERS, timeout=15)
        if resp.status_code != 200:
            return f"{ERR} HTTP {resp.status_code} for {url}"

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        md = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        desc = (md.get("content") or "").strip() if md else ""

        data = {"url": url, "status_code": resp.status_code, "title": title, "meta_description": desc}

        if extract_type in ("summary", "full_structure"):
            headings = [{"level": h.name, "text": h.get_text(" ", strip=True)}
                        for h in soup.find_all(["h1", "h2", "h3", "h4"])]
            data["headings"] = headings[:30]

        if extract_type in ("links", "full_structure"):
            seen, unique_links = set(), []
            for a in soup.find_all("a", href=True):
                href = urllib.parse.urljoin(url, a["href"])
                if href.startswith("http") and href not in seen:
                    seen.add(href)
                    unique_links.append({"text": a.get_text(" ", strip=True)[:80] or "link", "url": href})
            data["links"] = unique_links[:40]

        return json.dumps(data, indent=2)

    except PermissionError as e:
        return f"{ERR} {e}"
    except Exception as e:
        return f"{ERR} browser_agent({url}): {type(e).__name__}: {str(e)[:150]}"


