"""
Document loader — fetches content from a URL or accepts raw text.

Supports:
  - Plain text (passed directly as a string)
  - HTTP/HTTPS URLs (fetched with requests, HTML stripped to plain text)

Returns a LangChain Document so it plugs directly into the splitter.
"""

import re
import requests
from langchain_core.documents import Document


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    """
    Very lightweight HTML → plain text conversion.
    Removes script/style blocks and all remaining tags,
    then collapses whitespace so the splitter gets clean text.
    """
    # Drop script and style blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse runs of whitespace / blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ─── Public API ───────────────────────────────────────────────────────────────

def load_from_text(text: str) -> Document:
    """Wrap raw text in a LangChain Document."""
    if not text or not text.strip():
        raise ValueError("Uploaded text is empty.")
    return Document(page_content=text.strip(), metadata={"source": "text_upload"})


def load_from_url(url: str) -> Document:
    """
    Fetch a URL and convert its content to plain text.

    Raises:
        ValueError: if the URL is unreachable or returns non-200 status
        RuntimeError: if the fetched content is empty after stripping
    """
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "EzeeChatBot/1.0 (+knowledge-indexer)"},
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch URL '{url}': {e}")

    content_type = response.headers.get("Content-Type", "")
    raw = response.text

    # Strip HTML if the response is a webpage
    if "text/html" in content_type:
        raw = _strip_html(raw)

    if not raw.strip():
        raise RuntimeError(f"No usable content found at URL '{url}'.")

    return Document(page_content=raw, metadata={"source": url})
