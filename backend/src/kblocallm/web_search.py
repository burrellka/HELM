"""Web search for the model advisor. Ported from KAIROS tools/web_search.py.

Provided by the KAIROS dev as `docs/web-search-port-for-helm.md` — this is
their §2 module, adapted to HELM's one rule that differs: nothing reads
os.environ outside config.py, so the key is passed in explicitly rather than
sourced from the environment here (config.Settings provides it).

Gemini Google-Search grounding (default) runs a real Google Search server-side
and returns a synthesized, source-cited answer — exactly what a model advisor
wants, and grounded rather than recalled, so it fits the advisor's core rule.
Tavily is the alternative. Only needs httpx.

Failure is STRUCTURED, never raised (KAIROS §6): the advisor sees
`ok=False, error=...` and reasons about it, and the honest-degradation rule
holds — search down means "search unavailable", never a silent empty result
the model might fill from memory.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("kblocallm.web_search")

TAVILY_URL = "https://api.tavily.com/search"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


@dataclass
class SearchResult:
    ok: bool
    answer: str = ""                        # synthesized grounded answer
    sources: list[dict[str, str]] = field(default_factory=list)  # {title,url}
    error: str | None = None

    def as_text(self) -> str:
        """Compact, LLM-readable form to hand the advisor model."""
        if not self.ok:
            return f"[web search failed: {self.error}]"
        lines: list[str] = []
        if self.answer:
            lines.append(self.answer)
        if self.sources:
            lines.append("\nSources:")
            for i, s in enumerate(self.sources, 1):
                lines.append(f"{i}. {s.get('title', '')} — {s.get('url', '')}")
        return "\n".join(lines).strip()

    def as_dict(self) -> dict:
        return {"ok": self.ok, "answer": self.answer, "sources": self.sources,
                "error": self.error}


async def web_search(
    query: str,
    *,
    api_key: str,
    provider: str = "gemini",               # "gemini" | "tavily"
    model: str = "gemini-2.5-flash",        # gemini grounding only
    max_results: int = 5,
) -> SearchResult:
    query = (query or "").strip()
    if not query:
        return SearchResult(ok=False, error="missing_query")
    if not api_key:
        return SearchResult(ok=False, error="no_search_key")
    max_results = max(1, min(10, int(max_results or 5)))

    if provider == "tavily":
        return await _tavily(query, max_results, api_key)
    return await _gemini_grounding(query, max_results, api_key, model)


async def _gemini_grounding(query, max_results, api_key, model) -> SearchResult:
    model = (model or "gemini-2.5-flash").strip()
    url = f"{GEMINI_BASE}/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": query}]}],
        "tools": [{"google_search": {}}],
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body)
            if resp.status_code in (401, 403):
                return SearchResult(ok=False, error="unauthorized (check the web search key)")
            if resp.status_code == 429:
                return SearchResult(ok=False, error="quota exhausted (1500/day free on gemini-2.5-flash)")
            if resp.status_code == 400:
                # KAIROS §6: a model that does not support google_search 400s.
                return SearchResult(ok=False, error=(
                    f"http_400: the model may not support Google Search grounding "
                    f"(use gemini-2.5-flash). {resp.text[:150]}"))
            if resp.status_code >= 400:
                return SearchResult(ok=False, error=f"http_{resp.status_code}: {resp.text[:200]}")
            data = resp.json()
    except httpx.HTTPError as e:
        return SearchResult(ok=False, error=f"network: {e}")

    cands = data.get("candidates") or []
    if not cands:
        return SearchResult(ok=False, error="empty")
    parts = (cands[0].get("content") or {}).get("parts") or []
    answer = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    gm = cands[0].get("groundingMetadata") or {}
    sources: list[dict[str, str]] = []
    for ch in (gm.get("groundingChunks") or [])[:max_results]:
        web = (ch or {}).get("web") or {}
        title, uri = (web.get("title") or "").strip(), (web.get("uri") or "").strip()
        if title or uri:
            sources.append({"title": title or uri, "url": uri})
    if not answer and not sources:
        return SearchResult(ok=False, error="empty")
    return SearchResult(ok=True, answer=answer, sources=sources)


async def _tavily(query, max_results, api_key) -> SearchResult:
    body = {"api_key": api_key, "query": query, "max_results": max_results,
            "search_depth": "basic", "include_answer": True}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(TAVILY_URL, json=body)
            if resp.status_code >= 400:
                return SearchResult(ok=False, error=f"tavily_http_{resp.status_code}")
            data = resp.json()
    except httpx.HTTPError as e:
        return SearchResult(ok=False, error=f"tavily_network: {e}")
    sources = [{"title": r.get("title", ""), "url": r.get("url", "")}
               for r in (data.get("results") or [])]
    return SearchResult(ok=True, answer=(data.get("answer") or "").strip(), sources=sources)
