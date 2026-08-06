from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "input" / "manual_rag_corpus" / "raspa"
KEYWORD_PATH = DATASET_DIR / "raspa_manual_simulation_keywords.v1.jsonl"
SECTION_PATH = DATASET_DIR / "raspa_manual_sections.v1.jsonl"
EXAMPLE_PATH = DATASET_DIR / "raspa_manual_examples.v1.jsonl"


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 1]


def _score(query_tokens: List[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokens(text)
    if not text_tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for tok in text_tokens:
        counts[tok] = counts.get(tok, 0) + 1
    score = 0.0
    for tok in query_tokens:
        tf = counts.get(tok, 0)
        if tf:
            score += 1.0 + math.log(tf)
    return score / math.sqrt(len(text_tokens))


def _compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _rank(
    rows: Iterable[Dict[str, Any]],
    query: str,
    fields: Tuple[str, ...],
    top_k: int,
    *,
    boost_keywords: Iterable[str] = (),
) -> List[Dict[str, Any]]:
    qtok = _tokens(query)
    boosted = {str(x).lower() for x in boost_keywords}
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in rows:
        haystack = " ".join(str(row.get(field, "")) for field in fields)
        score = _score(qtok, haystack)
        row_keyword = str(row.get("keyword", "")).lower()
        if row_keyword and row_keyword in boosted:
            score += 5.0
        row_keywords = {str(x).lower() for x in row.get("keywords", []) if x}
        if row_keywords & boosted:
            score += min(5.0, 1.25 * len(row_keywords & boosted))
        if score > 0:
            out = dict(row)
            out["score"] = round(score, 6)
            scored.append((score, out))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]]


def load_raspa_manual_store() -> Dict[str, Any]:
    return {
        "keywords": _load_jsonl(KEYWORD_PATH),
        "sections": _load_jsonl(SECTION_PATH),
        "examples": _load_jsonl(EXAMPLE_PATH),
    }


def retrieve_raspa_manual_hints(
    query: str,
    *,
    top_keywords: int = 10,
    top_sections: int = 4,
    top_examples: int = 3,
    max_chars_per_hit: int = 700,
) -> Dict[str, Any]:
    store = load_raspa_manual_store()

    keyword_hits = _rank(
        store["keywords"],
        query,
        fields=("keyword", "group", "raw_text"),
        top_k=top_keywords,
    )
    section_hits = _rank(
        store["sections"],
        query,
        fields=("section_title", "keywords", "raw_text"),
        top_k=top_sections,
    )
    example_hits = _rank(
        store["examples"],
        query,
        fields=("example_title", "keywords", "raw_text"),
        top_k=top_examples,
    )

    lines: List[str] = []
    if keyword_hits:
        lines.append("[RASPA manual keyword evidence]")
        for hit in keyword_hits:
            loc = f"manual section {hit.get('section_number')} p.{hit.get('page_start')}"
            lines.append(
                f"- {hit.get('keyword')} [{hit.get('group')}; {loc}]: "
                f"{_compact(hit.get('raw_text', ''), max_chars_per_hit)}"
            )

    if example_hits:
        lines.append("\n[RASPA manual example evidence]")
        for hit in example_hits:
            lines.append(
                f"- Example {hit.get('example_number')}: {hit.get('example_title')} "
                f"(p.{hit.get('page_start')}-{hit.get('page_end')}): "
                f"{_compact(hit.get('raw_text', ''), max_chars_per_hit)}"
            )

    if section_hits:
        lines.append("\n[RASPA manual section evidence]")
        for hit in section_hits:
            lines.append(
                f"- {hit.get('section_number')} {hit.get('section_title')} "
                f"(p.{hit.get('page_start')}-{hit.get('page_end')}): "
                f"{_compact(hit.get('raw_text', ''), max_chars_per_hit)}"
            )

    return {
        "query": query,
        "keyword_hits": keyword_hits,
        "example_hits": example_hits,
        "section_hits": section_hits,
        "formatted_hints": "\n".join(lines).strip(),
    }


__all__ = ["load_raspa_manual_store", "retrieve_raspa_manual_hints"]
