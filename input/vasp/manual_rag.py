from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "input" / "manual_rag_corpus" / "vasp"
MANUAL_TAG_EVIDENCE_PATH = DATASET_DIR / "vasp_manual_incar_tag_evidence.v1.jsonl"

MAX_RECORDS = 8
MAX_CHARS_PER_RECORD = 1200

STOP_TERMS = {
    "and",
    "are",
    "for",
    "from",
    "incar",
    "manual",
    "mof",
    "tag",
    "tags",
    "the",
    "this",
    "vasp",
    "with",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _terms(text: str) -> set[str]:
    normalized = re.sub(r"[_/-]+", " ", (text or "").lower())
    return {
        term
        for term in re.findall(r"[a-z0-9]+", normalized)
        if len(term) >= 3 and term not in STOP_TERMS
    }


def _phrases(text: str) -> set[str]:
    words = [
        w
        for w in re.findall(r"[a-z0-9]+", re.sub(r"[_/-]+", " ", (text or "").lower()))
        if w not in STOP_TERMS
    ]
    return {" ".join(pair) for pair in zip(words, words[1:])}


def load_vasp_manual_evidence() -> List[Dict[str, Any]]:
    rows = _load_jsonl(MANUAL_TAG_EVIDENCE_PATH)
    for row in rows:
        row["_search_text"] = " ".join(
            str(x)
            for x in [
                row.get("tag"),
                row.get("section_title"),
                row.get("retrieval_text"),
                row.get("raw_text"),
            ]
            if x
        ).lower()
    return rows


def retrieve_vasp_manual_hints(
    query: str,
    *,
    top_k: int = MAX_RECORDS,
    max_chars_per_hit: int = MAX_CHARS_PER_RECORD,
) -> Dict[str, Any]:
    rows = load_vasp_manual_evidence()
    q_terms = _terms(query)
    q_phrases = _phrases(query)

    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        raw = row.get("_search_text", "")
        section_title = (row.get("section_title") or "").lower()
        tag = str(row.get("tag") or "").lower()
        score = 0

        if tag and tag in q_terms:
            score += 30
        for term in q_terms:
            if term in section_title:
                score += 5
            elif term in raw:
                score += 1
        for phrase in q_phrases:
            if phrase in section_title:
                score += 15
            elif phrase in raw:
                score += 4

        if score > 0:
            scored.append((score, -idx, row))

    scored.sort(reverse=True)
    hits = [row for _, _, row in scored[:top_k]]

    blocks: List[str] = []
    for rank, hit in enumerate(hits, start=1):
        text = (hit.get("raw_text") or "").strip()
        if len(text) > max_chars_per_hit:
            text = text[: max_chars_per_hit - 14].rstrip() + "\n[TRUNCATED]"
        blocks.append(
            "\n".join(
                [
                    f"[Retrieved VASP manual tag evidence {rank}]",
                    "source=vaspmanual.pdf",
                    f"tag={hit.get('tag')}",
                    f"section={hit.get('section_number')} {hit.get('section_title')}",
                    "text:",
                    text,
                ]
            )
        )

    formatted = "\n\n".join(
        [
            "RAG evidence from VASP manual Section 6 (INCAR File).",
            "Use exact INCAR tags and conservative values when the retrieved manual evidence is applicable.",
            "Do not add unrelated advanced tags unless the task explicitly asks for that calculation type.",
            "",
            "\n\n".join(blocks),
        ]
    ).strip()

    return {
        "query": query,
        "hits": hits,
        "formatted_hints": formatted if hits else "",
    }


def _clean_json_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_reranker_json(text: str) -> Dict[str, Any]:
    text = _clean_json_text(text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"ranked_record_ids": [], "rationale": text[:1000]}


def build_vasp_manual_reranker_messages(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    max_candidate_chars: int = 900,
) -> Tuple[List[Any], Dict[str, Dict[str, Any]]]:
    from langchain.schema import HumanMessage, SystemMessage

    candidate_by_id: Dict[str, Dict[str, Any]] = {}
    candidate_lines: List[str] = []
    for idx, row in enumerate(candidates, start=1):
        record_id = f"V{idx}"
        row = dict(row)
        row["record_id"] = record_id
        candidate_by_id[record_id] = row
        text = re.sub(r"\s+", " ", row.get("raw_text") or "").strip()
        if len(text) > max_candidate_chars:
            text = text[: max_candidate_chars - 14].rstrip() + " [TRUNCATED]"
        candidate_lines.extend(
            [
                f"[{record_id}] tag={row.get('tag')} section={row.get('section_number')} {row.get('section_title')}",
                text,
                "",
            ]
        )

    system = (
        "You rerank VASP INCAR manual evidence for input generation.\n"
        "Do not invent tags. Do not use hidden expected answers.\n"
        "Return strict JSON only."
    )
    human = "\n".join(
        [
            f"USER_QUERY:\n{query}",
            "",
            "CANDIDATE_VASP_MANUAL_EVIDENCE:",
            "\n".join(candidate_lines)[:24000],
            "",
            "Return JSON with this schema:",
            "{",
            '  "ranked_record_ids": ["V3", "V1", "V7"],',
            '  "support_record_ids": ["V2"],',
            '  "rationale": "Brief reason for the ordering."',
            "}",
            "",
            "Rules:",
            "- Prioritize records that activate the requested calculation mode or property.",
            "- Keep related setup tags after the main activation tags.",
            "- Rerank; do not reject candidates because the downstream prompt will receive a candidate tag index.",
        ]
    )
    return [
        SystemMessage(content=system),
        HumanMessage(content=human),
    ], candidate_by_id


def retrieve_vasp_manual_hints_reranked(
    query: str,
    *,
    llm: Optional[Any] = None,
    top_k: int = MAX_RECORDS,
    candidate_k: int = 24,
    max_chars_per_hit: int = MAX_CHARS_PER_RECORD,
    max_candidate_chars: int = 900,
) -> Dict[str, Any]:
    base = retrieve_vasp_manual_hints(
        query,
        top_k=max(candidate_k, top_k),
        max_chars_per_hit=max_candidate_chars,
    )
    candidates = list(base.get("hits") or [])
    if not candidates or llm is None:
        return retrieve_vasp_manual_hints(query, top_k=top_k, max_chars_per_hit=max_chars_per_hit)

    messages, candidate_by_id = build_vasp_manual_reranker_messages(
        query,
        candidates,
        max_candidate_chars=max_candidate_chars,
    )

    try:
        resp = llm.invoke(messages)
        selection = _parse_reranker_json(getattr(resp, "content", "") or "")
    except Exception as exc:
        out = retrieve_vasp_manual_hints(query, top_k=top_k, max_chars_per_hit=max_chars_per_hit)
        out["reranker_error"] = repr(exc)
        return out

    ranked_ids: List[str] = []
    for key in list(selection.get("ranked_record_ids", [])) + list(selection.get("support_record_ids", [])):
        key = str(key)
        if key in candidate_by_id and key not in ranked_ids:
            ranked_ids.append(key)
    for key in candidate_by_id:
        if key not in ranked_ids:
            ranked_ids.append(key)

    reranked_hits = [candidate_by_id[key] for key in ranked_ids]
    primary_hits = reranked_hits[:top_k]

    blocks: List[str] = []
    for rank, hit in enumerate(primary_hits, start=1):
        text = (hit.get("raw_text") or "").strip()
        if len(text) > max_chars_per_hit:
            text = text[: max_chars_per_hit - 14].rstrip() + "\n[TRUNCATED]"
        blocks.append(
            "\n".join(
                [
                    f"[Reranked VASP manual tag evidence {rank}]",
                    "source=vaspmanual.pdf",
                    f"record_id={hit.get('record_id')}",
                    f"tag={hit.get('tag')}",
                    f"section={hit.get('section_number')} {hit.get('section_title')}",
                    "text:",
                    text,
                ]
            )
        )

    tag_index = ", ".join(
        f"{hit.get('record_id')}={hit.get('tag')}" for hit in reranked_hits if hit.get("tag")
    )
    formatted = "\n\n".join(
        [
            "RAG evidence from VASP manual Section 6 (INCAR File), reranked for this query.",
            "Use exact INCAR tags and conservative values when applicable.",
            "Do not ignore the candidate tag index: it preserves deterministic retrieved tags that may be needed as supporting activation/setup tags.",
            f"Candidate tag index: {tag_index}",
            "",
            "\n\n".join(blocks),
            "",
            "[Reranker rationale]",
            str(selection.get("rationale", ""))[:1200],
        ]
    ).strip()

    return {
        "query": query,
        "hits": primary_hits,
        "all_candidate_hits": reranked_hits,
        "reranker_selection": selection,
        "formatted_hints": formatted,
    }


__all__ = [
    "build_vasp_manual_reranker_messages",
    "load_vasp_manual_evidence",
    "retrieve_vasp_manual_hints",
    "retrieve_vasp_manual_hints_reranked",
]
