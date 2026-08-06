from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


DEFAULT_CORPUS = (
    Path(__file__).resolve().parent
    / "lammps_error_knowledge"
    / "lammps_official_error_docs_chunks.jsonl"
)
DEFAULT_SOURCE_CORPUS = (
    Path(__file__).resolve().parent
    / "lammps_error_knowledge"
    / "lammps_3mar2020_source_chunks.jsonl"
)


class LAMMPSErrorKnowledgeBase:

    def __init__(self, corpus_path: Union[str, Path] = DEFAULT_CORPUS):
        self.corpus_path = Path(corpus_path)
        self.entries = self._load(self.corpus_path)
        if self.corpus_path.resolve() == DEFAULT_CORPUS.resolve():
            self.entries.extend(self._load(DEFAULT_SOURCE_CORPUS))

    @staticmethod
    def _load(path: Path) -> List[Dict[str, Any]]:
        if not path.is_file():
            return []
        entries: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stop = {"the", "and", "for", "with", "from", "this", "that", "file", "error"}
        return {
            tok.lower()
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9_+\-.]{2,}", text or "")
            if tok.lower() not in stop
        }

    @staticmethod
    def _entry_text(entry: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in ("error_key", "heading", "symptoms", "text", "risk", "source"):
            val = entry.get(key)
            if val:
                parts.append(str(val))
        for key in ("patterns", "suggested_actions", "files_to_edit", "section_id"):
            val = entry.get(key)
            if isinstance(val, list):
                parts.extend(str(x) for x in val)
            elif val:
                parts.append(str(val))
        return "\n".join(parts)

    def search(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        q_tokens = self._tokens(query)
        query_lower = (query or "").lower()
        scored: List[tuple[float, Dict[str, Any]]] = []

        for entry in self.entries:
            haystack = self._entry_text(entry)
            hay_lower = haystack.lower()
            score = 0.0

            for pattern in entry.get("patterns", []) or []:
                p = str(pattern).lower()
                if p and p in query_lower:
                    score += 12.0
                elif p and any(piece in query_lower for piece in p.split()):
                    score += 2.0
                p_tokens = self._tokens(p.replace("...", " "))
                if p_tokens and p_tokens <= q_tokens:
                    score += 8.0

            key = str(entry.get("error_key", "")).lower()
            heading = str(entry.get("heading", "")).lower()
            if key and key in query_lower:
                score += 10.0
            if heading and heading in query_lower:
                score += 8.0
            h_tokens = self._tokens(re.sub(r"^\d+(?:\.\d+)*\.\s*", "", heading).replace("...", " "))
            if h_tokens and q_tokens:
                heading_overlap = len(h_tokens & q_tokens)
                if heading_overlap:
                    score += 1.5 * heading_overlap
                if h_tokens <= q_tokens:
                    score += 8.0

            e_tokens = self._tokens(hay_lower)
            if q_tokens and e_tokens:
                overlap = len(q_tokens & e_tokens)
                score += overlap / max(len(q_tokens), 1)
                score += 0.2 * overlap

            if score > 0:
                item = dict(entry)
                item["score"] = round(score, 4)
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def format_hits(self, hits: Iterable[Dict[str, Any]], *, max_chars: int = 4000) -> str:
        blocks: List[str] = []
        for i, hit in enumerate(hits, start=1):
            patterns = hit.get("patterns") or []
            actions = hit.get("suggested_actions") or []
            files = hit.get("files_to_edit") or []
            block = (
                f"[{i}] source={hit.get('source')} heading={hit.get('heading')} "
                f"score={hit.get('score')}\n"
                f"url: {hit.get('url', '')}\n"
                f"patterns: {', '.join(str(x) for x in patterns)}\n"
                f"symptoms: {hit.get('symptoms', '')}\n"
                f"suggested_actions:\n"
                + "\n".join(f"- {a}" for a in actions)
                + f"\nfiles_to_edit: {', '.join(str(x) for x in files)}\n"
                f"risk: {hit.get('risk', '')}\n"
                f"note: {hit.get('text', '')}"
            )
            blocks.append(block)

        text = "\n\n".join(blocks).strip()
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "\n...[truncated]"
        return text
