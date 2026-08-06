from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


DEFAULT_CORPUS = Path(__file__).resolve().parent / "vasp_error_knowledge"


class VASPErrorKnowledgeBase:

    def __init__(self, corpus_path: Union[str, Path] = DEFAULT_CORPUS):
        self.corpus_path = Path(corpus_path)
        self.entries = self._load(self.corpus_path)

    @staticmethod
    def _load(path: Path) -> List[Dict[str, Any]]:
        if path.is_dir():
            entries: List[Dict[str, Any]] = []
            for corpus in sorted(path.glob("*.jsonl")):
                entries.extend(VASPErrorKnowledgeBase._load(corpus))
            return entries
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
        return {
            tok.lower()
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9_+\-.]{2,}", text or "")
            if tok.lower() not in {"the", "and", "for", "with", "from", "this", "that"}
        }

    @staticmethod
    def _entry_text(entry: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in (
            "error_key",
            "symptoms",
            "text",
            "risk",
            "source",
            "component",
        ):
            val = entry.get(key)
            if val:
                parts.append(str(val))
        for key in ("patterns", "candidate_actions", "files_to_edit", "files_to_handle"):
            val = entry.get(key)
            if isinstance(val, list):
                parts.extend(str(x) for x in val)
        return "\n".join(parts)

    def search(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        q_tokens = self._tokens(query)
        scored: List[tuple[float, Dict[str, Any]]] = []
        query_lower = (query or "").lower()

        for entry in self.entries:
            haystack = self._entry_text(entry)
            hay_lower = haystack.lower()
            score = 0.0

            for pattern in entry.get("patterns", []) or []:
                p = str(pattern).lower()
                if p and p in query_lower:
                    specificity = min(30.0, 3.0 * max(len(self._tokens(p)), 1))
                    score += 12.0 + specificity
                elif p and p in hay_lower and any(piece in query_lower for piece in p.split()):
                    score += 2.0

            key = str(entry.get("error_key", "")).lower()
            if key and key in query_lower:
                score += 10.0

            e_tokens = self._tokens(haystack)
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
            actions = hit.get("candidate_actions") or []
            patterns = hit.get("patterns") or []
            files = hit.get("files_to_edit") or []
            block = (
                f"[{i}] source={hit.get('source')} component={hit.get('component')} "
                f"error_key={hit.get('error_key')} score={hit.get('score')}\n"
                f"patterns: {', '.join(str(x) for x in patterns)}\n"
                f"symptoms: {hit.get('symptoms', '')}\n"
                f"candidate_actions:\n"
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
