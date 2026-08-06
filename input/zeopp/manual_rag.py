from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[2]
ZEOPP_EXAMPLE_HTML = (
    REPO_ROOT
    / "input"
    / "manual_rag_corpus"
    / "zeopp"
    / "zeopp_examples_original.html"
)
ZEOPP_EXAMPLE_EMBED_MODEL = os.getenv(
    "SIMMOF_ZEOPP_EXAMPLE_RAG_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
ZEOPP_HEADING_MAX_SECTION_CHARS = int(os.getenv("SIMMOF_ZEOPP_HEADING_RAG_MAX_SECTION_CHARS", "3500"))


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@lru_cache(maxsize=1)
def _load_heading_sections(html_path: str = str(ZEOPP_EXAMPLE_HTML)) -> Tuple[Dict[str, Any], ...]:
    path = Path(html_path)
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    headings = [_clean_text(h.get_text(" ", strip=True)) for h in soup.find_all("h2")]
    headings = [h for h in headings if h]

    raw_lines = soup.get_text("\n", strip=True).splitlines()
    lines = [_clean_text(line) for line in raw_lines]
    lines = [line for line in lines if line]

    heading_positions: List[Tuple[int, str]] = []
    search_start = 0
    for heading in headings:
        for pos in range(search_start, len(lines)):
            if lines[pos] == heading:
                heading_positions.append((pos, heading))
                search_start = pos + 1
                break

    sections: List[Dict[str, Any]] = []
    for idx, (start, title) in enumerate(heading_positions):
        end = heading_positions[idx + 1][0] if idx + 1 < len(heading_positions) else len(lines)
        text = _clean_text("\n".join(lines[start:end]))
        if not text:
            continue
        sections.append(
            {
                "section_id": idx,
                "heading": title,
                "filename": path.name,
                "chunk_id": f"heading::{idx}",
                "text": text[:ZEOPP_HEADING_MAX_SECTION_CHARS],
                "char_count": len(text),
            }
        )
    return tuple(sections)


@lru_cache(maxsize=1)
def _load_heading_index(
    html_path: str = str(ZEOPP_EXAMPLE_HTML),
    embed_model_name: str = ZEOPP_EXAMPLE_EMBED_MODEL,
) -> Tuple[SentenceTransformer, faiss.Index, Tuple[Dict[str, Any], ...]]:
    sections = _load_heading_sections(html_path)
    if not sections:
        raise RuntimeError(f"No Zeo++ heading sections parsed from {html_path}")
    embedder = SentenceTransformer(embed_model_name)
    texts = [f"{s['heading']}\n{s['text']}" for s in sections]
    embeddings = embedder.encode(texts, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embedder, index, sections


class ZeoppExampleRAG:

    def __init__(
        self,
        html_path: str | Path = ZEOPP_EXAMPLE_HTML,
        embed_model_name: str = ZEOPP_EXAMPLE_EMBED_MODEL,
    ):
        self.html_path = Path(html_path)
        self.embedder, self.index, self.meta = _load_heading_index(str(self.html_path), embed_model_name)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q_emb, min(top_k, len(self.meta)))
        hits: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            item = dict(self.meta[int(idx)])
            item["score"] = float(score)
            hits.append(item)
        return hits


def retrieve_zeopp_example_hints(
    query: str,
    *,
    top_k: int = 3,
    max_chars_per_hit: int = 3500,
) -> Dict[str, Any]:
    retriever = ZeoppExampleRAG()
    hits = retriever.search(query, top_k=top_k)

    blocks: List[str] = []
    for rank, hit in enumerate(hits, start=1):
        text = " ".join(str(hit.get("text") or "").split()).strip()
        if len(text) > max_chars_per_hit:
            text = text[: max_chars_per_hit - 14].rstrip() + "\n[TRUNCATED]"
        blocks.append(
            "\n".join(
                [
                    f"[Retrieved Zeo++ heading section {rank}]",
                    "source=zeopp_examples_original.html",
                    f"heading={hit.get('heading')}",
                    f"chunk_id={hit.get('chunk_id')}",
                    f"score={float(hit.get('score') or 0.0):.4f}",
                    "text:",
                    text,
                ]
            )
        )

    formatted = "\n\n".join(
        [
            "RAG evidence from the Zeo++ official examples/documentation page.",
            "Use exact documented flags and command syntax when present.",
            "Do not replace a documented flag with a semantically similar flag.",
            "If an example command appears, preserve all flags exactly, including combined flags such as '-ha -psd', '-ha -block', '-gridG', and '-r -nt2'.",
            "",
            "\n\n".join(blocks),
        ]
    ).strip()

    return {
        "query": query,
        "hits": hits,
        "formatted_hints": formatted if hits else "",
    }


__all__ = ["ZeoppExampleRAG", "retrieve_zeopp_example_hints"]
