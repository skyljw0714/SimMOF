
import os
import re
import sys
import pickle
import time
from typing import List, Dict, Optional, Tuple

import faiss
from sentence_transformers import SentenceTransformer

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.dirname(_BASE_DIR)

TXT_DIR    = os.getenv("SIMMOF_SI_TXT_DIR", os.path.join(_DATA_DIR, "data", "elsevier_SI", "files_txt"))
OUT_BASE   = os.getenv("SIMMOF_SI_VECTOR_DB_DIR", os.path.join(_DATA_DIR, "vector_db_SI"))
MODEL_NAME = os.getenv("SIMMOF_SI_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

FILE_SEP_RE  = re.compile(r'^=== .+ ===$', re.MULTILINE)
TABLE_RE     = re.compile(r'\[TABLE\].*?\[/TABLE\]', re.DOTALL)
SENT_RE      = re.compile(r'(?<=[.!?])\s+')



def _sentence_pack(text: str, max_chars: int) -> List[str]:
    sents = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    chunks, cur = [], []
    cur_len = 0
    for s in sents:
        if cur and cur_len + len(s) + 1 > max_chars:
            chunks.append(' '.join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(' '.join(cur))
    return chunks


def chunk_si_text(text: str, max_chars: int = 700, min_chars: int = 200) -> List[str]:
    final_chunks: List[str] = []
    carry = ""

    def flush_carry():
        nonlocal carry
        if carry:
            final_chunks.append(carry)
            carry = ""

    def emit(block: str):
        nonlocal carry
        block = block.strip()
        if not block:
            return
        merged = (carry + "\n" + block).strip() if carry else block
        if len(merged) >= min_chars:
            final_chunks.append(merged)
            carry = ""
        else:
            carry = merged

    def emit_text(raw: str):
        raw = raw.strip()
        if not raw:
            return
        packed = _sentence_pack(raw, max_chars)
        for i, chunk in enumerate(packed):
            if i < len(packed) - 1:
                nonlocal carry
                merged = (carry + " " + chunk).strip() if carry else chunk
                final_chunks.append(merged)
                carry = ""
            else:
                emit(chunk)

    def _flush_tab_rows(rows: List[str]):
        if not rows:
            return
        header = rows[0]
        cur = [header]
        for row in rows[1:]:
            candidate = '\n'.join(cur + [row])
            if len(candidate) > max_chars and len(cur) > 1:
                emit('\n'.join(cur))
                cur = [header, row]
            else:
                cur.append(row)
        emit('\n'.join(cur))

    def emit_table_block(block: str):
        block = block.strip()
        if len(block) <= max_chars:
            emit(block)
            return
        lines = block.splitlines()
        hdr = '\n'.join(lines[:2])
        data = [l for l in lines[2:] if l.strip() not in ('[/TABLE]', '')]
        cur_rows: List[str] = []
        for row in data:
            candidate = hdr + '\n' + '\n'.join(cur_rows + [row]) + '\n[/TABLE]'
            if cur_rows and len(candidate) > max_chars:
                emit(hdr + '\n' + '\n'.join(cur_rows) + '\n[/TABLE]')
                cur_rows = [row]
            else:
                cur_rows.append(row)
        if cur_rows:
            emit(hdr + '\n' + '\n'.join(cur_rows) + '\n[/TABLE]')

    file_parts = FILE_SEP_RE.split(text)

    for file_part in file_parts:
        flush_carry()

        segments = TABLE_RE.split(file_part)
        tables   = TABLE_RE.findall(file_part)

        for seg_idx, seg in enumerate(segments):
            lines = seg.splitlines()
            text_buf:  List[str] = []
            tab_rows:  List[str] = []

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if text_buf:
                        emit_text(' '.join(text_buf))
                        text_buf = []
                    if tab_rows:
                        _flush_tab_rows(tab_rows)
                        tab_rows = []
                    continue

                if '\t' in stripped:
                    if text_buf:
                        emit_text(' '.join(text_buf))
                        text_buf = []
                    tab_rows.append(stripped)
                else:
                    if tab_rows:
                        _flush_tab_rows(tab_rows)
                        tab_rows = []
                    text_buf.append(stripped)

            if text_buf:
                emit_text(' '.join(text_buf))
            if tab_rows:
                _flush_tab_rows(tab_rows)

            if seg_idx < len(tables):
                emit_table_block(tables[seg_idx])

    flush_carry()
    return final_chunks



def is_valid_document(text: str) -> bool:
    if len(text) < 500:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    return alpha_ratio > 0.25



def encode_and_add(
    model: SentenceTransformer,
    index: Optional[faiss.Index],
    texts: List[str],
    meta_rows: List[Dict],
    all_meta: List[Dict],
) -> faiss.Index:
    if not texts:
        return index
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=512,
    ).astype("float32")
    if index is None:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    all_meta.extend(meta_rows)
    return index



def build(file_batch: int = 200, encode_batch: int = 512):
    model_dir  = os.path.join(OUT_BASE, MODEL_NAME.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    index_path = os.path.join(model_dir, "index.faiss")
    meta_path  = os.path.join(model_dir, "metadata.pkl")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        print("기존 인덱스 발견 — 이어서 처리합니다.")
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            all_meta: List[Dict] = pickle.load(f)
        already_done = {m["filename"] for m in all_meta}
        print(f"  완료: {len(already_done)}파일, 청크: {len(all_meta)}")
    else:
        index, all_meta, already_done = None, [], set()

    filenames = sorted([f for f in os.listdir(TXT_DIR) if f.endswith(".txt")])
    remaining = [f for f in filenames if f not in already_done]
    print(f"처리할 파일: {len(remaining)}개 / 전체 {len(filenames)}개")

    model = SentenceTransformer(MODEL_NAME)

    pending_texts: List[str] = []
    pending_meta:  List[Dict] = []
    processed = skipped = 0
    t0 = time.time()

    def flush():
        nonlocal index, pending_texts, pending_meta
        index = encode_and_add(model, index, pending_texts, pending_meta, all_meta)
        pending_texts, pending_meta = [], []

    for i, fname in enumerate(remaining):
        fpath = os.path.join(TXT_DIR, fname)
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
        except Exception as e:
            print(f"[읽기 오류] {fname}: {e}")
            skipped += 1
            continue

        if not is_valid_document(content):
            skipped += 1
            continue

        chunks = chunk_si_text(content)
        if not chunks:
            skipped += 1
            continue

        for cid, ch in enumerate(chunks):
            ch = ch.strip()
            if not ch:
                continue
            pending_texts.append(ch)
            pending_meta.append({"filename": fname, "chunk_id": cid, "text": ch})
            if len(pending_texts) >= encode_batch:
                flush()

        processed += 1

        if (i + 1) % file_batch == 0:
            flush()
            faiss.write_index(index, index_path)
            with open(meta_path, "wb") as f:
                pickle.dump(all_meta, f)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (len(remaining) - i - 1) / rate
            print(
                f"[{i+1}/{len(remaining)}] 처리={processed} skip={skipped} "
                f"청크={len(all_meta)} 경과={elapsed/60:.1f}m ETA={eta/60:.1f}m",
                flush=True,
            )

    flush()

    if index is None:
        print("임베딩된 청크가 없습니다.")
        return

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(all_meta, f)

    elapsed = time.time() - t0
    print(f"\n완료! 처리={processed} skip={skipped} 총청크={len(all_meta)} 시간={elapsed/60:.1f}분")
    print(f"저장: {model_dir}")


if __name__ == "__main__":
    build()
