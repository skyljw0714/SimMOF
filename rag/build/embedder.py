import os
from typing import List, Dict, Optional

import faiss
import pickle
from sentence_transformers import SentenceTransformer

from parser import chunk_text
from utils import sanitize_model_name
from config import PARSED_TEXT_DIR, VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME, VERBOSE


def save_faiss_index(index, path: str):
    faiss.write_index(index, os.path.join(path, "index.faiss"))


def save_metadata(metadata: List[Dict], path: str):
    with open(os.path.join(path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)


def is_valid_document(text: str) -> bool:
    if len(text) < 1000:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    return alpha_ratio > 0.5


def _encode_and_add(
    model: SentenceTransformer,
    index: Optional[faiss.Index],
    texts: List[str],
    metadata_rows: List[Dict],
    all_metadata: List[Dict],
    *,
    show_progress: bool = True,
):
    if not texts:
        return index

    embeddings = model.encode(texts, show_progress_bar=show_progress, normalize_embeddings=True).astype("float32")

    if index is None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    all_metadata.extend(metadata_rows)
    return index


def build_vector_store_batched(
    file_batch_size: int = 200,
    encode_batch_size: int = 512,
):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model_dir_name = sanitize_model_name(EMBEDDING_MODEL_NAME)
    model_vector_dir = os.path.join(VECTOR_STORE_DIR, model_dir_name)

    filenames = sorted([f for f in os.listdir(PARSED_TEXT_DIR) if f.endswith(".txt")])

    index = None
    all_metadata: List[Dict] = []
    pending_texts: List[str] = []
    pending_meta: List[Dict] = []

    for i in range(0, len(filenames), file_batch_size):
        batch_files = filenames[i : i + file_batch_size]

        for fname in batch_files:
            fpath = os.path.join(PARSED_TEXT_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                if VERBOSE:
                    print(f"Skipping empty file: {fname}")
                continue

            if not is_valid_document(content):
                if VERBOSE:
                    print(f"Skipping low-content file: {fname}")
                continue

            chunks = chunk_text(content)
            if not chunks:
                if VERBOSE:
                    print(f"No chunks produced: {fname}")
                continue

            for cid, ch in enumerate(chunks):
                ch = ch.strip()
                if not ch:
                    continue

                pending_texts.append(ch)
                pending_meta.append({"filename": fname, "chunk_id": cid, "text": ch})

                if len(pending_texts) >= encode_batch_size:
                    index = _encode_and_add(model, index, pending_texts, pending_meta, all_metadata, show_progress=True)
                    pending_texts = []
                    pending_meta = []

        if pending_texts:
            index = _encode_and_add(model, index, pending_texts, pending_meta, all_metadata, show_progress=True)
            pending_texts = []
            pending_meta = []

    if index is None:
        raise RuntimeError("No valid chunks were embedded; index is None.")

    os.makedirs(model_vector_dir, exist_ok=True)
    save_faiss_index(index, model_vector_dir)
    save_metadata(all_metadata, model_vector_dir)

    if VERBOSE:
        print(f"Saved FAISS index to: {os.path.join(model_vector_dir, 'index.faiss')}")
        print(f"Saved metadata to: {os.path.join(model_vector_dir, 'metadata.pkl')}")
        print(f"Total embedded chunks: {len(all_metadata)}")


if __name__ == "__main__":
    build_vector_store_batched(file_batch_size=200, encode_batch_size=512)
