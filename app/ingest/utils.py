from __future__ import annotations

import hashlib
from typing import List
from qdrant_client.http.models import VectorParams, Distance


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    """
    Super simple chunker by characters.
    Phase 1 goal: working pipeline > perfect chunking.
    Later: token-based chunking.
    """
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, chunk_size - chunk_overlap)
    return chunks


def stable_point_id(source: str, page: int, chunk_idx: int, chunk: str) -> int:
    """
    Create stable numeric IDs so re-ingest doesn't create duplicates.
    Qdrant supports int IDs.
    """
    s = f"{source}|p{page}|c{chunk_idx}|{chunk[:100]}"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    # fit into signed 64-bit range
    return int(h[:16], 16)


def ensure_collection(qdrant, name: str, dim: int = 384):
    cols = [x.name for x in qdrant.get_collections().collections]
    if name not in cols:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )