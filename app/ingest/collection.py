from __future__ import annotations

import hashlib
from typing import List
from qdrant_client.http.models import VectorParams, Distance


def ensure_collection(qdrant, name: str, dim: int = 384):
    cols = [x.name for x in qdrant.get_collections().collections]
    if name not in cols:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )