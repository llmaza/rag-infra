# app/core/embeddings.py
from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    # Good default: 384-dim vectors
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # CPU for now. Later: .to("cuda") after NVIDIA toolkit setup
    return SentenceTransformer(model_name, device="cpu")