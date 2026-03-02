from qdrant_client import QdrantClient
from .config import settings

def get_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)