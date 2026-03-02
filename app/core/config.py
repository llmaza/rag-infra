import os
from pydantic import BaseModel

class Settings(BaseModel):
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    collection_name: str = os.getenv("COLLECTION_NAME", "rag_chunks")

settings = Settings()