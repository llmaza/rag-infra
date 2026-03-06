import os
from pathlib import Path
from pydantic import BaseModel, Field

class Settings(BaseModel):
    qdrant_url: str = Field(default_factory=lambda: os.getenv("QDRANT_URL", "http://qdrant:6333"))
    collection_name: str = Field(default_factory=lambda: os.getenv("COLLECTION_NAME", "rag_chunks"))

    data_root: Path = Field(default_factory=lambda: Path(os.getenv("DATA_ROOT", "/app/data")))

settings = Settings()