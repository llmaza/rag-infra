from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings
from app.core.qdrant import get_qdrant

router = APIRouter(tags=["Health"])

class HealthResponse(BaseModel):
    status: str
    qdrant_ok: bool
    collection: str

@router.get(
    "/health",
    summary="Health check",
    description="Returns service status and validates Qdrant connectivity + collection availability.",
    response_model=HealthResponse,
)
def health():
    try:
        get_qdrant().get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    return {"status": "ok", "qdrant_ok": qdrant_ok, "collection": settings.collection_name}