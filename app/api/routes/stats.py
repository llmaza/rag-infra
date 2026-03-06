from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.qdrant import get_qdrant

router = APIRouter(tags=["Stats"])


class StatsResponse(BaseModel):
    status: str
    collection: str
    points_count: int
    indexed_vectors_count: int
    vector_size: int
    distance: str
    segments_count: int


@router.get(
    "/stats",
    summary="Qdrant collection stats",
    description="Returns collection metadata and counts from Qdrant for the active collection.",
    response_model=StatsResponse,
)
def stats() -> StatsResponse:
    qdrant = get_qdrant()
    try:
        info = qdrant.get_collection(settings.collection_name)
        vectors = info.config.params.vectors

        return StatsResponse(
            status="ok",
            collection=settings.collection_name,
            points_count=info.points_count,
            indexed_vectors_count=info.indexed_vectors_count,
            vector_size=vectors.size,
            distance=str(vectors.distance),
            segments_count=info.segments_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))