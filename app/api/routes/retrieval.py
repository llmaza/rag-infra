from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from time import perf_counter

from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from app.core.config import settings
from app.core.qdrant import get_qdrant
from app.core.embeddings import get_embedder

router = APIRouter(tags=["Retrieval"])


# ---------- Schemas (Swagger will look clean) ----------

class QueryRequest(BaseModel):
    query: str = Field(..., examples=["What is paged attention in vLLM?"])
    top_k: int = Field(5, ge=1, le=20, examples=[5])
    source: Optional[str] = Field(None, description="Optional source filter (e.g. PDF filename).", examples=["flashattention.pdf"])


class TimingMS(BaseModel):
    embed_ms: float
    search_ms: float
    total_ms: float


class RetrievedChunk(BaseModel):
    id: str
    score: float
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_idx: Optional[int] = None
    text_preview: str
    text: str


class QueryResponse(BaseModel):
    query: str
    top_k: int
    results: List[RetrievedChunk]
    timings_ms: TimingMS


# ---------- Endpoint ----------

@router.post(
    "/query",
    summary="Retrieve top-k chunks",
    description=(
        "Embeds the query, searches Qdrant, returns top-k chunks with scores and timing breakdown.\n\n"
        "- `embed_ms`: embedding time\n"
        "- `search_ms`: Qdrant search time\n"
        "- `total_ms`: end-to-end time"
    ),
    response_model=QueryResponse,
)
def query(req: QueryRequest) -> QueryResponse:
    qdrant = get_qdrant()
    embedder = get_embedder()

    t_total0 = perf_counter()

    # 1) embed timing
    t0 = perf_counter()
    qvec = embedder.encode([req.query], normalize_embeddings=True)[0].tolist()
    embed_ms = (perf_counter() - t0) * 1000.0

    # build filter
    qfilter = None
    if req.source:
        qfilter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=req.source))]
        )

    # 2) search timing
    t1 = perf_counter()
    hits = qdrant.search(
        collection_name=settings.collection_name,
        query_vector=qvec,
        limit=req.top_k,
        query_filter=qfilter,
        with_payload=True,
    )
    search_ms = (perf_counter() - t1) * 1000.0

    # format results
    results: List[RetrievedChunk] = []
    for h in hits:
        p = h.payload or {}
        text = p.get("text", "")
        preview = (text[:240] + "...") if len(text) > 240 else text

        results.append(
            RetrievedChunk(
                id=str(h.id),
                score=float(h.score),
                source=p.get("source"),
                page=p.get("page"),
                chunk_idx=p.get("chunk_idx"),
                text_preview=preview,
                text=text,
            )
        )

    total_ms = (perf_counter() - t_total0) * 1000.0

    return QueryResponse(
        query=req.query,
        top_k=req.top_k,
        results=results,
        timings_ms=TimingMS(
            embed_ms=round(embed_ms, 3),
            search_ms=round(search_ms, 3),
            total_ms=round(total_ms, 3),
        ),
    )