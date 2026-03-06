from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

from pypdf import PdfReader
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue

from app.core.config import settings
from app.core.qdrant import get_qdrant
from app.core.embeddings import get_embedder

# import these from wherever you currently have them
from app.ingest.utils import chunk_text  
from app.ingest.collection import ensure_collection  
from app.ingest.utils import stable_point_id  
from app.core.config import settings


DATA_ROOT = settings.data_root
router = APIRouter(tags=["Ingest"])

class IngestRequest(BaseModel):
    subdir: str = Field("raw", description="Subdirectory under /app/data", examples=["raw"])
    path: Optional[str] = Field(None, description="Optional single PDF filename to ingest", examples=["flashattention.pdf"])
    recreate: bool = Field(False, description="Drop and recreate collection before ingest")
    delete_source: Optional[str] = Field(None, description="Delete points for this source before ingest", examples=["flashattention.pdf"])

    chunk_size: int = Field(900, ge=100, le=4000, description="Chunk size in characters")
    chunk_overlap: int = Field(150, ge=0, le=2000, description="Chunk overlap in characters")
    batch_size: int = Field(64, ge=1, le=512, description="Upsert batch size")


class IngestResponse(BaseModel):
    status: str
    ingested_files: int
    points_upserted: int
    detail: Optional[str] = None


@router.post(
    "/ingest",
    summary="Ingest PDFs into Qdrant",
    description=(
        "Reads PDFs from /app/data/<subdir>, chunks pages, embeds chunks, and upserts vectors into Qdrant.\n\n"
        "Options:\n"
        "- `recreate`: drop+recreate collection\n"
        "- `delete_source`: delete existing points for a given PDF before ingest\n"
        "- `path`: ingest a single file (otherwise all PDFs in subdir)"
    ),
    response_model=IngestResponse,
)
def ingest(req: IngestRequest) -> IngestResponse:
    qdrant = get_qdrant()
    embedder = get_embedder()
    data_dir = DATA_ROOT / req.subdir

    # Ensure collection exists (and optionally recreate)
    if req.recreate:
        cols = [x.name for x in qdrant.get_collections().collections]
        if settings.collection_name in cols:
            qdrant.delete_collection(settings.collection_name)
        ensure_collection(qdrant, settings.collection_name, dim=384)
    else:
        ensure_collection(qdrant, settings.collection_name, dim=384)

    # Optionally delete all points from a specific source before ingesting
    if req.delete_source:
        qdrant.delete(
            collection_name=settings.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=req.delete_source))]
            ),
        )

    # Choose PDFs to ingest
    if req.path:
        pdf_paths = [data_dir / req.path]
    else:
        pdf_paths = sorted(data_dir.glob("*.pdf"))

    if not pdf_paths:
        return IngestResponse(
            status="ok",
            ingested_files=0,
            points_upserted=0,
            detail=f"No PDFs found in {data_dir}",
        )

    points_total = 0
    files_done = 0

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            continue

        reader = PdfReader(str(pdf_path))
        file_points: List[PointStruct] = []

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            chunks = chunk_text(text, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
            if not chunks:
                continue

            vectors = embedder.encode(chunks, normalize_embeddings=True).tolist()

            for chunk_idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
                pid = stable_point_id(str(pdf_path.name), page_idx, chunk_idx, chunk)
                payload = {
                    "source": pdf_path.name,
                    "page": page_idx,
                    "chunk_idx": chunk_idx,
                    "text": chunk,
                }
                file_points.append(PointStruct(id=pid, vector=vec, payload=payload))

        # Upsert in batches
        for i in range(0, len(file_points), req.batch_size):
            batch = file_points[i : i + req.batch_size]
            if batch:
                qdrant.upsert(collection_name=settings.collection_name, points=batch)
                points_total += len(batch)

        files_done += 1

    return IngestResponse(status="ok", ingested_files=files_done, points_upserted=points_total)