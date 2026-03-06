from fastapi import FastAPI
from app.api.routes.health import router as health_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.stats import router as stats_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.ui import router as ui_router
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import hashlib

from pypdf import PdfReader
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue

from app.core.config import settings
from app.core.qdrant import get_qdrant
from app.core.embeddings import get_embedder

from qdrant_client.http.models import VectorParams, Distance
from fastapi import HTTPException
from time import perf_counter

def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Service",
        version="0.2.0",
        swagger_ui_parameters={
            "docExpansion": "none",
            "defaultModelsExpandDepth": 2,
            "displayRequestDuration": True,
            "filter": True,
        },
    )
    app.include_router(health_router)
    app.include_router(retrieval_router)
    app.include_router(stats_router)
    app.include_router(ingest_router)
    app.include_router(ui_router)


    return app

app = create_app()

DATA_ROOT = Path("/app/data")



class IngestRequest(BaseModel):
    subdir: str = "raw"
    path: Optional[str] = None  # if None -> ingest all PDFs in subdir
    chunk_size: int = 900
    chunk_overlap: int = 150
    batch_size: int = 64

    recreate: bool = False               # drop+create collection
    delete_source: Optional[str] = None  # delete all points for this pdf before ingest


# @app.post("/ingest")
# def ingest(req: IngestRequest):
#     qdrant = get_qdrant()
#     embedder = get_embedder()
#     DATA_DIR = DATA_ROOT / req.subdir

#     # Ensure collection exists (and optionally recreate)
#     if req.recreate:
#         cols = [x.name for x in qdrant.get_collections().collections]
#         if settings.collection_name in cols:
#             qdrant.delete_collection(settings.collection_name)
#         ensure_collection(qdrant, settings.collection_name, dim=384)
#     else:
#         ensure_collection(qdrant, settings.collection_name, dim=384)

#     # Optionally delete all points from a specific source before ingesting
#     if req.delete_source:
#         qdrant.delete(
#             collection_name=settings.collection_name,
#             points_selector=Filter(
#                 must=[FieldCondition(key="source", match=MatchValue(value=req.delete_source))]
#             ),
#         )

#     # Choose PDFs to ingest
#     if req.path:
#         pdf_paths = [DATA_DIR / req.path]
#     else:
#         pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

#     if not pdf_paths:
#         return {
#             "status": "ok",
#             "ingested_files": 0,
#             "points_upserted": 0,
#             "detail": f"No PDFs found in {DATA_DIR}",
#         }
    
#     points_total = 0
#     files_done = 0

#     for pdf_path in pdf_paths:
#         if not pdf_path.exists():
#             continue

#         reader = PdfReader(str(pdf_path))
#         file_points: List[PointStruct] = []

#         for page_idx, page in enumerate(reader.pages):
#             text = page.extract_text() or ""
#             chunks = chunk_text(text, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)

#             if not chunks:
#                 continue

#             vectors = embedder.encode(chunks, normalize_embeddings=True).tolist()

#             for chunk_idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
#                 pid = stable_point_id(str(pdf_path.name), page_idx, chunk_idx, chunk)
#                 payload = {
#                     "source": pdf_path.name,
#                     "page": page_idx,
#                     "chunk_idx": chunk_idx,
#                     "text": chunk,
#                 }
#                 file_points.append(PointStruct(id=pid, vector=vec, payload=payload))

#         # Upsert in batches
#         for i in range(0, len(file_points), req.batch_size):
#             batch = file_points[i : i + req.batch_size]
#             if batch:
#                 qdrant.upsert(collection_name=settings.collection_name, points=batch)
#                 points_total += len(batch)

#         files_done += 1

#     return {"status": "ok", "ingested_files": files_done, "points_upserted": points_total}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    source: Optional[str] = None  # optional filter by pdf filename

