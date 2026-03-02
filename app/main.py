from fastapi import FastAPI
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

app = FastAPI(title="RAG Service")

DATA_ROOT = Path("/app/data")


@app.get("/health")
def health():
    qdrant_ok = False
    try:
        get_qdrant().get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    return {
        "status": "ok",
        "qdrant_ok": qdrant_ok,
        "collection": settings.collection_name,
    }

@app.get("/stats")
def stats():
    qdrant = get_qdrant()
    try:
        info = qdrant.get_collection(settings.collection_name)
        return {
            "status": "ok",
            "collection": settings.collection_name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
            "segments_count": info.segments_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    """
    Super simple chunker by characters.
    Phase 1 goal: working pipeline > perfect chunking.
    Later: token-based chunking.
    """
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, chunk_size - chunk_overlap)
    return chunks


def stable_point_id(source: str, page: int, chunk_idx: int, chunk: str) -> int:
    """
    Create stable numeric IDs so re-ingest doesn't create duplicates.
    Qdrant supports int IDs.
    """
    s = f"{source}|p{page}|c{chunk_idx}|{chunk[:100]}"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    # fit into signed 64-bit range
    return int(h[:16], 16)


class IngestRequest(BaseModel):
    subdir: str = "raw"
    path: Optional[str] = None  # if None -> ingest all PDFs in subdir
    chunk_size: int = 900
    chunk_overlap: int = 150
    batch_size: int = 64

    recreate: bool = False               # drop+create collection
    delete_source: Optional[str] = None  # delete all points for this pdf before ingest

def ensure_collection(qdrant, name: str, dim: int = 384):
    cols = [x.name for x in qdrant.get_collections().collections]
    if name not in cols:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

@app.post("/ingest")
def ingest(req: IngestRequest):
    qdrant = get_qdrant()
    embedder = get_embedder()
    DATA_DIR = DATA_ROOT / req.subdir

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
        pdf_paths = [DATA_DIR / req.path]
    else:
        pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_paths:
        return {
            "status": "ok",
            "ingested_files": 0,
            "points_upserted": 0,
            "detail": f"No PDFs found in {DATA_DIR}",
        }
    
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

    return {"status": "ok", "ingested_files": files_done, "points_upserted": points_total}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    source: Optional[str] = None  # optional filter by pdf filename


@app.post("/query")
def query(req: QueryRequest):
    qdrant = get_qdrant()
    embedder = get_embedder()

    qvec = embedder.encode([req.query], normalize_embeddings=True)[0].tolist()

    qfilter = None
    if req.source:
        qfilter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=req.source))]
        )

    hits = qdrant.search(
        collection_name=settings.collection_name,
        query_vector=qvec,
        limit=req.top_k,
        query_filter=qfilter,
        with_payload=True,
    )

    results = []
    for h in hits:
        p = h.payload or {}
        text = p.get("text", "")
        results.append(
            {
                "id": h.id,
                "score": float(h.score),
                "source": p.get("source"),
                "page": p.get("page"),
                "chunk_idx": p.get("chunk_idx"),
                "text_preview": (text[:240] + "...") if len(text) > 240 else text,
                "text": text,
            }
        )

    return {"query": req.query, "top_k": req.top_k, "results": results}