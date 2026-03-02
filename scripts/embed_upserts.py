import os, json
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "rag_chunks")
IN_PATH = Path(os.getenv("CHUNKS_IN", "data/processed/chunks.jsonl"))

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH = int(os.getenv("EMBED_BATCH", "64"))

def main():
    assert IN_PATH.exists(), f"Missing {IN_PATH}. Run ingest_docs.py first."

    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(MODEL_NAME)

    dim = model.get_sentence_embedding_dimension()

    # (re)create collection if missing
    cols = client.get_collections().collections
    names = {c.name for c in cols}
    if COLLECTION not in names:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    # read chunks
    rows = []
    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # embed + upsert
    points = []
    for i in tqdm(range(0, len(rows), BATCH), desc="embed"):
        batch = rows[i:i+BATCH]
        texts = [r["text"] for r in batch]
        vecs = model.encode(texts, normalize_embeddings=True)

        for r, v in zip(batch, vecs):
            points.append(PointStruct(
                id=r["id"],
                vector=v.tolist(),
                payload={
                    "text": r["text"],
                    "source": r["source"],
                    "chunk_index": r["chunk_index"],
                }
            ))

        # flush in chunks to avoid huge RAM
        if len(points) >= 1000:
            client.upsert(collection_name=COLLECTION, points=points)
            points = []

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    print(f"Upserted {len(rows)} vectors into '{COLLECTION}' (dim={dim})")

if __name__ == "__main__":
    main()