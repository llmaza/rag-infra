````md
## Phase 1 — Vector Search (RAG Retrieval)

This service ingests PDF documents, converts them into text chunks, embeds each chunk into a vector, and stores them in **Qdrant**.  
Then you can query by meaning (semantic search) to retrieve the most relevant chunks.

### Services
- **api**: FastAPI app (`http://localhost:8000`)
- **qdrant**: vector DB (`http://localhost:6333`)

### Run
```bash
docker compose up -d
````

### Health / Stats

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/stats
```

### Ingest PDFs

Put PDFs into `./data/raw/` (mounted into the container as `/app/data/raw`).

Ingest all PDFs:

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"subdir":"raw"}'
```

Ingest one PDF:

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"subdir":"raw","path":"flashattention.pdf"}'
```

Recreate the collection (dev reset):

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"subdir":"raw","recreate":true}'
```

Delete chunks for one PDF and re-ingest it:

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"subdir":"raw","delete_source":"flashattention.pdf","path":"flashattention.pdf"}'
```

### Semantic Query (Top-K chunks)

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is PagedAttention in vLLM?", "top_k":5}'
```

Filter by source PDF:

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is attention?", "top_k":5, "source":"attention_is_all_you_need.pdf"}'
```

### Notes

* Qdrant stores **points**: `{id, vector(384-dim), payload{source,page,chunk_idx,text}}`
* Embeddings model: `sentence-transformers/all-MiniLM-L6-v2` (384 dims, cosine similarity)

```
```


### Download demo papers (arXiv)
```bash
./scripts/download_demo_papers.sh