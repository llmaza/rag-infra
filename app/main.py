from fastapi import FastAPI

app = FastAPI(title="RAG Service")

@app.get("/health")
def health():
    return {"status": "ok"}