import os, re, json, hashlib
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from pypdf import PdfReader

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
OUT_PATH = Path(os.getenv("CHUNKS_OUT", "data/processed/chunks.jsonl"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))      # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150")) # characters


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


def chunk_id(source: str, idx: int, chunk: str) -> str:
    h = hashlib.sha1((source + f"#{idx}#" + chunk).encode("utf-8", errors="ignore")).hexdigest()
    return h


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in RAW_DIR.rglob("*") if p.is_file()])
    if not files:
        print(f"No files found in {RAW_DIR}. Put .txt or .pdf there.")
        return

    written = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for path in tqdm(files, desc="ingest"):
            ext = path.suffix.lower()
            if ext not in [".txt", ".pdf"]:
                continue

            raw = read_pdf(path) if ext == ".pdf" else read_txt(path)
            text = clean(raw)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

            source = str(path.relative_to(RAW_DIR))
            for i, ch in enumerate(chunks):
                rec: Dict = {
                    "id": chunk_id(source, i, ch),
                    "source": source,
                    "chunk_index": i,
                    "text": ch,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()