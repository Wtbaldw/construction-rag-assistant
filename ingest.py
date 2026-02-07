import os
import hashlib
from typing import List, Dict, Tuple

import chromadb
from chromadb.config import Settings
import ollama

# ✅ MATCH YOUR STRUCTURE
DATA_DIR = "ragData"          # <-- folder shown in your screenshot
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "construction_sc"

CHUNK_SIZE = 900       # characters (simple starter chunking)
CHUNK_OVERLAP = 150
BATCH = 64

# Extensions we will include (and also accept files with no extension)
ALLOWED_EXTS = {".txt", ".md", ".text", ""}

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max(chunk_size - overlap, 1)
    return chunks

def stable_id(rel_path: str, chunk_index: int, chunk: str) -> str:
    h = hashlib.sha1(f"{rel_path}:{chunk_index}:{chunk}".encode("utf-8")).hexdigest()
    return h

def embed(texts: List[str]) -> List[List[float]]:
    out = ollama.embed(model="nomic-embed-text", input=texts)
    return out["embeddings"]

def discover_files(root: str) -> List[Tuple[str, str]]:
    """
    Returns list of (abs_path, rel_path) for allowed files under root.
    """
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            abs_path = os.path.join(dirpath, fn)
            # ignore junk
            if fn.lower() in {".ds_store", "desktop.ini"}:
                continue

            _, ext = os.path.splitext(fn)
            if ext.lower() not in ALLOWED_EXTS:
                # If file has an extension we don't recognize, skip it
                continue

            rel_path = os.path.relpath(abs_path, root)
            out.append((abs_path, rel_path))
    return sorted(out, key=lambda x: x[1])

def category_from_relpath(rel_path: str) -> str:
    # First folder becomes the category label (Services, South Carolina, Universal)
    parts = rel_path.split(os.sep)
    return parts[0] if parts else "Unknown"

def main():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"❌ Can't find DATA_DIR='{DATA_DIR}'. Run from the folder that contains '{DATA_DIR}'.")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    files = discover_files(DATA_DIR)
    if not files:
        raise SystemExit(f"❌ No ingestible files found in {DATA_DIR}")

    all_docs: List[str] = []
    all_ids: List[str] = []
    all_metas: List[Dict] = []

    for abs_path, rel_path in files:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        if not text:
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        category = category_from_relpath(rel_path)

        for i, ch in enumerate(chunks):
            doc_id = stable_id(rel_path, i, ch)
            all_docs.append(ch)
            all_ids.append(doc_id)
            all_metas.append({
                "source": rel_path,     # keeps folder + filename
                "category": category,   # Services / South Carolina / Universal
                "chunk_index": i
            })

    if not all_docs:
        raise SystemExit("❌ Found files, but they were empty or produced no chunks.")

    # Re-run safe: delete IDs first if they already exist
    try:
        collection.delete(ids=all_ids)
    except Exception:
        pass

    # Add in batches
    for i in range(0, len(all_docs), BATCH):
        batch_docs = all_docs[i:i+BATCH]
        batch_ids = all_ids[i:i+BATCH]
        batch_metas = all_metas[i:i+BATCH]
        batch_embeds = embed(batch_docs)

        collection.add(
            documents=batch_docs,
            embeddings=batch_embeds,
            ids=batch_ids,
            metadatas=batch_metas
        )

    print(f"✅ Ingest complete: {len(all_docs)} chunks")
    print(f"📦 Stored in: {CHROMA_DIR} | Collection: {COLLECTION_NAME}")
    print(f"📁 Ingested from: {DATA_DIR} (including subfolders)")

if __name__ == "__main__":
    main()
