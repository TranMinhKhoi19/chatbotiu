# Python 3.10
import os, json
os.environ.setdefault("CHROMADB_TELEMETRY", "False")

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

KB_PATH = "data/kb_ics.json"
VSTORE = os.getenv("VECTORSTORE_DIR", "vectorstore")
COLL   = os.getenv("VECTOR_COLL", "iu_docs")

def rows_from_kb(kb: dict):
    rows = []
    for major, fields in kb.items():
        for key, text in fields.items():
            if not text: 
                continue
            rows.append({
                "text": f"[{major.upper()}] {key}: {text}",
                "meta": {"major": major, "field": key, "source": "kb_ics"}
            })
    return rows

if __name__ == "__main__":
    if not os.path.exists(KB_PATH):
        raise FileNotFoundError(f"Missing {KB_PATH}")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    rows = rows_from_kb(kb)
    if not rows:
        raise SystemExit("KB empty")

    os.makedirs(VSTORE, exist_ok=True)
    client = PersistentClient(path=VSTORE)
    db = client.get_or_create_collection(COLL)

    model = SentenceTransformer("BAAI/bge-m3")
    texts = [r["text"] for r in rows]
    metas = [r["meta"] for r in rows]
    embs  = model.encode(texts, normalize_embeddings=True).tolist()

    ids = [f"kb-{i}" for i in range(len(texts))]
    try:
        db.delete(ids=ids)
    except Exception:
        pass

    db.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)
    print(f"Indexed {len(texts)} chunks → {VSTORE} / {COLL}")
