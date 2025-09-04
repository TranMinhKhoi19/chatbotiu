# Python 3.10
import os
from typing import List, Tuple
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

VSTORE = os.getenv("VECTORSTORE_DIR", "vectorstore")
COLL   = os.getenv("VECTOR_COLL", "iu_docs")

_emb = SentenceTransformer("BAAI/bge-m3")
_db  = PersistentClient(path=VSTORE).get_or_create_collection(COLL)

def retrieve(query: str, k: int = 6) -> List[Tuple[str, dict]]:
    try:
        qv = _emb.encode([query], normalize_embeddings=True).tolist()
        hits = _db.query(query_embeddings=qv, n_results=k)
        docs = hits.get("documents", [[]])[0]
        metas = hits.get("metadatas", [[]])[0]
        return list(zip(docs, metas))
    except Exception:
        return []
