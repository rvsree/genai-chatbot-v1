# app/service/chroma_db_client.py
# Fix: never pass {} to collection.query/query_embeddings; pass None instead.

from typing import List, Dict, Any, Optional
from uuid import uuid4
import chromadb
from chromadb.utils import embedding_functions
from app.config.app_config import AppConfigSingleton
from app.utils.app_logging import get_logger

_cfg = AppConfigSingleton.instance()
_logger = get_logger(_cfg)

class ChromaDBClient:
    def __init__(self, collection_name: str = "documents_collection"):
        self.client = chromadb.PersistentClient(path=_cfg.chroma_dir)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        _logger.info("[ChromaDBClient] Ready collection=%s path=%s", collection_name, _cfg.chroma_dir)

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "chroma_dir": _cfg.chroma_dir}

    def count(self) -> int:
        return int(self.collection.count())

    def next_id(self) -> str:
        return str(uuid4())

    def upsert_items(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        _logger.info("[ChromaDBClient] Upsert items n=%d", len(ids))
        self.collection.upsert(documents=texts, metadatas=metadatas, ids=ids)
        _logger.info("[ChromaDBClient] Upsert complete n=%d", len(ids))

    def get_ids_by_parent(self, parent_id: str) -> List[str]:
        res = self.collection.get(where={"parent_id": {"$eq": parent_id}})
        ids = res.get("ids", [])
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return ids or []

    def delete_by_parent(self, parent_id: str) -> int:
        ids = self.get_ids_by_parent(parent_id)
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    def delete(self, doc_id: str) -> int:
        self.collection.delete(ids=[doc_id])
        return 1

    def get(self, doc_id: str) -> Dict[str, Any]:
        return self.collection.get(ids=[doc_id])

    def save_metadata(self, doc_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get(doc_id)
        if not current.get("ids"):
            raise ValueError("Document not found")
        text = current["documents"][0]
        meta = (current["metadatas"][0] or {})
        meta.update(patch or {})
        self.collection.upsert(documents=[text], metadatas=[meta], ids=[doc_id])
        return meta

    def _normalize_where(self, filt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filt:
            return None
        # if already normalized with operators, trust caller
        if any(isinstance(v, dict) and any(k.startswith("$") for k in v.keys()) for v in filt.values()):
            return filt
        items = [{k: {"$eq": v}} for k, v in filt.items()]
        return items[0] if len(items) == 1 else {"$and": items}

    def query(self, query_text: str, n_results: int = 8, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not query_text or not query_text.strip():
            raise ValueError("query_text is required")
        norm_where = self._normalize_where(where)
        return self.collection.query(query_texts=[query_text], n_results=n_results, where=norm_where)

    def query_by_vector(self, query_vector: List[float], n_results: int = 8, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        norm_where = self._normalize_where(where)
        return self.collection.query(query_embeddings=[query_vector], n_results=n_results, where=norm_where)
