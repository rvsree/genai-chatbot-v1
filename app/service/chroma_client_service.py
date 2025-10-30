from typing import List, Dict, Any, Optional
from uuid import uuid4
import chromadb
from chromadb.utils import embedding_functions
from app.config.app_config import AppConfig
from app.utils.app_logging import get_logger

cfg = AppConfig()
logger = get_logger(cfg)

class ChromaClientService:
    """
    Single-source Chroma access used across the app.
    Provides upsert_items(), get_ids_by_parent(), delete_by_parent(), query(), etc.
    """

    def __init__(self, collection_name: str = "documents_collection"):
        logger.info("[ChromaClientService] Connecting to Chroma path=%s", cfg.chroma_dir)
        self.client = chromadb.PersistentClient(path=cfg.chroma_dir)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("[ChromaClientService] Ready collection=%s", collection_name)

    # ---------- Basic utilities ----------
    def count(self) -> int:
        return int(self.collection.count())

    def next_id(self) -> str:
        return str(uuid4())

    # ---------- Upsert/bulk ----------
    def upsert_items(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        logger.info("[ChromaClientService] Upsert items n=%d", len(ids))
        self.collection.upsert(documents=texts, metadatas=metadatas, ids=ids)
        logger.info("[ChromaClientService] Upsert complete n=%d", len(ids))

    # ---------- Parent helpers (idempotency and cleanup) ----------
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

    # ---------- Single-id ops ----------
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

    # ---------- Query ----------
    def _normalize_where(self, filt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filt:
            return None
        items = [{k: {"$eq": v}} for k, v in filt.items()]
        return items[0] if len(items) == 1 else {"$and": items}

    def query(self, query_text: str, n_results: int = 8, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not query_text or not query_text.strip():
            raise ValueError("query_text is required")
        norm_where = self._normalize_where(where)
        return self.collection.query(query_texts=[query_text], n_results=n_results, where=norm_where)
