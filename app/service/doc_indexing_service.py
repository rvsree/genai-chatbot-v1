from typing import List, Dict, Any, Optional
from uuid import uuid4
import chromadb
from chromadb.utils import embedding_functions
from app.config.app_config import AppConfig
from app.utils.app_logging import get_logger

cfg = AppConfig()
logger = get_logger(cfg)
ef = embedding_functions.DefaultEmbeddingFunction()

class IndexingService:
    def __init__(self, collection_name: str = "documents_collection"):
        logger.info("[IndexingService] Trying to connect with Chroma DB")
        self.client = chromadb.PersistentClient(path=cfg.chroma_dir)
        logger.info("[IndexingService] Successfully connected to Chroma DB")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("[IndexingService] Ready collection=%s path=%s", collection_name, cfg.chroma_dir)

    def count(self) -> int:
        try:
            c = int(self.collection.count())
            logger.info("[IndexingService] Count=%d", c)
            return c
        except Exception as e:
            logger.exception("[IndexingService] Count failed: %s", e)
            raise

    def next_id(self) -> str:
        return str(uuid4())

    def upsert_items(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        logger.info("[IndexingService] Upsert items: n=%d", len(ids))
        self.collection.upsert(documents=texts, metadatas=metadatas, ids=ids)
        logger.info("[IndexingService] Upsert complete: n=%d", len(ids))

    def get_ids_by_parent(self, parent_id: str) -> List[str]:
        logger.info("[IndexingService] Lookup by parent_id=%s", parent_id)
        res = self.collection.get(where={"parent_id": {"$eq": parent_id}})
        ids = res.get("ids", [])
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        logger.info("[IndexingService] Found %d ids for parent=%s", len(ids or []), parent_id)
        return ids or []

    def delete_by_parent(self, parent_id: str) -> int:
        ids = self.get_ids_by_parent(parent_id)
        if not ids:
            logger.info("[IndexingService] No chunks to delete for parent=%s", parent_id)
            return 0
        logger.info("[IndexingService] Deleting %d ids for parent=%s", len(ids), parent_id)
        self.collection.delete(ids=ids)
        logger.info("[IndexingService] Deleted %d ids for parent=%s", len(ids), parent_id)
        return len(ids)

    def _normalize_where(self, filt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filt:
            return None
        items = [{k: {"$eq": v}} for k, v in filt.items()]
        return items[0] if len(items) == 1 else {"$and": items}

    def query(self, query_text: str, n_results: int = 8, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not query_text or not query_text.strip():
            raise ValueError("query_text is required")
        norm_where = self._normalize_where(where)
        logger.info("[IndexingService] Trying query='%s' n=%d where=%s", query_text, n_results, norm_where)
        try:
            res = self.collection.query(query_texts=[query_text], n_results=n_results, where=norm_where)
            logger.info("[IndexingService] Query ok sets=%d", len(res.get("ids", [])))
            return res
        except Exception as e:
            logger.exception("[IndexingService] Query failed: %s", e)
            raise

    def delete(self, doc_id: str) -> int:
        logger.info("[IndexingService] Delete id=%s", doc_id)
        self.collection.delete(ids=[doc_id])
        logger.info("[IndexingService] Delete ok id=%s", doc_id)
        return 1

    def get(self, doc_id: str) -> Dict[str, Any]:
        logger.info("[IndexingService] Get id=%s", doc_id)
        res = self.collection.get(ids=[doc_id])
        logger.info("[IndexingService] Get ok id=%s", doc_id)
        return res

    def save_metadata(self, doc_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[IndexingService] Save metadata id=%s keys=%s", doc_id, list(patch.keys()))
        current = self.get(doc_id)
        if not current.get("ids"):
            logger.error("[IndexingService] Save metadata failed: id not found %s", doc_id)
            raise ValueError("Document not found")
        text = current["documents"][0]
        meta = (current["metadatas"][0] or {})
        meta.update(patch or {})
        self.collection.upsert(documents=[text], metadatas=[meta], ids=[doc_id])
        logger.info("[IndexingService] Save metadata ok id=%s", doc_id)
        return meta
