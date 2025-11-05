# app/service/chroma_client_service.py
# If this file still exists in imports elsewhere, replace its internals to delegate to the new clients
# without changing its public surface. This avoids refactors in other modules.
from typing import List, Dict, Any, Optional
from uuid import uuid4
from app.config.app_config import AppConfigSingleton
from app.utils.app_logging import get_logger
from app.config.chroma_db_client import ChromaDBClient
from app.config.vector_db_client import VectorDBClient

_cfg = AppConfigSingleton.instance()
_logger = get_logger(_cfg)

class ChromaClientService:
    def __init__(self, collection_name: str = "documents_collection", backend: str = "chroma"):
        self._chroma = ChromaDBClient(collection_name=collection_name)
        self._vector = VectorDBClient(backend=backend)

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "backend": "vector+chroma"}

    # Backward-compatible methods preserved:
    def count(self) -> int: return self._chroma.count()
    def next_id(self) -> str: return str(uuid4())
    def upsert_items(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        self._chroma.upsert_items(texts, metadatas, ids)
    def get_ids_by_parent(self, parent_id: str) -> List[str]:
        return self._chroma.get_ids_by_parent(parent_id)
    def delete_by_parent(self, parent_id: str) -> int:
        return self._chroma.delete_by_parent(parent_id)
    def delete(self, doc_id: str) -> int:
        return self._chroma.delete(doc_id)
    def get(self, doc_id: str) -> Dict[str, Any]:
        return self._chroma.get(doc_id)
    def save_metadata(self, doc_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        return self._chroma.save_metadata(doc_id, patch)

    # Legacy text path
    def query(self, query_text: str, n_results: int = 8, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._chroma.query(query_text=query_text, n_results=n_results, where=where)

    # New vector-agnostic path
    def query_with_reusable_embedding(self, query_text: str, n_results: int = 8, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._vector.search(query=query_text, top_k=n_results, where=where or {})
