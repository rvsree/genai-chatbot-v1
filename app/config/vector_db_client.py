# app/service/vector_db_client.py
# Unified client for BOTH indexing (CRUD) and search with circuit breaker + retries; async-first.

from typing import Dict, Any, List, Optional
from uuid import uuid4
from app.config.app_config import AppConfigSingleton
from app.utils.app_logging import get_logger
from app.utils.circuit_breaker import CircuitBreaker, with_retries_async

_cfg = AppConfigSingleton.instance()
_logger = get_logger(_cfg)

class _EmbeddingService:
    def __init__(self):
        from chromadb.utils import embedding_functions
        self._ef = embedding_functions.DefaultEmbeddingFunction()
    def embed_one(self, text: str) -> List[float]:
        return self._ef([text])[0]
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        return self._ef(texts)

class VectorBackend:
    # CRUD
    def upsert_items(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None: raise NotImplementedError
    def get_ids_by_parent(self, parent_id: str) -> List[str]: raise NotImplementedError
    def delete_by_parent(self, parent_id: str) -> int: raise NotImplementedError
    def delete(self, doc_id: str) -> int: raise NotImplementedError
    def get(self, doc_id: str) -> Dict[str, Any]: raise NotImplementedError
    def save_metadata(self, doc_id: str, patch: Dict[str, Any]) -> Dict[str, Any]: raise NotImplementedError
    def count(self) -> int: raise NotImplementedError
    # Search
    def query_by_text(self, query_text: str, n_results: int, where: Optional[Dict[str, Any]]): raise NotImplementedError
    def query_by_vector(self, query_vector: List[float], n_results: int, where: Optional[Dict[str, Any]]): raise NotImplementedError

class _ChromaBackend(VectorBackend):
    def __init__(self, collection_name: str = "documents_collection"):
        import chromadb
        from chromadb.utils import embedding_functions
        self.client = chromadb.PersistentClient(path=_cfg.chroma_dir)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.ef, metadata={"hnsw:space": "cosine"}
        )
        _logger.info("[VectorDBClient.Chroma] collection=%s path=%s", collection_name, _cfg.chroma_dir)

    def _normalize_where(self, filt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filt: return None
        if any(isinstance(v, dict) and any(k.startswith("$") for k in v.keys()) for v in filt.values()):
            return filt
        items = [{k: {"$eq": v}} for k, v in filt.items()]
        return items[0] if len(items) == 1 else {"$and": items}

    # CRUD
    def upsert_items(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        self.collection.upsert(documents=texts, metadatas=metadatas, ids=ids)
    def get_ids_by_parent(self, parent_id: str) -> List[str]:
        res = self.collection.get(where={"parent_id": {"$eq": parent_id}})
        ids = res.get("ids", [])
        if isinstance(ids, list) and ids and isinstance(ids[0], list): ids = ids[0]
        return ids or []
    def delete_by_parent(self, parent_id: str) -> int:
        ids = self.get_ids_by_parent(parent_id)
        if not ids: return 0
        self.collection.delete(ids=ids); return len(ids)
    def delete(self, doc_id: str) -> int:
        self.collection.delete(ids=[doc_id]); return 1
    def get(self, doc_id: str) -> Dict[str, Any]:
        return self.collection.get(ids=[doc_id])
    def save_metadata(self, doc_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get(doc_id)
        if not current.get("ids"): raise ValueError("Document not found")
        text = current["documents"][0]; meta = (current["metadatas"][0] or {}); meta.update(patch or {})
        self.collection.upsert(documents=[text], metadatas=[meta], ids=[doc_id]); return meta
    def count(self) -> int:
        return int(self.collection.count())

    # Search
    def query_by_text(self, query_text: str, n_results: int, where: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return self.collection.query(query_texts=[query_text], n_results=n_results, where=self._normalize_where(where))
    def query_by_vector(self, query_vector: List[float], n_results: int, where: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return self.collection.query(query_embeddings=[query_vector], n_results=n_results, where=self._normalize_where(where))

class _PGVectorBackend(VectorBackend):
    def __init__(self): pass

class _OpenSearchBackend(VectorBackend):
    def __init__(self): pass

_vector_breaker = CircuitBreaker(failure_threshold=3, recovery_time_sec=20.0)

def _is_retryable_vector(err: Exception) -> bool:
    msg = str(err).lower()
    if "invalid api key" in msg or "authentication" in msg:
        return False
    if "expected where to have exactly one operator" in msg:
        return False
    return True

class VectorDBClient:
    def __init__(self, backend: str = "chroma", collection_name: str = "documents_collection"):
        b = (backend or "chroma").lower()
        if b == "chroma":
            self._backend: VectorBackend = _ChromaBackend(collection_name=collection_name)
        elif b == "pgvector":
            self._backend = _PGVectorBackend()
        elif b == "opensearch":
            self._backend = _OpenSearchBackend()
        else:
            raise ValueError(f"Unsupported vector backend: {backend}")
        self._backend_name = b
        self._embed = _EmbeddingService()
        self._cache: Dict[str, List[float]] = {}
        _logger.info("[VectorDBClient] backend=%s ready", self._backend_name)

    # Utilities
    def health(self) -> Dict[str, Any]: return {"status": "ok", "backend": self._backend_name}
    def next_id(self) -> str: return str(uuid4())

    # CRUD (sync-safe for indexers/routers calling from request thread)
    def upsert_items(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None: self._backend.upsert_items(texts, metadatas, ids)
    def get_ids_by_parent(self, parent_id: str) -> List[str]: return self._backend.get_ids_by_parent(parent_id)
    def delete_by_parent(self, parent_id: str) -> int: return self._backend.delete_by_parent(parent_id)
    def delete(self, doc_id: str) -> int: return self._backend.delete(doc_id)
    def get(self, doc_id: str) -> Dict[str, Any]: return self._backend.get(doc_id)
    def save_metadata(self, doc_id: str, patch: Dict[str, Any]) -> Dict[str, Any]: return self._backend.save_metadata(doc_id, patch)
    def count(self) -> int: return self._backend.count()

    # Embeddings
    def get_query_embedding(self, query: str) -> List[float]:
        if query in self._cache: return self._cache[query]
        vec = self._embed.embed_one(query); self._cache[query] = vec; return vec

    # Async search; callers must await
    async def search_async(self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        vec = self.get_query_embedding(query)
        async def _op():
            return self._backend.query_by_vector(query_vector=vec, n_results=top_k, where=where)
        res = await with_retries_async(_op, _is_retryable_vector, _vector_breaker, max_attempts=3, base_backoff=0.5)
        ids = res.get("ids"); docs = res.get("documents"); metas = res.get("metadatas")
        if ids is None or docs is None or metas is None:
            ids = [res.get("ids", [])]; docs = [res.get("documents", [])]; metas = [res.get("metadatas", [])]
        return {"ids": ids, "documents": docs, "metadatas": metas}

    # Provide a familiar name; still async; always await this in async contexts
    async def search(self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self.search_async(query, top_k, where)

    # Optional: text path (sync-friendly), used in scripts/tests
    def search_text(self, query_text: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = self._backend.query_by_text(query_text=query_text, n_results=top_k, where=where)
        ids = res.get("ids"); docs = res.get("documents"); metas = res.get("metadatas")
        if ids is None or docs is None or metas is None:
            ids = [res.get("ids", [])]; docs = [res.get("documents", [])]; metas = [res.get("metadatas", [])]
        return {"ids": ids, "documents": docs, "metadatas": metas}
