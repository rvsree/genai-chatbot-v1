# app/vector/vector_service.py
# Orchestrator: computes per-variant embeddings, caches them in-memory for the run, and queries the selected backend.

from typing import Dict, Any, List, Optional
from app.config.app_config import AppConfigSingleton
from app.utils.app_logging import get_logger
from app.vector.embedding_service import EmbeddingService
from app.vector.vector_client import VectorClient, ChromaVectorClient
from app.config.chroma_client_service import ChromaClientService

_cfg = AppConfigSingleton.instance()
_logger = get_logger(_cfg)

class VectorService:
    def __init__(self, backend: str = "chroma"):
        self._backend = backend.lower()
        self._embed = EmbeddingService()
        self._cache: Dict[str, List[float]] = {}

        if self._backend == "chroma":
            self.db_client: VectorClient = ChromaVectorClient(ChromaClientService())
        elif self._backend == "pgvector":
            from app.vector.vector_client import PGVectorClient
            self.db_client = PGVectorClient()
        elif self._backend == "opensearch":
            from app.vector.vector_client import OpenSearchVectorClient
            self.db_client = OpenSearchVectorClient()
        else:
            raise ValueError(f"Unsupported vector backend: {backend}")

    def get_query_embedding(self, query: str) -> List[float]:
        if query in self._cache:
            return self._cache[query]
        vec = self._embed.embed([query])[0]
        self._cache[query] = vec
        return vec

    def search(self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        vec = self.get_query_embedding(query)
        return self.db_client.query(query_vector=vec, top_k=top_k, where=where or {})
