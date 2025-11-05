# app/vector/vector_client.py
# Vector-agnostic interface and concrete clients. Only this layer changes when swapping backends.

from typing import List, Dict, Any, Optional, Protocol

class VectorClient(Protocol):
    def query(self, query_vector: List[float], top_k: int, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

class ChromaVectorClient:
    def __init__(self, chroma_service):
        self._chroma = chroma_service

    def query(self, query_vector: List[float], top_k: int, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Delegate to ChromaService with explicit vector
        return self._chroma.query_by_vector(query_vector=query_vector, n_results=top_k, where=where)

class PGVectorClient:
    # Placeholder: implement with asyncpg/psycopg vector ops or pgvector extension
    def __init__(self, conn=None, table: str = "documents"):
        self._conn = conn
        self._table = table

    def query(self, query_vector: List[float], top_k: int, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Implement cosine/inner-product via pgvector; return {"ids":[[]], "documents":[[]], "metadatas":[[]]}
        raise NotImplementedError("PGVectorClient.query not implemented yet")

class OpenSearchVectorClient:
    # Placeholder: implement with opensearch-py knn plugin or ANN
    def __init__(self, index: str = "documents", client=None):
        self._index = index
        self._client = client

    def query(self, query_vector: List[float], top_k: int, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Implement knn search against index; return {"ids":[[]], "documents":[[]], "metadatas":[[]]}
        raise NotImplementedError("OpenSearchVectorClient.query not implemented yet")
