# app/service/rag_search_service.py
from typing import List
from app.config.app_config import AppConfigSingleton
from app.utils.app_logging import get_logger
from app.config.vector_db_client import VectorDBClient
from app.models.rag_models import RetrieveResponseHit, RetrieveResponse

_cfg = AppConfigSingleton.instance()
_logger = get_logger(_cfg)
_vdb = VectorDBClient(backend="chroma")

class RAGSearchService:
    def retrieve(self, query: str, n: int = 5) -> RetrieveResponse:
        res = _vdb.search(query=query, top_k=n)
        ids = res.get("ids", [[]])[0]; docs = res.get("documents", [[]])[0]; metas = res.get("metadatas", [[]])[0]
        hits: List[RetrieveResponseHit] = []
        for i, _id in enumerate(ids):
            meta = metas[i] or {}
            hits.append(RetrieveResponseHit(
                id=_id,
                parent_id=meta.get("parent_id") or (_id.split("::chunk::")[0] if "::chunk::" in _id else _id),
                text=docs[i] or "",
                metadata=meta,
                score=meta.get("score")
            ))
        return RetrieveResponse(query=query, hits=hits)
