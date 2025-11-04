from typing import Dict, Any, Optional, List
from app.utils.app_logging import get_logger
from app.config.app_config import AppConfig
from app.service.chroma_client_service import ChromaClientService

cfg = AppConfig()
logger = get_logger(cfg)
chroma = ChromaClientService()

class RetrievalTools:
    """Adapters to shared retrieval; no duplication."""

    @staticmethod
    def index_lookup(parent_id: str) -> Dict[str, Any]:
        logger.info("[Tools] index_lookup parent_id=%s", parent_id)
        ids = chroma.get_ids_by_parent(parent_id)
        return {"parent_id": parent_id, "count": len(ids), "ids": ids}

    @staticmethod
    def vector_search(query: str, n_results: int = 5,
                      advisor_id: Optional[str] = None,
                      client_id: Optional[str] = None,
                      doc_type: Optional[str] = None) -> Dict[str, Any]:
        query = (query or "").strip()
        try:
            n_results = int(n_results)
        except Exception:
            n_results = 5
        where: Dict[str, Any] = {}
        if advisor_id: where["advisor_id"] = advisor_id
        if client_id: where["client_id"] = client_id
        if doc_type: where["doc_type"] = doc_type

        logger.info("[Tools] vector_search begin query='%s' where=%s n=%d", query, where or None, n_results)
        res = chroma.query(query_text=query, n_results=n_results, where=where or None)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        hits: List[Dict[str, Any]] = []
        for i, _id in enumerate(ids):
            meta = metas[i] or {}
            hits.append({
                "id": _id,
                "parent_id": meta.get("parent_id") or _id.split("::chunk::")[0],
                "text": (docs[i] or "")[:800],
                "metadata": meta
            })
        logger.info("[Tools] vector_search hits=%d", len(hits))
        return {"query": query, "hits": hits}

    @staticmethod
    def get_chunk(id: str) -> Dict[str, Any]:
        logger.info("[Tools] get_chunk id=%s", id)
        res = chroma.get(id)
        if not res.get("ids"):
            return {"id": id, "found": False}
        return {"id": id, "found": True, "text": res["documents"][0], "metadata": res["metadatas"][0]}
