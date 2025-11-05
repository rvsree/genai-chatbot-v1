# app/adapters/feature/react_single_agent/tool_adapters.py
from typing import Dict, Any, Optional, List, Tuple
from app.config.app_config import AppConfigSingleton
from app.utils.app_logging import get_logger
from app.config.vector_db_client import VectorDBClient

_cfg = AppConfigSingleton.instance()
_logger = get_logger(_cfg)
_vdb = VectorDBClient(backend="chroma")

def _normalize_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not where: return None
    norm: Dict[str, Any] = {}
    for k, v in where.items():
        lk = k.lower()
        if lk in ["year", "filing_year"]:
            norm["year"] = str(v)
        elif lk in ["form", "filing_form", "sec_form"]:
            norm["form"] = str(v).lower()
        elif lk in ["doc_type", "doctype", "document_type"]:
            norm["doc_type"] = str(v)
        elif lk in ["issuer", "company", "ticker"]:
            norm["issuer"] = str(v)
        else:
            norm[lk] = v
    return norm

def _query_with_where(query: str, top_k: int, where: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return _vdb.search(query=query, top_k=top_k, where=where)

def _build_hits(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    hits: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        meta = metas[i] or {}
        pid = meta.get("parent_id")
        if not pid:
            cid = ids[i] or ""
            pid = cid.split("::chunk::")[0] if "::chunk::" in cid else cid
        hits.append({
            "id": ids[i],
            "text": docs[i],
            "parent_id": pid,
            "meta": meta
        })
    return hits

class RetrievalTools:
    @staticmethod
    async def vector_search(query: str, n_results: int = 5, where: Dict[str, Any] = None) -> Dict[str, Any]:
        candidates: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        base = _normalize_where(where)
        candidates.append(("strict", base))
        if base:
            bB = dict(base)
            if "year" in bB:
                bB["form"] = (bB.get("form") or "10-k").lower()
                candidates.append(("year+form", bB))
        if base and "year" in base:
            candidates.append(("year-only", {"year": base["year"]}))
        candidates.append(("unfiltered", None))

        for stage, filt in candidates:
            _logger.info("[Tools] vector_search stage=%s query='%s' where=%s n=%d", stage, query, filt, n_results)
            res = await _query_with_where(query, n_results, filt)
            hits = _build_hits(res)
            _logger.info("[Tools] vector_search stage=%s hits=%d", stage, len(hits))
            if hits:
                res["hits"] = hits
                res["stage"] = stage
                return res

        return {"hits": [], "ids": [[]], "documents": [[]], "metadatas": [[]], "stage": "none", "latency_ms": 0}
