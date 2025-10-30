from fastapi import APIRouter, Query, HTTPException, Path as FPath
from typing import Optional, Dict, Any, List
import time
from pydantic import BaseModel
from app.utils.app_logging import get_logger
from app.config.app_config import AppConfig
from app.service.chroma_client_service import ChromaClientService
from app.service.rag_search_service import RAGSearchService
from app.models.rag_models import RetrieveResponse, RetrieveResponseHit, QARequest, QAResponse

cfg = AppConfig()
logger = get_logger(cfg)
rag_router = APIRouter(prefix="/rag-search", tags=["rag-search"])
chromaClientService = ChromaClientService()
ragService = RAGSearchService()

class BatchQARequest(BaseModel):
    questions: List[str]
    n_results: int = 8
    top_k_ctx: int = 4

@rag_router.get("/retrieve", response_model=RetrieveResponse)
async def retrieve(query: str = Query(...), n_results: int = Query(8, ge=1, le=25),
                   advisor_id: Optional[str] = None, client_id: Optional[str] = None, doc_type: Optional[str] = None):
    t0 = time.perf_counter()
    where: Dict[str, Any] = {}
    if advisor_id: where["advisor_id"] = advisor_id
    if client_id: where["client_id"] = client_id
    if doc_type: where["doc_type"] = doc_type
    logger.info("[RAG] Retrieve begin query='%s' where=%s", query, where or None)
    try:
        res = chromaClientService.query(query_text=query, n_results=n_results, where=where or None)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        hits: List[RetrieveResponseHit] = []
        for i, _id in enumerate(ids):
            hits.append(RetrieveResponseHit(
                id=_id,
                parent_id=(metas[i] or {}).get("parent_id"),
                text=(docs[i] or "")[:600],
                metadata=metas[i]
            ))
        lapse = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("[RAG] Retrieve ok hits=%d lapse_ms=%.2f", len(hits), lapse)
        return RetrieveResponse(query=query, hits=hits, retrieval_lapse_time=lapse)
    except Exception as e:
        lapse = round((time.perf_counter() - t0) * 1000, 2)
        logger.exception("[RAG] Retrieve failed: %s lapse_ms=%.2f", e, lapse)
        return RetrieveResponse(query=query, hits=[], retrieval_lapse_time=lapse,
                                file_error_info={"stage":"retrieve","type":e.__class__.__name__,"message":str(e)})

@rag_router.get("/get_full_text/{id}")
async def get_full_text(id: str = FPath(...)) -> dict:
    logger.info("[RAG] Get full text id=%s", id)
    res = chromaClientService.get(id)
    if not res.get("ids"):
        logger.error("[RAG] Full text not found id=%s", id)
        raise HTTPException(status_code=404, detail="Document not found")
    logger.info("[RAG] Full text ok id=%s", id)
    return {"id": id, "text": res["documents"][0], "metadata": res["metadatas"][0]}

@rag_router.post("/user_query", response_model=QAResponse)
async def user_query(req: QARequest):
    logger.info("[RAG] QA begin question='%s' n_results=%d top_k=%d", req.question, req.n_results, req.top_k_ctx)
    out = ragService.ask(question=req.question, n_results=req.n_results, top_k_ctx=req.top_k_ctx)
    if out.get("file_llm_status") == "success":
        logger.info("[RAG] QA ok citations=%s llm_ms=%.2f", out.get("citations"), out.get("llm_lapse_time", 0.0))
    else:
        logger.error("[RAG] QA failed error_info=%s", out.get("file_error_info"))
    # FastAPI will validate against QAResponse model
    return out

@rag_router.post("/user_query_debug")
async def user_query_debug(req: QARequest):
    logger.info("[RAG] user_query_debug q='%s'", req.question)
    return ragService.ask_with_debug(question=req.question, n_results=req.n_results, top_k_ctx=req.top_k_ctx)

@rag_router.post("/user_query_eval")
async def user_query_eval(req: BatchQARequest):
    logger.info("[RAG] user_query_eval n=%d", len(req.questions))
    return ragService.ask_batch(questions=req.questions, n_results=req.n_results, top_k_ctx=req.top_k_ctx)