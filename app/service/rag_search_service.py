from typing import List, Dict, Any
import time
from openai import OpenAI
from app.config.app_config import AppConfig
from app.utils.app_logging import get_logger
from app.service.chroma_client_service import ChromaClientService
from app.prompts.lab_prompts import LAB_SYSTEM_PROMPT, LAB_USER_TEMPLATE

cfg = AppConfig()
logger = get_logger(cfg)

MAX_CHARS_PER_CHUNK = 1200

class RAGSearchService:
    def __init__(self):
        logger.info("[RAGSearchService] Initialize")
        self.svc = ChromaClientService()
        self.client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
        self.model = cfg.openai_llm_model
        logger.info("[RAGSearchService] Ready model=%s", self.model)

    def _context_from_query(self, retr: Dict[str, Any], top_k: int) -> tuple[str, List[str], List[Dict[str, Any]]]:
        ids = retr.get("ids", [[]])[0]
        docs = retr.get("documents", [[]])[0]
        metas = retr.get("metadatas", [[]])[0]
        blocks: List[str] = []
        parent_ids: List[str] = []
        debug_blocks: List[Dict[str, Any]] = []
        for i, _id in enumerate(ids[:top_k]):
            meta = metas[i] if i < len(metas) else {}
            parent_id = meta.get("parent_id") or _id.split("::chunk::")[0]
            parent_ids.append(parent_id)
            header = f"[{parent_id}] {meta.get('filename','')}"
            body = (docs[i] or "").strip()[:MAX_CHARS_PER_CHUNK]
            blocks.append(f"{header}\n{body}")
            debug_blocks.append({"id": _id, "parent_id": parent_id, "snippet": body})
        # De-dupe parent ids
        seen = set()
        uniq = []
        for p in parent_ids:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return "\n\n".join(blocks), uniq, debug_blocks

    def ask(self, question: str, n_results: int = 8, top_k_ctx: int = 4) -> Dict[str, Any]:
        logger.info("[RAGSearchService] QA begin q='%s' n_results=%d top_k=%d", question, n_results, top_k_ctx)
        t0 = time.perf_counter()
        try:
            r0 = time.perf_counter()
            retr = self.svc.query(query_text=question, n_results=n_results, where=None)
            retrieval_ms = round((time.perf_counter() - r0) * 1000, 2)
            logger.info("[RAGSearchService] Retrieved in %.2f ms", retrieval_ms)

            context, parent_ids, _ = self._context_from_query(retr, top_k_ctx)
            messages = [
                {"role": "system", "content": LAB_SYSTEM_PROMPT},
                {"role": "user", "content": LAB_USER_TEMPLATE.format(question=question, context=context)}
            ]

            l0 = time.perf_counter()
            logger.info("[RAGSearchService] Calling OpenAI model=%s", self.model)
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.2, top_p=1.0, max_tokens=512
            )
            llm_ms = round((time.perf_counter() - l0) * 1000, 2)
            answer = resp.choices[0].message.content
            logger.info("[RAGSearchService] QA ok llm_ms=%.2f", llm_ms)

            return {
                "question": question,
                "answer": answer,
                "citations": parent_ids[:top_k_ctx],
                "retrieval_lapse_time": retrieval_ms,
                "llm_lapse_time": llm_ms,
                "file_llm_status": "success"
            }
        except Exception as e:
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.exception("[RAGSearchService] QA failed total_ms=%.2f error=%s", total_ms, e)
            return {
                "question": question,
                "answer": "",
                "citations": [],
                "retrieval_lapse_time": 0.0,
                "llm_lapse_time": 0.0,
                "file_llm_status": "failed",
                "file_error_info": {"stage": "rag-search", "type": e.__class__.__name__, "message": str(e)}
            }

    def ask_with_debug(self, question: str, n_results: int = 8, top_k_ctx: int = 4) -> Dict[str, Any]:
        logger.info("[RAGSearchService] QA debug begin q='%s'", question)
        t0 = time.perf_counter()
        try:
            retr = self.svc.query(query_text=question, n_results=n_results, where=None)
            context, parent_ids, debug = self._context_from_query(retr, top_k_ctx)
            messages = [
                {"role": "system", "content": LAB_SYSTEM_PROMPT},
                {"role": "user", "content": LAB_USER_TEMPLATE.format(question=question, context=context)}
            ]
            l0 = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.2, top_p=1.0, max_tokens=512
            )
            llm_ms = round((time.perf_counter() - l0) * 1000, 2)
            answer = resp.choices[0].message.content
            return {
                "question": question,
                "context_blocks": debug,
                "answer": answer,
                "citations": parent_ids[:top_k_ctx],
                "retrieval_lapse_time": 0.0,
                "llm_lapse_time": llm_ms,
                "file_llm_status": "success"
            }
        except Exception as e:
            logger.exception("[RAGSearchService] QA debug failed: %s", e)
            return {
                "question": question,
                "context_blocks": [],
                "answer": "",
                "citations": [],
                "retrieval_lapse_time": 0.0,
                "llm_lapse_time": 0.0,
                "file_llm_status": "failed",
                "file_error_info": {"stage": "rag-search", "type": e.__class__.__name__, "message": str(e)}
            }

    def ask_batch(self, questions: List[str], n_results: int = 8, top_k_ctx: int = 4) -> Dict[str, Any]:
        logger.info("[RAGSearchService] Batch eval n=%d", len(questions))
        results: List[Dict[str, Any]] = []
        for q in questions:
            results.append(self.ask(q, n_results=n_results, top_k_ctx=top_k_ctx))
        return {"results": results}
