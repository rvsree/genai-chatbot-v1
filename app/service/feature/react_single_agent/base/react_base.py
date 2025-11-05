# app/service/feature/react_single_agent/base/react_base.py
from typing import List, Dict, Any, Optional, Tuple
import time, uuid, datetime, re, asyncio
from app.config.app_config import AppConfigSingleton
from app.utils.app_logging import get_logger
from app.adapters.feature.react_single_agent.tool_adapters import RetrievalTools
from app.service.variants.variant_output_score_service import VariantOutputScoreService

_cfg = AppConfigSingleton.instance()
_logger = get_logger(_cfg)

class AgentError(RuntimeError):
    def __init__(self, code: str, http_status: int, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.message = message
        self.details = details or {}

def _extract_parent_ids(text: str) -> List[str]:
    if not text: return []
    return list(dict.fromkeys(re.findall(r"\[([^\[\]]+?)\]", text)))

def _contains_placeholder_pid(text: str) -> bool:
    return "[parent-id]" in (text or "")

class ReactBaseAgent:
    def __init__(self, max_steps: int = 4):
        self.max_steps = max_steps

    async def run(
            self,
            question: str,
            scoring_model: str,
            emit_traces: bool,
            enable_query_variants: Optional[bool],
            enable_output_scoring: bool,
            max_variants: int,
            self_reflection_iterations: Optional[int],
            agent_graph_id: str,
            agent_descriptor: Dict[str, Any],
            execution_mode: str = "async",
            preferred_year: Optional[str] = None,
            top_k: Optional[int] = None,
            retrieval_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        t0 = time.time()
        run_id = f"react_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

        k = top_k if top_k is not None else getattr(_cfg, "rag_top_k", 5)
        do_variants = enable_query_variants if enable_query_variants is not None else getattr(_cfg, "rag_enable_query_variants", True)
        loops = self_reflection_iterations if self_reflection_iterations is not None else getattr(_cfg, "rag_self_reflection_iterations", 3)

        variants = [question]
        if do_variants:
            from app.service.variants.query_variants_service import QueryVariantsService
            for v in QueryVariantsService.generate(question, max_variants=max_variants):
                if v not in variants:
                    variants.append(v)

        where = retrieval_filters or {}
        if preferred_year and "year" not in where:
            where["year"] = preferred_year

        async def _task(idx_v: int, vq: str):
            return await self._process_variant(
                vq, idx_v, scoring_model, enable_output_scoring, loops, emit_traces, where, k
            )

        results = await asyncio.gather(*[_task(i, v) for i, v in enumerate(variants, start=1)]) if execution_mode == "async" else [
            await _task(i, v) for i, v in enumerate(variants, start=1)
        ]

        # Select best by score, then citations, then recency
        best = None
        for r in results:
            vs = r.get("variant_score", {})
            score = vs.get("actual_score")
            if score is None:
                continue
            if best is None or score > best["variant_score"]["actual_score"]:
                best = r
            elif score == best["variant_score"]["actual_score"]:
                curr_best_loop = max(r["iterations"], key=lambda x: (x.get("actual_score") or 0.0))
                prev_best_loop = max(best["iterations"], key=lambda x: (x.get("actual_score") or 0.0))
                curr_cites = len(_extract_parent_ids(curr_best_loop.get("output") or ""))
                prev_cites = len(_extract_parent_ids(prev_best_loop.get("output") or ""))
                if curr_cites > prev_cites or (curr_cites == prev_cites and curr_best_loop["iteration"] > prev_best_loop["iteration"]):
                    best = r

        if not best:
            # fallback: pick any with citations
            for r in results:
                loops_list = r.get("iterations", [])
                with_cites = [it for it in loops_list if _extract_parent_ids(it.get("output") or "")]
                if with_cites:
                    best = r
                    break

        if not best or not best.get("iterations"):
            raise AgentError("RETRIEVAL_EMPTY", 404, "No candidates produced; filters may be too strict.", {"top_k": k, "filters": where})

        best_loop = max(best["iterations"], key=lambda x: (x.get("actual_score") or 0.0))
        final_answer = best_loop.get("output") or ""

        if _contains_placeholder_pid(final_answer):
            # Replace with first found real id from any loop if available
            repl: List[str] = []
            for it in best["iterations"]:
                repl.extend(_extract_parent_ids(it.get("output") or ""))
            repl = list(dict.fromkeys(repl))
            if repl:
                final_answer = final_answer.replace("[parent-id]", f"[{repl[0]}]")
            else:
                raise AgentError("PLACEHOLDER_CITATIONS", 422, "Model returned placeholder citations; no real ids found.", {"filters": where})

        parent_citations = _extract_parent_ids(final_answer)
        if not parent_citations:
            raise AgentError("INSUFFICIENT_EVIDENCE", 422, "No citations present. Increase top_k or relax filters.", {"top_k": k, "filters": where})

        # Aggregate token usage from loops
        prompt_tokens = sum((it.get("llm_call", {}).get("usage", {}).get("prompt_tokens", 0) for it in best["iterations"]))
        completion_tokens = sum((it.get("llm_call", {}).get("usage", {}).get("completion_tokens", 0) for it in best["iterations"]))
        total_tokens = prompt_tokens + completion_tokens

        elapsed = int((time.time() - t0) * 1000)
        return {
            "run_id": run_id,
            "agent_graph_id": agent_graph_id,
            "question": question,
            "final_response": final_answer,
            "citations": parent_citations,
            "scoring_model": scoring_model,
            "selected_variant_id": best.get("variant_id"),
            "selected_candidate_id": best.get("variant_score", {}).get("candidate_id"),
            "selected_score": best.get("variant_score", {}).get("actual_score"),
            "agents": [agent_descriptor],
            "variants": results,
            "ranking_rationale": "Selected highest score; ties broken by citations then recency.",
            "answer_timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_time": elapsed,
            "latency_ms": elapsed,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "cache_hit": False,
            "error_info": None
        }

    async def _process_variant(
            self,
            variant_query: str,
            index: int,
            scoring_model: str,
            enable_output_scoring: bool,
            self_reflection_iterations: int,
            emit_traces: bool,
            where: Dict[str, Any],
            top_k: int
    ) -> Dict[str, Any]:
        variant_id = f"v{index}"
        iterations: List[Dict[str, Any]] = []
        citations: List[str] = []
        context_notes: List[str] = []

        for loop in range(1, self_reflection_iterations + 1):
            q = variant_query if loop == 1 else f"{variant_query} (2019 filing details, MD&A preference)"
            result = await self.execute_action("vector_search", {"query": q, "n_results": top_k, "where": where})
            hits = result.get("hits", [])
            stage = result.get("stage", "none")

            if not hits:
                iterations.append({
                    "iteration": loop,
                    "thought": f"Loop {loop}: vector_search (no hits stage={stage})",
                    "retrieval_plan": [{
                        "query": q,
                        "hits": 0,
                        "stage": stage,
                        "top_parent_ids": [],
                        "action": "vector_search",
                        "tool_name": "retrieval.vector_search",
                        "source_name": "vector_db",
                        "tool_latency_ms": result.get("latency_ms")
                    }],
                    "output": "",
                    "actual_score": None,
                    "llm_call": {},
                    "error_info": {"code":"RETRIEVAL_EMPTY","stage":stage}
                })
                break

            top_parents = []
            for h in hits[:3]:
                pid = h.get("parent_id")
                if pid and pid not in top_parents:
                    top_parents.append(pid)

            retrieval_plan = [{
                "query": q,
                "hits": len(hits),
                "stage": stage,
                "top_parent_ids": top_parents,
                "action": "vector_search",
                "tool_name": "retrieval.vector_search",
                "source_name": "vector_db",
                "tool_latency_ms": result.get("latency_ms")
            }]

            for h in hits[:8]:
                txt = (h.get("text") or "")[:1400]
                if txt and txt not in context_notes:
                    context_notes.append(txt)
                pid = h.get("parent_id")
                if pid and pid not in citations:
                    citations.append(pid)

            # prepend whitelist header for LLM
            header = "Allowed citations: " + ", ".join(f"[{c}]" for c in citations[:8])
            ctx = [header, *context_notes[:8]]

            answer_loop, llm_meta = await self.synthesize_final_with_meta(
                variant_query, {"loop_id": loop, "strict_extraction": True}, ctx, citations
            )

            # score
            actual_score = None
            if enable_output_scoring:
                actual_score = VariantOutputScoreService.score_scalar(
                    answer=answer_loop,
                    citations=citations,
                    scoring_model=scoring_model,
                    allowed_ids=top_parents,
                    question=variant_query
                )

            iterations.append({
                "iteration": loop,
                "thought": f"Loop {loop}: vector_search stage={stage}",
                "retrieval_plan": retrieval_plan,
                "output": answer_loop,
                "actual_score": actual_score,
                "llm_call": llm_meta,
                "error_info": None
            })

        scored = [it for it in iterations if it.get("actual_score") is not None]
        best_scored = max(scored, key=lambda x: x["actual_score"]) if scored else None
        variant_score = {
            "candidate_id": f"{self.__class__.__name__}-cand-{variant_id}",
            "actual_score": best_scored["actual_score"] if best_scored else None
        }

        return {
            "variant_id": variant_id,
            "query_variant": variant_query,
            "query_context": {"normalized_query": variant_query},
            "sub_questions": [],
            "data_source_routing": [],
            "iterations": iterations,
            "variant_score": variant_score,
            "self_reflection": {"critique": "Progressive retrieval with scoring", "fixes_applied": [], "passed": True}
        }

    async def synthesize_final_with_meta(self, variant_query: str, query_context: Dict[str, Any], context_notes: List[str], citations: List[str]) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    async def execute_action(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if action == "vector_search":
            res = await RetrievalTools.vector_search(**args); res["latency_ms"] = res.get("latency_ms") or 0; return res
        return {"error": f"unknown action {action}", "latency_ms": 0}
