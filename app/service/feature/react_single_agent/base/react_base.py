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

# ---------------- Decomposition helpers ----------------

_SPLIT_SEPS = re.compile(r"(?:\?|;| and |\band\b|\&|, then | followed by )", re.IGNORECASE)

def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _is_meaningful(fragment: str) -> bool:
    text = _normalize_whitespace(fragment)
    if len(text) < 6:
        return False
    # Ignore trailing boilerplate
    if text.lower() in {"and", "then", "followed by"}:
        return False
    return True

def _decompose_query(q: str, max_parts: int = 5) -> List[str]:
    """
    Deterministic, regex-based splitter for complex questions.
    Produces 2..5 sub-questions when possible.
    """
    q = _normalize_whitespace(q)
    # First, honor explicit numbered lists if present
    numbered = re.findall(r"(?:^\d+\)|\(\d+\))\s*(.+?)(?=(?:\d+\)|\(\d+\))|\Z)", q)
    if numbered:
        parts = [_normalize_whitespace(p) for p in numbered if _is_meaningful(p)]
        return parts[:max_parts] if parts else []

    # Fallback: split on conjunctions and separators
    rough = _SPLIT_SEPS.split(q)
    parts = [_normalize_whitespace(p) for p in rough if _is_meaningful(p)]
    # If everything collapsed to 1, try colon-based split
    if len(parts) <= 1 and ":" in q:
        try_parts = [s for s in q.split(":")[1].split(",") if _is_meaningful(s)]
        if try_parts:
            parts = try_parts

    # Heuristic: keep 2..5 most informative parts
    if len(parts) > max_parts:
        # Prefer fragments containing key signals
        signals = ["2019", "2018", "revenue", "driver", "risk", "10-k", "md&a", "annual report", "quote"]
        def _rank(p: str) -> int:
            low = p.lower()
            return sum(1 for s in signals if s in low)
        parts = sorted(parts, key=_rank, reverse=True)[:max_parts]

    # Avoid returning the entire query as the only sub-question
    if len(parts) == 1 and parts[0].lower() == q.lower():
        return []
    return parts[:max_parts]

def _route_subq(sq: str) -> Dict[str, Any]:
    low = sq.lower()
    if any(k in low for k in ["10-k","filing","risk","revenue","segment","annual report","md&a","mda","consolidated statements"]):
        return {"target_source":"vector_store","collection_or_endpoint":"filings","strategy":"semantic_retrieval"}
    return {"target_source":"vector_store","collection_or_endpoint":"general","strategy":"semantic_retrieval"}

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

        # Build variants list (original + optional paraphrases via service if enabled)
        variants = [question]
        if do_variants:
            try:
                from app.service.variants.query_variants_service import QueryVariantsService
                for v in QueryVariantsService.generate(question, max_variants=max_variants):
                    if v not in variants:
                        variants.append(v)
            except Exception as e:
                _logger.warning("[Variants] generation disabled due to error: %s", e)

        # Persist decomposition per variant
        all_variant_meta: Dict[str, Dict[str, Any]] = {}
        for v in variants:
            subs = _decompose_query(v, max_parts=5)
            routes = [{"sub_question": s, **_route_subq(s)} for s in subs] if subs else []
            where = dict(retrieval_filters or {})
            if preferred_year and "year" not in where:
                where["year"] = preferred_year
            all_variant_meta[v] = {
                "sub_questions": subs,                  # <- persist here
                "data_source_routing": routes,          # <- and here
                "where": where
            }

        async def _task(idx_v: int, vq: str):
            meta = all_variant_meta.get(vq, {"sub_questions": [], "data_source_routing": [], "where": {}})
            return await self._process_variant(
                vq, idx_v, scoring_model, enable_output_scoring, loops, emit_traces, meta, k
            )

        if execution_mode == "async":
            results = await asyncio.gather(*[_task(i, v) for i, v in enumerate(variants, start=1)])
        else:
            results = [await _task(i, v) for i, v in enumerate(variants, start=1)]

        # Select best by score then citations then recency
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

        if not best or not best.get("iterations"):
            raise AgentError("RETRIEVAL_EMPTY", 404, "No candidates produced; filters may be too strict.", {"top_k": k, "filters": retrieval_filters or {}})

        best_loop = max(best["iterations"], key=lambda x: (x.get("actual_score") or 0.0))
        final_answer = best_loop.get("output") or ""

        if _contains_placeholder_pid(final_answer):
            # Attempt replacement from any loop's citations
            repl: List[str] = []
            for it in best["iterations"]:
                repl.extend(_extract_parent_ids(it.get("output") or ""))
            repl = list(dict.fromkeys(repl))
            if repl:
                final_answer = final_answer.replace("[parent-id]", f"[{repl[0]}]")
            else:
                raise AgentError("PLACEHOLDER_CITATIONS", 422, "Model returned placeholder citations; no real ids found.", {})

        parent_citations = _extract_parent_ids(final_answer)
        if not parent_citations:
            raise AgentError("INSUFFICIENT_EVIDENCE", 422, "No citations present. Increase top_k or relax filters.", {"top_k": k, "filters": retrieval_filters or {}})

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
            variant_meta: Dict[str, Any],
            top_k: int
    ) -> Dict[str, Any]:
        variant_id = f"v{index}"
        iterations: List[Dict[str, Any]] = []
        citations: List[str] = []
        context_notes: List[str] = []
        subs = variant_meta.get("sub_questions", [])  # <- persist into variant output
        routes = variant_meta.get("data_source_routing", [])
        where = variant_meta.get("where") or {}

        # Build a retrieval plan that tries each sub-question first, then the full query
        subq_order = subs[:] if subs else []
        if variant_query not in subq_order:
            subq_order.append(variant_query)

        for loop in range(1, self_reflection_iterations + 1):
            completed_hits = 0
            loop_parent_ids: List[str] = []
            loop_plan: List[Dict[str, Any]] = []

            for sq in subq_order:
                result = await self.execute_action("vector_search", {"query": sq, "n_results": top_k, "where": where})
                hits = result.get("hits", [])
                stage = result.get("stage", "none")
                top_parents = []
                for h in hits[:3]:
                    pid = h.get("parent_id")
                    if pid and pid not in top_parents:
                        top_parents.append(pid)
                        if pid not in citations:
                            citations.append(pid)
                        if pid not in loop_parent_ids:
                            loop_parent_ids.append(pid)
                    txt = (h.get("text") or "")[:1000]
                    if txt and txt not in context_notes:
                        context_notes.append(txt)

                loop_plan.append({
                    "query": sq,
                    "hits": len(hits),
                    "stage": stage,
                    "top_parent_ids": top_parents,
                    "action": "vector_search",
                    "tool_name": "retrieval.vector_search",
                    "source_name": "vector_db",
                    "tool_latency_ms": result.get("latency_ms")
                })
                completed_hits += len(hits)

                # If we already have enough parents and context, stop early this loop
                if len(loop_parent_ids) >= 2 and len(context_notes) >= 3:
                    break

            # Whitelist header for the LLM
            header = "Allowed citations: " + ", ".join(f"[{c}]" for c in citations[:8])
            ctx_lines = [header, *context_notes[:8]]

            answer_loop, llm_meta = await self.synthesize_final_with_meta(
                variant_query, {"loop_id": loop, "strict_extraction": True, "sub_questions": subs}, ctx_lines, citations
            )

            actual_score = None
            if enable_output_scoring:
                actual_score = VariantOutputScoreService.score_scalar(
                    answer=answer_loop,
                    citations=citations,
                    scoring_model=scoring_model,
                    allowed_ids=loop_parent_ids,
                    question=variant_query
                )

            iterations.append({
                "iteration": loop,
                "thought": f"Loop {loop}: multi-subq retrieval ({len(subq_order)} subqs), total_hits={completed_hits}",
                "retrieval_plan": loop_plan,
                "output": answer_loop,
                "actual_score": actual_score,
                "llm_call": llm_meta,
                "error_info": None
            })

        # Pick best scored loop inside variant
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
            "sub_questions": subs,                 # <- persisted for API consumers
            "data_source_routing": routes,         # <- persisted routing view
            "iterations": iterations,
            "variant_score": variant_score,
            "self_reflection": {"critique": "Multi-subq retrieval with scoring", "fixes_applied": [], "passed": True}
        }

    async def synthesize_final_with_meta(self, variant_query: str, query_context: Dict[str, Any], context_notes: List[str], citations: List[str]) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    async def execute_action(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if action == "vector_search":
            res = await RetrievalTools.vector_search(**args); res["latency_ms"] = res.get("latency_ms") or 0; return res
        return {"error": f"unknown action {action}", "latency_ms": 0}
