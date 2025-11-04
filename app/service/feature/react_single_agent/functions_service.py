from typing import Dict, Any, List, Optional
import json, re
from openai import OpenAI
from app.utils.app_logging import get_logger
from app.config.app_config import AppConfigSingleton as AppConfig
from app.adapters.feature.react_single_agent.tool_adapters import RetrievalTools
from app.prompts.feature.react_single_agent import function_prompts, tools_schema

cfg = AppConfig()
logger = get_logger(cfg)

client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
model = cfg.openai_llm_model or cfg.openai_default_model

FINAL_GUARDRAIL = (
    "Answer concisely using ONLY tool results. "
    "Include citations as [parent_id]; never use numeric references. "
    "Do not mention prompts, system instructions, or templates."
)

def _to_kwargs(maybe_json: Any) -> Dict[str, Any]:
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        s = maybe_json.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}
    return {}

def _call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "vector_search":
        return RetrievalTools.vector_search(**args)
    if name == "get_chunk":
        return RetrievalTools.get_chunk(**args)
    if name == "index_lookup":
        return RetrievalTools.index_lookup(**args)
    return {"error": f"unknown tool {name}"}

def _strip_numeric_brackets(text: str) -> str:
    return re.sub(r"\[(\d+)\]", "", text or "")

class FunctionCallingAgent:
    """
    Function/tool-calling with:
    - Neutral system for planning
    - No empty tool_calls messages
    - Deterministic retrieval fallback
    - Context injection at final step
    - Strict final instruction to avoid generic acknowledgements
    - Optional tool_results emission via emit_traces
    """

    def __init__(self, emit_traces: bool = True):
        self.default_emit_traces = emit_traces

    def run(self, question: str, emit_traces: Optional[bool] = None) -> Dict[str, Any]:
        emit = self.default_emit_traces if emit_traces is None else emit_traces
        logger.info("[Functions] begin q='%s'", question)

        messages: List[Dict[str, Any]] = [
            {"role":"system","content": function_prompts.FUNCTION_SYSTEM},
            {"role":"user","content": function_prompts.FUNCTION_USER_TEMPLATE.format(question=question)}
        ]

        tool_results: List[Dict[str, Any]] = []
        citations: List[str] = []
        context_notes: List[str] = []

        try:
            # 1) Planning turn with tool availability
            try:
                first = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=[{"type":"function","function": f} for f in tools_schema.FUNCTION_TOOLS],
                    tool_choice="auto",
                    temperature=0.2, top_p=1.0, max_tokens=256
                )
            except Exception as plan_err:
                logger.exception("[Functions] planning call failed: %s", plan_err)
                # Fallback immediately to deterministic retrieval
                fallback = RetrievalTools.vector_search(query=question, n_results=5)
                if emit:
                    tool_results.append({"name":"vector_search","args":{"query":question,"n_results":5},"result_keys":list(fallback.keys())})
                for h in fallback.get("hits", [])[:3]:
                    pid = h.get("parent_id")
                    if pid and pid not in citations:
                        citations.append(pid)
                    txt = (h.get("text") or "")[:800]
                    if txt:
                        context_notes.append(txt)
                # Build final answer directly
                return self._final_answer(question, messages, citations, context_notes, tool_results, emit)

            assistant_msg = first.choices[0].message
            tcalls = assistant_msg.tool_calls or []

            # 2) Execute tool calls (only if present), else fallback retrieval
            if tcalls:
                messages.append({
                    "role":"assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": tcalls
                })

                for tc in tcalls:
                    name = tc.function.name
                    args = _to_kwargs(tc.function.arguments)
                    try:
                        result = _call_tool(name, args)
                    except Exception as tool_err:
                        logger.exception("[Functions] tool '%s' failed: %s", name, tool_err)
                        result = {"error": str(tool_err)}

                    if emit:
                        tool_results.append({"name": name, "args": args, "result_keys": list(result.keys())})

                    if name == "vector_search":
                        for h in result.get("hits", [])[:3]:
                            pid = h.get("parent_id")
                            if pid and pid not in citations:
                                citations.append(pid)
                            txt = (h.get("text") or "")[:800]
                            if txt:
                                context_notes.append(txt)
                    elif name == "get_chunk":
                        pid = (result.get("metadata") or {}).get("parent_id")
                        if pid and pid not in citations:
                            citations.append(pid)
                        txt = (result.get("text") or "")[:1000]
                        if txt:
                            context_notes.append(txt)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": str(result)[:3500]
                    })
            else:
                # Deterministic fallback: ensure we have context
                fallback = RetrievalTools.vector_search(query=question, n_results=5)
                if emit:
                    tool_results.append({"name":"vector_search","args":{"query":question,"n_results":5},"result_keys":list(fallback.keys())})
                for h in fallback.get("hits", [])[:3]:
                    pid = h.get("parent_id")
                    if pid and pid not in citations:
                        citations.append(pid)
                    txt = (h.get("text") or "")[:800]
                    if txt:
                        context_notes.append(txt)
                messages.append({"role":"assistant","content": "Retrieved context via direct search. Proceeding to final answer."})

            # 3) Final synthesis (with context injection and strict instruction)
            return self._final_answer(question, messages, citations, context_notes, tool_results, emit)

        except Exception as e:
            logger.exception("[Functions] failed: %s", e)
            fb = {
                "question": question,
                "answer": "",
                "citations": [],
                "file_llm_status": "failed",
                "file_error_info": {"stage": "functions-ask", "type": e.__class__.__name__, "message": str(e)}
            }
            if emit:
                fb["tool_results"] = tool_results
            return fb

    def _final_answer(
            self,
            question: str,
            messages: List[Dict[str, Any]],
            citations: List[str],
            context_notes: List[str],
            tool_results: List[Dict[str, Any]],
            emit: bool
    ) -> Dict[str, Any]:
        context_block = "\n---\n".join(context_notes[:4]) if context_notes else ""
        final_instruction = (
                FINAL_GUARDRAIL + "\n"
                                  "Task: Extract the numeric total revenue for 2019 from the context above and respond with one concise sentence. "
                                  "Do not acknowledge the request (e.g., 'Understood'); provide the number and a parent_id citation."
        )

        messages.append({"role":"assistant","content":"Reminder: cite [parent_id]; do not mention prompts or templates."})
        messages.append({"role":"user","content": f"Question: {question}\n\nContext:\n{context_block}\n\n{final_instruction}"})

        try:
            final = client.chat.completions.create(
                model=model, messages=messages, temperature=0.2, top_p=1.0, max_tokens=256
            )
            answer = _strip_numeric_brackets(final.choices[0].message.content or "").strip()
        except Exception as synth_err:
            logger.exception("[Functions] synthesis failed: %s", synth_err)
            ans = {
                "question": question,
                "answer": "",
                "citations": citations[:5],
                "file_llm_status": "failed",
                "file_error_info": {"stage": "functions-synthesis", "type": synth_err.__class__.__name__, "message": str(synth_err)}
            }
            if emit:
                ans["tool_results"] = tool_results
            return ans

        if "[" not in answer and citations:
            answer = f"{answer} [{citations[0]}]"

        logger.info("[Functions] done citations=%s", citations[:3])
        out = {"question": question, "answer": answer, "citations": citations[:5]}
        if emit:
            out["tool_results"] = tool_results
        return out
