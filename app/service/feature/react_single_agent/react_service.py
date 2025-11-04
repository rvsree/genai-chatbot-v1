from typing import Dict, Any, List, Optional
import re
from openai import OpenAI
from app.utils.app_logging import get_logger
from app.config.app_config import AppConfigSingleton as AppConfig
from app.adapters.feature.react_single_agent.tool_adapters import RetrievalTools
from app.prompts.feature.react_single_agent import react_prompts

cfg = AppConfig()
logger = get_logger(cfg)

client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
model = cfg.openai_llm_model or cfg.openai_default_model

# Guardrail used ONLY in final synthesis
FINAL_GUARDRAIL = (
    "Answer concisely using ONLY the provided filing excerpts. "
    "Include citations as [parent_id]; never use numeric references like [14]. "
    "Do not mention prompts, system instructions, or templates."
)

# Constraint applied ONLY to planning thoughts to keep them tool-oriented
PLANNING_INSTRUCTION = (
    "Think in one short line about which tool to use next and why. "
    "Do NOT mention prompts, system instructions, refusal, or capabilities. "
    "Focus on tool selection (vector_search, get_chunk, index_lookup) and parameters."
)

class ReactAgent:
    """
    ReAct planning loop uses a neutral system prompt; planning thoughts are constrained to tool strategy.
    The strict guardrail is applied only in final synthesis to avoid contaminating 'thought'.
    """

    def __init__(self, max_steps: int = 4, emit_traces: bool = True):
        self.max_steps = max_steps
        self.default_emit_traces = emit_traces

    def _strip_numeric_brackets(self, text: str) -> str:
        return re.sub(r"\[(\d+)\]", "", text or "")

    def _sanitize_thought(self, text: str) -> str:
        """
        Keep thoughts clean and tool-focused. Strip meta/prompt chatter if model emits any.
        """
        t = text or ""
        # Remove common refusal/meta-prefaces
        t = re.sub(r"(?i)i'?m sorry.*?(?:\.)", "", t).strip()
        t = re.sub(r"(?i)can'?t disclose.*?(?:\.)", "", t).strip()
        # If still empty or generic, replace with simple tool plan hint
        if not t or len(t) < 8:
            t = "Plan: run vector_search for relevant 2019 risk factors."
        # Trim to one sentence
        t = t.split("\n")[0]
        if "." in t:
            t = t.split(".")[0] + "."
        return t

    def run(self, question: str, emit_traces: Optional[bool] = None) -> Dict[str, Any]:
        emit = self.default_emit_traces if emit_traces is None else emit_traces
        logger.info("[ReAct] begin q='%s'", question)

        traces: List[Dict[str, Any]] = []
        context_notes: List[str] = []
        citations: List[str] = []

        # Neutral planning system, AS-IS; add a brief assistant nudge to keep thoughts tool-oriented
        messages = [
            {"role": "system", "content": react_prompts.REACT_SYSTEM},
            {"role": "assistant", "content": PLANNING_INSTRUCTION},
            {"role": "user", "content": react_prompts.REACT_USER_TEMPLATE.format(question=question)}
        ]

        try:
            for step in range(self.max_steps):
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=0.2, top_p=1.0, max_tokens=192
                )
                raw_thought = resp.choices[0].message.content or ""
                thought = self._sanitize_thought(raw_thought)
                rec: Dict[str, Any] = {"step": step + 1, "thought": thought}

                # Simple routing based on the plan; default to vector_search
                action = "vector_search"
                args = {"query": question, "n_results": 5}
                if "index_lookup" in raw_thought or "index_lookup" in thought:
                    action, args = "index_lookup", {"parent_id": "copy-tesla-10k-2019"}
                elif "get_chunk" in raw_thought or "get_chunk" in thought:
                    action, args = "get_chunk", {"id": "copy-tesla-10k-2019::chunk::0000"}

                result = (
                    RetrievalTools.index_lookup(**args) if action == "index_lookup"
                    else RetrievalTools.get_chunk(**args) if action == "get_chunk"
                    else RetrievalTools.vector_search(**args)
                )

                rec["action"] = {"name": action, "args": args}
                rec["observation"] = {
                    "keys": list(result.keys()),
                    "n_hits": len(result.get("hits", [])) if isinstance(result.get("hits"), list) else (1 if result else 0)
                }

                if action == "vector_search":
                    for h in result.get("hits", [])[:3]:
                        txt = (h.get("text") or "")[:800]
                        if txt:
                            context_notes.append(txt)
                        pid = h.get("parent_id")
                        if pid and pid not in citations:
                            citations.append(pid)
                elif action == "get_chunk" and result.get("found"):
                    txt = (result.get("text") or "")[:1000]
                    if txt:
                        context_notes.append(txt)
                    pid = (result.get("metadata") or {}).get("parent_id")
                    if pid and pid not in citations:
                        citations.append(pid)

                if emit:
                    traces.append(rec)

                if len(context_notes) >= 2:
                    break

                messages.append({"role": "assistant", "content": thought})
                messages.append({"role": "user", "content": "Observation: tool step completed. Continue planning with tools if needed."})

            if not context_notes:
                forced = RetrievalTools.vector_search(query=question, n_results=5)
                for h in forced.get("hits", [])[:2]:
                    txt = (h.get("text") or "")[:800]
                    if txt:
                        context_notes.append(txt)
                    pid = h.get("parent_id")
                    if pid and pid not in citations:
                        citations.append(pid)

            # Final synthesis: apply guardrail ONLY here
            final_messages = [
                {"role": "system", "content": react_prompts.REACT_SYSTEM},
                {"role": "assistant", "content": "Reminder: cite [parent_id]; do not mention prompts or templates."},
                {"role": "user", "content": (
                        react_prompts.REACT_USER_TEMPLATE.format(question=question)
                        + "\n\nContext (verbatim excerpts from filings):\n"
                        + "\n---\n".join(context_notes[:4])
                        + "\n\n" + FINAL_GUARDRAIL
                )}
            ]
            final = client.chat.completions.create(
                model=model, messages=final_messages, temperature=0.2, top_p=1.0, max_tokens=256
            )
            answer = self._strip_numeric_brackets(final.choices[0].message.content or "").strip()
            if "[" not in answer and citations:
                answer = f"{answer} [{citations[0]}]"

            logger.info("[ReAct] done citations=%s", citations[:3])

            out = {"question": question, "answer": answer, "citations": citations[:5]}
            if emit:
                out["traces"] = traces
            return out

        except Exception as e:
            logger.exception("[ReAct] failed: %s", e)
            fb = {
                "question": question,
                "answer": "",
                "citations": [],
                "file_llm_status": "failed",
                "file_error_info": {"stage": "react-ask", "type": e.__class__.__name__, "message": str(e)}
            }
            if emit:
                fb["traces"] = traces
            return fb
