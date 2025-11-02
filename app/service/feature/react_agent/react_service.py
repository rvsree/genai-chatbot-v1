from typing import Dict, Any, List
import time
from openai import OpenAI
from app.utils.app_logging import get_logger
from app.config.app_config import AppConfig
from app.adapters.feature.fin_analysis_agent.tool_adapters import RetrievalTools
from app.prompts.feature.fin_analysis_agent import fin_analysis_agent_react_prompt
from app.prompts.registry.prompt_registry import PromptRegistry, PromptBundle

cfg = AppConfig()
logger = get_logger(cfg)

# Reuse singleton OpenAI client/model from config
client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
model = cfg.openai_llm_model

registry = PromptRegistry(
    react_bundle=PromptBundle(system=fin_analysis_agent_react_prompt.REACT_SYSTEM, user_template=fin_analysis_agent_react_prompt.REACT_USER_TEMPLATE),
    func_bundle=PromptBundle(system="", user_template="") # not used here
)

class ReactAgent:
    """
    Minimal ReAct: loop of Thought -> Action(tool) -> Observation, capped by max_steps.
    Tools: RetrievalTools.index_lookup, RetrievalTools.vector_search, RetrievalTools.get_chunk
    """
    def __init__(self, max_steps: int = 4):
        self.max_steps = max_steps

    def run(self, question: str) -> Dict[str, Any]:
        logger.info("[ReActV2] begin q='%s'", question)
        traces: List[Dict[str, Any]] = []
        context_notes: List[str] = []
        citations: List[str] = []

        # Seed messages
        messages = [
            {"role": "system", "content": registry.react.system},
            {"role": "user", "content": registry.react.user_template.format(question=question)}
        ]

        for step in range(self.max_steps):
            # Ask the model what to do next (no tools here; we parse its suggestion)
            start = time.perf_counter()
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.2, top_p=1.0, max_tokens=256
            )
            thought = resp.choices[0].message.content or ""
            traces.append({"step": step+1, "thought": thought})
            logger.info("[ReActV2] step=%d thought_len=%d", step+1, len(thought))

            # Heuristic: choose a tool by keywords (simple, deterministic)
            action = None
            if "index_lookup" in thought:
                action = "index_lookup"
            elif "get_chunk" in thought:
                action = "get_chunk"
            else:
                action = "vector_search"

            # Extract naive arguments (very constrained demo); in practice use structured parsing
            args: Dict[str, Any] = {}
            if action == "index_lookup":
                # look for parent_id hints
                for token in thought.split():
                    if "::" not in token and "-" in token:
                        args["parent_id"] = token.strip("[](),.")
                        break
                args.setdefault("parent_id", "tesla-2023")
                result = RetrievalTools.index_lookup(**args)
            elif action == "get_chunk":
                for token in thought.split():
                    if "::chunk::" in token:
                        args["id"] = token.strip("[](),.")
                        break
                args.setdefault("id", "tesla-2023::chunk::0000")
                result = RetrievalTools.get_chunk(**args)
            else:
                args["query"] = question
                args["n_results"] = 5
                result = RetrievalTools.vector_search(**args)

            traces[-1]["action"] = {"name": action, "args": args}
            traces[-1]["observation"] = {"summary": f"keys={list(result.keys())}", "n_hits": len(result.get("hits", [])) if isinstance(result.get("hits"), list) else 1}

            # Collect small context from hits or chunk
            if action == "vector_search":
                for h in result.get("hits", [])[:3]:
                    context_notes.append(h.get("text","")[:400])
                    pid = h.get("parent_id")
                    if pid and pid not in citations:
                        citations.append(pid)
            elif action == "get_chunk" and result.get("found"):
                context_notes.append(result.get("text","")[:600])
                meta = result.get("metadata", {})
                pid = meta.get("parent_id")
                if pid and pid not in citations:
                    citations.append(pid)

            # Simple stopping criterion: if we gathered enough context
            if len(context_notes) >= 2:
                break

            # Feed an observation back to help next step
            messages.append({"role":"assistant", "content": thought})
            messages.append({"role":"user", "content": f"Observation: received {traces[-1]['observation']}. Continue."})

        # Final answer with gathered notes
        final_messages = [
            {"role":"system", "content": registry.react.system},
            {"role":"user", "content": f"{registry.react.user_template.format(question=question)}\n\nContext:\n" + "\n---\n".join(context_notes)}
        ]
        final_resp = client.chat.completions.create(model=model, messages=final_messages, temperature=0.2, top_p=1.0, max_tokens=512)
        answer = final_resp.choices[0].message.content or ""
        logger.info("[ReActV2] done citations=%s", citations[:3])
        return {"question": question, "answer": answer, "citations": citations[:5], "traces": traces}
