from typing import Dict, Any, List
import json
from openai import OpenAI
from app.utils.app_logging import get_logger
from app.config.app_config import AppConfig, AppConfigSingleton
from app.adapters.feature.fin_analysis_agent.tool_adapters import RetrievalTools

cfg = AppConfigSingleton.instance()
logger = get_logger(cfg)
client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
model = cfg.openai_llm_model

FUNCTIONS: List[Dict[str, Any]] = [
    {
        "name": "vector_search",
        "description": "Search the indexed filings and return top chunks with metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "n_results": {"type": "integer", "default": 5},
                "advisor_id": {"type": "string"},
                "client_id": {"type": "string"},
                "doc_type": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_chunk",
        "description": "Fetch a specific chunk by id and return full text and metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"}
            },
            "required": ["id"]
        }
    }
]

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

def _call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == "vector_search":
        return RetrievalTools.vector_search(**arguments)
    if name == "get_chunk":
        return RetrievalTools.get_chunk(**arguments)
    return {"error": f"unknown tool {name}"}

class FunctionCalling:
    def run(self, question: str) -> Dict[str, Any]:
        logger.info("[FunctionsV2] begin q='%s'", question)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a structured financial assistant. Use functions when available."},
            {"role": "user", "content": f"Question: {question}"}
        ]

        # First model turn permitting function calls
        first = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[{"type": "function", "function": f} for f in FUNCTIONS],
            tool_choice="auto",
            temperature=0.2,
            top_p=1.0,
            max_tokens=256
        )

        tool_results: List[Dict[str, Any]] = []
        citations: List[str] = []

        # Append the assistant message that requested tools (required for pairing)
        assistant_msg = first.choices[0].message
        messages.append({"role": "assistant", "content": assistant_msg.content or "", "tool_calls": assistant_msg.tool_calls or []})

        # For each tool_call in this assistant message, execute and append a paired tool message
        for tc in assistant_msg.tool_calls or []:
            name = tc.function.name
            raw_args = tc.function.arguments
            args = _to_kwargs(raw_args)
            logger.info("[FunctionsV2] tool_call name=%s args=%s", name, args)

            result = _call_tool(name, args)
            tool_results.append({"name": name, "args": args, "result_keys": list(result.keys())})

            if name == "vector_search":
                for h in result.get("hits", [])[:3]:
                    pid = h.get("parent_id")
                    if pid and pid not in citations:
                        citations.append(pid)
            elif name == "get_chunk":
                meta = result.get("metadata", {})
                pid = meta.get("parent_id")
                if pid and pid not in citations:
                    citations.append(pid)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": str(result)[:3500]
            })

        # Final answer turn (now that tool messages are paired with the assistant that called them)
        final = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            top_p=1.0,
            max_tokens=512
        )
        answer = final.choices[0].message.content or ""
        logger.info("[FunctionsV2] done citations=%s", citations[:3])
        return {
            "question": question,
            "answer": answer,
            "citations": citations[:5],
            "tool_results": tool_results
        }
