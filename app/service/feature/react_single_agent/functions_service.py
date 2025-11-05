# app/service/feature/react_single_agent/functions_service.py
from typing import Dict, Any, Tuple, List, Optional
import re, asyncio
from openai import OpenAI, AuthenticationError, APIConnectionError, RateLimitError
from app.config.app_config import AppConfigSingleton
from app.utils.app_logging import get_logger
from app.prompts.feature.react_single_agent import function_prompts
from app.service.feature.react_single_agent.base.react_base import ReactBaseAgent, AgentError
from app.utils.circuit_breaker import CircuitBreaker, with_retries_async

_cfg = AppConfigSingleton.instance()
_logger = get_logger(_cfg)
_client = OpenAI(api_key=_cfg.openai_api_key, base_url=_cfg.openai_base_url) if _cfg.openai_api_key else None
_MODEL = _cfg.openai_llm_model or _cfg.openai_default_model

_llm_breaker = CircuitBreaker(failure_threshold=3, recovery_time_sec=20.0)

def _is_retryable_llm(err: Exception) -> bool:
    if isinstance(err, AuthenticationError):
        return False
    if isinstance(err, (APIConnectionError, RateLimitError)):
        return True
    msg = str(err).lower()
    return "timeout" in msg or "connection" in msg or "temporarily" in msg

class ReactFunctionCallingAgent(ReactBaseAgent):
    async def synthesize_final_with_meta(
            self,
            variant_query: str,
            query_context: Dict[str, Any],
            context_notes: List[str],
            citations: List[str],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        if not _client:
            raise AgentError("LLM_AUTH", 401, "LLM client not initialized; set OPENAI_API_KEY/base_url/model.")

        temp = temperature if temperature is not None else getattr(_cfg, "openai_llm_temperature", 0.3)
        max_toks = max_tokens if max_tokens is not None else getattr(_cfg, "rag_max_tokens", 256)

        ctx_str = "\n".join(context_notes if isinstance(context_notes, list) else [str(context_notes)])

        messages = [
            {"role":"system","content": function_prompts.FUNCTION_SYSTEM_STRICT},
            {"role":"user","content": function_prompts.FUNCTION_USER_EXTRACT.format(
                question=variant_query,
                context=ctx_str
            )}
        ]

        async def _op():
            return await asyncio.to_thread(
                _client.chat.completions.create,
                model=_MODEL, messages=messages, temperature=temp, top_p=1.0, max_tokens=max_toks
            )

        try:
            resp = await with_retries_async(_op, _is_retryable_llm, _llm_breaker, max_attempts=3, base_backoff=0.4)
        except AuthenticationError as e:
            raise AgentError("LLM_AUTH", 401, f"Authentication failed: {str(e)}")
        except RateLimitError as e:
            raise AgentError("LLM_RATE_LIMIT", 429, f"Rate limit: {str(e)}")
        except APIConnectionError as e:
            raise AgentError("LLM_UNAVAILABLE", 503, f"Connection failure: {str(e)}")
        except Exception as e:
            raise AgentError("LLM_UNAVAILABLE", 503, f"LLM error: {str(e)}")

        answer = (resp.choices[0].message.content or "").strip()

        usage = getattr(resp, "usage", None)
        usage_meta = {}
        if usage:
            usage_meta = {
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or (usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or (usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0)
                }
            }

        if "[parent-id]" in answer:
            if citations:
                answer = answer.replace("[parent-id]", f"[{citations[0]}]")
            else:
                raise AgentError("PLACEHOLDER_CITATIONS", 422, "Model returned placeholder citations; no whitelist available.", {})
        if "[" not in answer or "]" not in answer:
            if citations:
                answer = f"{answer} [{citations[0]}]"
            else:
                raise AgentError("INSUFFICIENT_EVIDENCE", 422, "No citations available to attach.", {})

        meta = {"provider": "openai", "model": _MODEL, "temperature": temp, "status": "success"}
        meta.update(usage_meta)
        return answer, meta
