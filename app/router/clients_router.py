from fastapi import APIRouter, HTTPException
from typing import Any, List
from app.config.app_config import AppConfig, AppConfigSingleton
from app.utils.app_logging import get_logger

# Chroma
import chromadb
import logging as pylogging
pylogging.getLogger("chromadb").setLevel(pylogging.CRITICAL)

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# router = APIRouter(tags=["clients"])
# cfg = AppConfig()
# logger = get_logger(cfg)

router = APIRouter(prefix="/clients", tags=["clients"])
cfg = AppConfigSingleton.instance()  # Replaces cfg = AppConfig()
logger = get_logger(cfg)

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/openai/heartbeat", summary="Connectivity check using config (no payload)")
async def test_openai_client():
    if OpenAI is None:
        logger.error("Failed to connect to OpenAI: SDK import not available")
        raise HTTPException(status_code=500, detail="OpenAI SDK not available")
    logger.info("Trying to connect with OpenAI")
    try:
        client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
        models = client.models.list()
        model_ids: List[str] = [m.id for m in getattr(models, "data", [])][:10]
        logger.info("Successfully connected to OpenAI")
        return {
            "ok": True,
            "models_seen": model_ids,
            "default_model": cfg.openai_llm_model
        }
    except Exception as e:
        logger.exception("Failed to connect to OpenAI")
        raise HTTPException(status_code=502, detail=f"OpenAI connectivity failed")

@router.get("/chroma/heartbeat", summary="Chroma heartbeat using configured persistence path")
async def chroma_heartbeat():
    logger.info("Trying to connect with Chroma DB")
    try:
        client = chromadb.PersistentClient(path=cfg.chroma_dir)
        hb: Any = client.heartbeat()
        logger.info("Successfully connected to Chroma DB")
        return {"ok": True, "heartbeat_ns": hb, "chroma_dir": cfg.chroma_dir}
    except Exception:
        logger.exception("Failed to connect to Chroma DB")
        raise HTTPException(status_code=500, detail="Chroma heartbeat failed")
