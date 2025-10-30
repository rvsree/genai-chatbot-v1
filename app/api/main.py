from fastapi import FastAPI
from app.config.app_config import AppConfig, ensure_data_dirs
from app.utils.app_logging import get_logger

# Routers
from app.router.clients_router import router as clients_router
from app.router.doc_indexing_router import indexing_router
from app.router.rag_search_router import rag_router

app = FastAPI(title="GL RAG FastAPI", version="0.1.1")
cfg = AppConfig()
logger = get_logger(cfg)

ensure_data_dirs(cfg)
logger.info("App starting with data_dir=%s, chroma_dir=%s", cfg.data_dir, cfg.chroma_dir)

@app.get("/doc-indexing/health")
async def health():
    return {"status": "ok", "app": app.title, "version": app.version}

# app.include_router(clients_router, prefix="/doc-indexing/clients")
# app.include_router(documents_router, prefix="/doc-indexing")
app.include_router(clients_router, prefix="/clients")
app.include_router(indexing_router)   # exposes /doc-indexing/*
app.include_router(rag_router)        # exposes /rag-search/*