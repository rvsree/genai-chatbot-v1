from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi import Path as FPath
from pathlib import Path
import time
from typing import Optional
from app.config.app_config import AppConfig
from app.utils.app_logging import get_logger
from app.service.chroma_client_service import ChromaClientService
from app.service.chunked_indexer_service import ChunkedIndexerService
from app.models.indexing_models import (
    IndexResponse, SaveMetadataRequest, SaveMetadataResponse, DeleteResponse
)

cfg = AppConfig()
logger = get_logger(cfg)
chromaClientService = ChromaClientService()
indexer = ChunkedIndexerService(chromaClientService)
indexing_router = APIRouter(prefix="/doc-indexing", tags=["doc-indexing"])

def _safe_filename(name: str) -> str:
    base = "".join(c for c in (name or "") if c.isalnum() or c in ("-", "_", ".")).strip()
    return base or "upload.pdf"

def _validate_single_file_part(request: Request) -> Optional[str]:
    ctype = request.headers.get("content-type", "")
    if "multipart/form-data" not in ctype:
        return "Expected multipart/form-data; Content-Type incorrect or missing boundary"
    return None

@indexing_router.get("/count")
async def count() -> dict:
    c = chromaClientService.count()
    logger.info("[Indexing] Count=%d", c)
    return {"collection_count_after": c}

@indexing_router.get("/list_local_pdfs")
async def list_local_pdfs() -> dict:
    base = Path(cfg.documents_dir)
    pdfs = [p.name for p in base.glob("*.pdf")]
    logger.info("[Indexing] List PDFs n=%d dir=%s", len(pdfs), base)
    return {"folder": str(base), "pdfs": pdfs, "files_count": len(pdfs)}

@indexing_router.post("/index", response_model=IndexResponse)
async def index(
        request: Request,
        advisor_id: str = Form(...),
        client_id: str = Form(...),
        doc_type: str = Form(...),
        file_version: str = Form(...),
        strategy: str = Form(...),
        file_type: str = Form(...),
        document_id: Optional[str] = Form(None),
        files: UploadFile = File(..., description="PDF file")
):
    start = time.perf_counter()
    logger.info("[Indexing] Begin index advisor=%s client=%s doc_type=%s", advisor_id, client_id, doc_type)
    prelim_error = _validate_single_file_part(request)
    if prelim_error:
        logger.error("[Indexing] Multipart error: %s", prelim_error)
        return IndexResponse(
            parent_id=document_id or "", file_name="", file_version=file_version, file_type=file_type,
            files_count=0, chunks_indexed=0, collection_count_after=chromaClientService.count(),
            file_index_status="failed", file_llm_status="not_applicable", file_index_lapse_time=0.0,
            file_error_info={"stage":"doc-indexing","type":"MultipartError","message":prelim_error}
        )

    try:
        raw_bytes = await files.read()
        if not raw_bytes:
            logger.error("[Indexing] Empty file stream for 'files'")
            return IndexResponse(
                parent_id=document_id or "", file_name=files.filename or "", file_version=file_version, file_type=file_type,
                files_count=0, chunks_indexed=0, collection_count_after=chromaClientService.count(),
                file_index_status="failed", file_llm_status="not_applicable", file_index_lapse_time=0.0,
                file_error_info={"stage":"doc-indexing","type":"EmptyFile","message":"No file content received for 'files' (duplicate keys?)"}
            )

        file_name = _safe_filename(files.filename or "upload.pdf")
        dest = Path(cfg.documents_dir) / file_name
        dest.write_bytes(raw_bytes)
        logger.info("[Indexing] File saved path=%s size=%d", dest, len(raw_bytes))

        parent_id = (document_id or Path(file_name).stem)
        existing_ids = chromaClientService.get_ids_by_parent(parent_id)
        if existing_ids:
            lapse = round((time.perf_counter() - start) * 1000, 2)
            logger.info("[Indexing] Skipped existing parent=%s chunks=%d", parent_id, len(existing_ids))
            return IndexResponse(
                parent_id=parent_id, file_name=file_name, file_version=file_version, file_type=file_type,
                files_count=1, chunks_indexed=0, existing_chunks=len(existing_ids),
                collection_count_after=chromaClientService.count(), file_index_status="skipped",
                file_llm_status="not_applicable", file_index_lapse_time=lapse,
                file_error_info={"stage":"doc-indexing","type":"AlreadyIndexed","message":"parent_id exists; use /doc-indexing/reindex"}
            )

        base_meta = {
            "advisor_id": advisor_id, "client_id": client_id, "doc_type": doc_type,
            "version": file_version, "strategy": strategy, "file_type": file_type,
            "document_id": parent_id
        }
        parent_id, chunks = indexer.index_pdf_path(dest, base_meta)
        lapse = round((time.perf_counter() - start) * 1000, 2)
        logger.info("[Indexing] Success parent=%s chunks=%d lapse_ms=%.2f", parent_id, chunks, lapse)

        return IndexResponse(
            parent_id=parent_id, file_name=file_name, file_version=file_version, file_type=file_type,
            files_count=1, chunks_indexed=chunks, collection_count_after=chromaClientService.count(),
            file_index_status="success", file_llm_status="not_applicable", file_index_lapse_time=lapse
        )
    except Exception as e:
        lapse = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("[Indexing] Failed: %s lapse_ms=%.2f", e, lapse)
        return IndexResponse(
            parent_id=document_id or "", file_name=files.filename or "", file_version=file_version, file_type=file_type,
            files_count=1, chunks_indexed=0, collection_count_after=chromaClientService.count(),
            file_index_status="failed", file_llm_status="not_applicable", file_index_lapse_time=lapse,
            file_error_info={"stage":"doc-indexing","type":e.__class__.__name__,"message":str(e)}
        )

@indexing_router.post("/reindex", response_model=IndexResponse)
async def reindex(
        filename: str = Form(...), advisor_id: str = Form(...), client_id: str = Form(...),
        doc_type: str = Form(...), file_version: str = Form(...), strategy: str = Form(...),
        file_type: str = Form("pdf"), document_id: Optional[str] = Form(None)
):
    start = time.perf_counter()
    logger.info("[Indexing] Begin reindex filename=%s", filename)
    try:
        path = Path(cfg.documents_dir) / _safe_filename(filename)
        if not path.exists():
            logger.error("[Indexing] Reindex file not found: %s", path)
            raise HTTPException(status_code=404, detail="File not found in data/documents")

        parent_id = (document_id or path.stem)
        base_meta = {
            "advisor_id": advisor_id, "client_id": client_id, "doc_type": doc_type,
            "version": file_version, "strategy": strategy, "file_type": file_type,
            "document_id": parent_id
        }

        removed = chromaClientService.delete_by_parent(parent_id)
        logger.info("[Indexing] Reindex removed parent=%s chunks=%d", parent_id, removed)
        parent_id, chunks = indexer.index_pdf_path(path, base_meta)
        lapse = round((time.perf_counter() - start) * 1000, 2)
        logger.info("[Indexing] Reindex success parent=%s chunks=%d lapse_ms=%.2f", parent_id, chunks, lapse)

        return IndexResponse(
            parent_id=parent_id, file_name=path.name, file_version=file_version, file_type=file_type,
            files_count=1, replaced_chunks=removed, chunks_indexed=chunks, collection_count_after=chromaClientService.count(),
            file_index_status="success", file_llm_status="not_applicable", file_index_lapse_time=lapse
        )
    except HTTPException:
        raise
    except Exception as e:
        lapse = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("[Indexing] Reindex failed: %s lapse_ms=%.2f", e, lapse)
        return IndexResponse(
            parent_id=document_id or "", file_name=filename, file_version=file_version, file_type=file_type,
            files_count=1, chunks_indexed=0, collection_count_after=chromaClientService.count(),
            file_index_status="failed", file_llm_status="not_applicable", file_index_lapse_time=lapse,
            file_error_info={"stage":"doc-indexing","type":e.__class__.__name__,"message":str(e)}
        )

@indexing_router.delete("/delete/{id}", response_model=DeleteResponse)
async def delete(id: str = FPath(...)):
    try:
        if "::chunk::" in id:
            logger.info("[Indexing] Delete single id=%s", id)
            deleted = chromaClientService.delete(id)
            msg = "Deleted 1 record" if deleted else "No record found for given id"
            logger.info("[Indexing] Delete single result=%s", msg)
            return DeleteResponse(
                id=id, scope="single", deleted=deleted, collection_count_after=chromaClientService.count(), message=msg
            )

        logger.info("[Indexing] Delete by parent id=%s", id)
        removed = chromaClientService.delete_by_parent(id)
        msg = "Deleted all chunks for parent" if removed else "No chunks found for given parent"
        logger.info("[Indexing] Delete by parent result=%s (removed=%d)", msg, removed)
        return DeleteResponse(
            id=id, scope="parent", deleted=removed, collection_count_after=chromaClientService.count(), message=msg
        )
    except Exception as e:
        logger.exception("[Indexing] Delete failed id=%s: %s", id, e)
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

@indexing_router.post("/save_metadata", response_model=SaveMetadataResponse)
async def save_metadata(req: SaveMetadataRequest):
    updated = chromaClientService.save_metadata(req.id, req.metadata or {})
    logger.info("[Indexing] Save metadata ok id=%s", req.id)
    return SaveMetadataResponse(id=req.id, metadata=updated)

@indexing_router.get("/get_metadata/{id}")
async def get_metadata(id: str = FPath(...)) -> dict:
    res = chromaClientService.get(id)
    if not res.get("ids"):
        raise HTTPException(status_code=404, detail="Document id not found")
    return {"id": id, "metadata": res["metadatas"][0]}
