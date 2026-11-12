import asyncio
import logging
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api import admin, chat, documents, illustrations
from app.config import get_settings

# Configure application logging before importing/serving requests.
s = get_settings()
configured_log_level = getattr(logging, s.log_level.upper(), logging.INFO)
logging.basicConfig(
    level=configured_log_level,
    format="%(asctime)s %(levelname)s %(name)s [%(threadName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
request_logger = logging.getLogger("app.requests")
request_logger.setLevel(configured_log_level)

app = FastAPI(
    title=s.app_name,
    version="0.2.0",
    description="Local Moodle RAG backend using FastAPI, ChromaDB, and AI providers.",
    docs_url=None if s.app_env == "production" else "/docs",
    redoc_url=None if s.app_env == "production" else "/redoc",
    openapi_url=None if s.app_env == "production" else "/openapi.json",
)

app.include_router(admin.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(illustrations.router, prefix="/api/v1")


@app.on_event("startup")
async def warm_vector_store():
    """Initialize Chroma and run its one-time metadata migration before traffic."""
    started = time.perf_counter()
    request_logger.info("startup.vector_store.start")
    try:
        from app.services.vector_store import get_collection

        collection = await asyncio.to_thread(get_collection)
        count = await asyncio.to_thread(collection.count)
        request_logger.info(
            "startup.vector_store.ready chunks=%s duration_ms=%.2f",
            count,
            (time.perf_counter() - started) * 1000,
        )
    except Exception:
        request_logger.exception(
            "startup.vector_store.failed duration_ms=%.2f",
            (time.perf_counter() - started) * 1000,
        )
        # Keep health/admin endpoints available for diagnosis. /ready reports
        # not_ready and the first retrieval attempt will surface the same error.


@app.middleware("http")
async def log_requests(request, call_next):
    """Attach a correlation ID and log every request without logging secrets."""
    started = time.perf_counter()
    request_id = uuid.uuid4().hex
    status_code = "error"
    request.state.request_id = request_id
    client_host = request.client.host if request.client else "unknown"

    request_logger.info(
        "request.start id=%s method=%s path=%s client=%s",
        request_id,
        request.method,
        request.url.path,
        client_host,
    )

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        request_logger.exception(
            "request.error id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise
    finally:
        request_logger.info(
            "request.finish id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status_code,
            (time.perf_counter() - started) * 1000,
        )


@app.get("/health")
def health():
    return {"status": "ok", "service": s.app_name, "environment": s.app_env}


@app.get("/ready")
def ready():
    try:
        from app.services.vector_store import get_collection

        collection = get_collection()
        return {"status": "ready", "indexed_chunks": collection.count()}
    except Exception:
        logging.exception("Readiness check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "backend not ready"},
        )
