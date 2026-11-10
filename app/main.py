import logging
import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.api import chat, documents, illustrations
from app.config import get_settings

# Init the app 
s = get_settings()
configured_log_level = getattr(logging, s.log_level.upper(), logging.DEBUG)
logging.basicConfig(
    level=configured_log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
request_logger = logging.getLogger("app.requests")
request_logger.setLevel(configured_log_level)

# Init FastAPI app
app = FastAPI(
    title=s.app_name,
    version="0.2.0",
    description="Local Moodle RAG backend using FastAPI, ChromaDB, and Geminis.",
    docs_url=None if s.app_env == "production" else "/docs",
    redoc_url=None if s.app_env == "production" else "/redoc",
    openapi_url=None if s.app_env == "production" else "/openapi.json",
)

# Include router for API endpoints
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(illustrations.router, prefix="/api/v1")

# setup middleware to log requests and responses
@app.middleware("http")
async def log_requests(request, call_next):
    started_at = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    request_logger.debug("Request started: %s %s from %s",
                         request.method, request.url.path, client_host)
    try:
        response = await call_next(request)
    finally:
        duration_ms = (time.perf_counter() - started_at) * 1000
        status_code = response.status_code if "response" in locals() else "error"
        request_logger.debug("Request finished: %s %s -> %s (%.2f ms)",
                             request.method, request.url.path, status_code, duration_ms)
    return response

# route for health check and readiness check
@app.get("/health")
def health():
    return {"status": "ok", "service": s.app_name, "environment": s.app_env}

# route to check if the server is ready to serve request
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