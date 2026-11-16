"""Retrieval helpers shared by the chat endpoint and the RAG graph."""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from app.config import Settings, get_settings
from app.schemas import SourceItem

logger = logging.getLogger("app.api.chat")

NO_CONTEXT_ANSWER = (
    "I couldn't find relevant text in the documents available to you "
    "for this course. Upload or index course materials first, or try a "
    "more specific question."
)

# Retrieval uses Chroma's synchronous client and synchronous embedding SDKs.
# Keep this I/O-heavy work off the event loop and independent of Starlette's
# smaller default worker pool. Tune with RETRIEVAL_WORKERS for the host.
# Created lazily so an app shutdown (lifespan) can stop it and a later
# startup in the same process gets a fresh pool.
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def get_retrieval_executor() -> ThreadPoolExecutor:
    """Return the shared retrieval thread pool, creating it on first use."""
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=max(1, get_settings().retrieval_workers),
                thread_name_prefix="rag-retrieval",
            )
        return _executor


def shutdown_retrieval_executor() -> bool:
    """Stop the pool without waiting for in-flight searches. Returns False if none was running."""
    global _executor
    with _executor_lock:
        executor, _executor = _executor, None
    if executor is None:
        return False
    executor.shutdown(wait=False, cancel_futures=True)
    return True


_warned_uncalibrated: set[tuple[str, str]] = set()


def enabled_only(matches: list[dict]) -> list[dict]:
    """Keep matches whose document is enabled for retrieval (missing flag means enabled)."""
    return [match for match in matches if match["metadata"].get("retrieval_enabled", True)]


def filter_by_relevance(matches: list[dict], settings: Settings, request_id: str) -> list[dict]:
    """Drop chunks too far from the question (cosine distance above the threshold)."""
    if not matches:
        return matches
    threshold = settings.resolved_retrieval_max_distance
    if threshold <= 0:
        embedding = (settings.embedding_provider.lower(), settings.resolved_embedding_model)
        if settings.retrieval_max_distance is None and embedding not in _warned_uncalibrated:
            _warned_uncalibrated.add(embedding)
            logger.warning(
                "chat.retrieval.threshold_disabled embedding=%s/%s reason=uncalibrated_model "
                "(set RETRIEVAL_MAX_DISTANCE to enable)", *embedding,
            )
        return matches
    kept = [m for m in matches if m.get("distance") is None or m["distance"] <= threshold]
    distances = [m["distance"] for m in matches if m.get("distance") is not None]
    logger.info(
        "chat.retrieval.filtered id=%s dropped=%s kept=%s threshold=%.3f best_distance=%s",
        request_id, len(matches) - len(kept), len(kept), threshold,
        f"{min(distances):.3f}" if distances else "n/a",
    )
    return kept


def build_context_and_sources(matches: list[dict]) -> tuple[str, list[SourceItem]]:
    """Number passages [S1..Sn] and group them into one source per document page."""
    passages, sources, source_indexes = [], [], {}
    for i, match in enumerate(matches, 1):
        metadata = match["metadata"]
        page = metadata.get("page", -1)
        page = page if isinstance(page, int) and page > 0 else None
        document_id = str(metadata.get("document_id", ""))
        passages.append(
            f"[S{i}] File: {metadata.get('filename', 'Unknown')}; "
            f"page/slide: {page or 'not available'}\n{match['text']}"
        )
        key = (document_id, page)
        source_index = source_indexes.get(key)
        if source_index is None:
            sources.append(
                SourceItem(
                    document_id=document_id,
                    filename=str(metadata.get("filename", "Unknown")),
                    page=page,
                    chunk_id=match["chunk_id"],
                    excerpt=match["text"][:350],
                    markers=[f"S{i}"],
                )
            )
            source_indexes[key] = len(sources) - 1
        else:
            sources[source_index].markers.append(f"S{i}")
    return "\n\n".join(passages), sources
