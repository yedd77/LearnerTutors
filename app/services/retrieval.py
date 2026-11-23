"""Retrieval helpers shared by the chat endpoint and the RAG graph."""
from collections import Counter
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from app.config import Settings, get_settings
from app.schemas import SourceItem
from app.services import vector_store

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


def chunk_count(matches: list[dict]) -> int:
    """Chunks behind the passages: a merged passage counts every chunk it contains."""
    return sum(int(match.get("chunk_count", 1)) for match in matches)


def document_scope(matches: list[dict]) -> tuple[str, str | None] | None:
    """The document (and heading) most hits come from, if at least half do.

    Lets /summarize and /studyguide read that document or section in order
    instead of scattered similar chunks. None when hits are spread out or were
    indexed before chunk positions were stored.
    """
    positioned = [m for m in matches if isinstance(m["metadata"].get("chunk_index"), int)]
    if not positioned:
        return None
    document_id, count = Counter(str(m["metadata"].get("document_id")) for m in positioned).most_common(1)[0]
    if count * 2 < len(matches):
        return None
    headings = Counter(str(m["metadata"].get("heading") or "") for m in positioned
                       if str(m["metadata"].get("document_id")) == document_id)
    heading, heading_count = headings.most_common(1)[0]
    return document_id, (heading if heading and heading_count * 2 >= count else None)


def document_passages(matches: list[dict], course_id: int, max_chars: int) -> list[dict] | None:
    """Read the dominant document/section of ``matches`` in order, one passage per page.

    Returns None (use the normal passages) when there is no dominant document
    or it does not fit in ``max_chars``. Reads go through the vector store's
    access filter.
    """
    scope = document_scope(matches)
    if scope is None:
        return None
    document_id, heading = scope
    chunks = vector_store.ordered_document_chunks(document_id, course_id, heading)
    if not chunks or sum(len(c["text"]) for c in chunks) > max_chars:
        return None
    passages: list[dict] = []
    for chunk in chunks:
        if passages and passages[-1]["metadata"].get("page") == chunk["metadata"].get("page"):
            passages[-1]["text"] = vector_store.merge_chunk_texts(passages[-1]["text"], chunk["text"])
            passages[-1]["chunk_count"] += 1
        else:
            passages.append({**chunk, "distance": None, "chunk_count": 1})
    return passages


def build_context_and_sources(matches: list[dict]) -> tuple[str, list[SourceItem]]:
    """Number passages [S1..Sn] and group them into one source per document page."""
    passages, sources, source_indexes = [], [], {}
    for i, match in enumerate(matches, 1):
        metadata = match["metadata"]
        page = metadata.get("page", -1)
        page = page if isinstance(page, int) and page > 0 else None
        document_id = str(metadata.get("document_id", ""))
        heading = str(metadata.get("heading") or "")
        passages.append(
            f"[S{i}] File: {metadata.get('filename', 'Unknown')}; "
            f"page/slide: {page or 'not available'}"
            + (f"; section: {heading}" if heading else "")
            + f"\n{match['text']}"
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
