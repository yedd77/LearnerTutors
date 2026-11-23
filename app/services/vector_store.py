"""ChromaDB storage: one collection per embedding model, course-scoped search.

Every chunk stores ``document_id``, ``filename``, ``course_id``, ``owner_id``,
``visibility`` ("course"), ``retrieval_enabled``, ``page``, ``heading`` and
``chunk_index`` (its position in the document; absent on chunks indexed before
it existed). Chunks are embedded with their document title and heading in
front, so short chunks still carry their topic. Search always filters on
course, retrieval flag and visibility.
"""
import hashlib
import logging
import math
from pathlib import Path
import re
import threading
import time
from typing import Optional
import chromadb
from app.config import get_settings
from app.services import keyword_index
from app.services.embedding_provider import embed_texts

_client = None
_collection = None
_collection_identity = None     # embedding identity the open collection belongs to
_retrieval_metadata_migrated = False 
_lock = threading.Lock()
logger = logging.getLogger(__name__)

def reset_vector_store_cache() -> None:
    """Reopen Chroma on the next operation after embedding configuration changes."""
    global _client, _collection, _collection_identity, _retrieval_metadata_migrated
    with _lock:
        _client = None
        _collection = None
        _collection_identity = None
        _retrieval_metadata_migrated = False
    keyword_index.invalidate_course()



class DocumentCourseConflictError(ValueError):
    """Raised when a document ID is already owned by another course."""

# Return the persistent ChromaDB collection, initializing it if necessary.
def get_collection():
    """Return the persistent ChromaDB collection for the active embedding identity.

    Moodle can change the embedding provider/model at runtime, and every worker
    notices through ``get_settings()``; the collection is reopened when the
    identity no longer matches the one it was opened for.
    """
    global _client, _collection, _collection_identity, _retrieval_metadata_migrated

    with _lock:
        settings = get_settings()
        if _collection is not None and _collection_identity != settings.embedding_identity:
            logger.info("Chroma collection switch: embedding configuration changed.")
            _collection = None
            _retrieval_metadata_migrated = False
        if _collection is None:
            initialization_started = time.perf_counter()
            _client = chromadb.PersistentClient(
                path=settings.chroma_absolute_path
            )
            # Keep each embedding space isolated so vectors from different
            # models are never compared against one another.
            embedding_provider = settings.embedding_provider.lower()
            embedding_model = settings.resolved_embedding_model
            identity = settings.embedding_identity
            suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
            model_slug = re.sub(r"[^a-zA-Z0-9_]", "_", embedding_model)[:16].strip("_")
            collection_name = f"{settings.chroma_collection[:25]}_{embedding_provider}_{model_slug}_{suffix}"[:63]
            _collection = _client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_provider": embedding_provider,
                    "embedding_model": embedding_model,
                },
            )
            _collection_identity = identity
            logger.info(
                "Chroma collection opened name=%s provider=%s model=%s duration_ms=%.2f",
                collection_name,
                embedding_provider,
                embedding_model,
                (time.perf_counter() - initialization_started) * 1000,
            )

        # Preserve retrieval access for chunks indexed before this feature.
        if not _retrieval_metadata_migrated:
            migration_started = time.perf_counter()
            logger.info("Chroma metadata migration started.")
            _migrate_retrieval_metadata(_collection)
            _retrieval_metadata_migrated = True
            logger.info(
                "Chroma metadata migration finished duration_ms=%.2f",
                (time.perf_counter() - migration_started) * 1000,
            )

    return _collection

# Migrate existing chunks in the collection to ensure they have a retrieval_enabled flag.
def _migrate_retrieval_metadata(collection):
    """Treat existing chunks without a flag as enabled."""
    offset = 0
    batch_size = 1000

    while True:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )

        ids = result.get("ids", [])
        metadatas = result.get("metadatas") or []

        if not ids:
            break

        update_ids = []
        update_metadatas = []

        for chunk_id, metadata in zip(ids, metadatas):
            metadata = dict(metadata or {})

            if "retrieval_enabled" not in metadata:
                metadata["retrieval_enabled"] = True
                update_ids.append(chunk_id)
                update_metadatas.append(metadata)

        if update_ids:
            collection.update(
                ids=update_ids,
                metadatas=update_metadatas,
            )

        offset += len(ids)

# Generate a unique ID for a chunk based on its document ID, index, and text.
def _chunk_id(document_id: str, index: int, text: str) -> str:
    """Deterministic chunk ID: ``<document_id>_<index>_<hash of id, index and text>``."""
    digest = hashlib.sha256(
        f"{document_id}:{index}:{text}".encode()
    ).hexdigest()[:20]

    return f"{document_id}_{index}_{digest}"

def _document_title(filename: str) -> str:
    """A readable title from a file name: ``week_3-pharmacogenomics.pdf`` -> ``week 3 pharmacogenomics``."""
    return " ".join(re.split(r"[_\-\s]+", Path(filename).stem)).strip()


def _embedding_text(filename: str, heading: str, text: str) -> str:
    """Contextual text that is embedded for a chunk (the stored text stays plain)."""
    lines = [f"Document: {_document_title(filename)}"]
    if heading:
        lines.append(f"Section: {heading}")
    return "\n".join(lines) + "\n\n" + text

# Visibility stored on every chunk. Personal (student-owned) uploads were removed, so
# course-wide documents are the only kind that can be indexed or retrieved.
COURSE_VISIBILITY = "course"

# Index a document in the vector store.
def index_document(
    document_id: str,
    filename: str,
    course_id: int,
    owner_id: int,
    chunks: list[dict],
    retrieval_enabled: Optional[bool] = None,
) -> int:
    """Embed and store a document's chunks, replacing any previous version; return the count.

    Raises DocumentCourseConflictError if ``document_id`` already belongs to a
    different course. ``retrieval_enabled=None`` keeps the previous flag (or
    True for a new document). If embedding or storage fails part-way, the
    partially written chunks are removed before the error is re-raised.
    """
    collection = get_collection()
    document_filter = {"document_id": str(document_id)}
    existing = collection.get(
        where=document_filter,
        limit=1,
        include=["metadatas"],
    )
    existing_metadatas = existing.get("metadatas") or []
    if existing_metadatas:
        existing_course_id = existing_metadatas[0].get("course_id")
        if existing_course_id != int(course_id):
            raise DocumentCourseConflictError(
                f"Document '{document_id}' belongs to another course."
            )

    where = {
        "$and": [
            document_filter,
            {"course_id": int(course_id)},
        ]
    }

    if retrieval_enabled is None:
        retrieval_enabled = (
            bool(existing_metadatas[0].get("retrieval_enabled", True))
            if existing_metadatas
            else True
        )

    collection.delete(where=where)

    total = 0

    try:
        # Match the embedding provider's 100-input request batch to avoid
        # paying a full network round-trip for multiple small Chroma writes.
        for offset in range(0, len(chunks), 100):
            batch = chunks[offset:offset + 100]
            vectors = embed_texts(
                [_embedding_text(filename, chunk.get("heading") or "", chunk["text"]) for chunk in batch],
                task_type="RETRIEVAL_DOCUMENT",
            )

            ids = []
            metadatas = []

            for i, chunk in enumerate(batch, offset):
                ids.append(
                    _chunk_id(document_id, i, chunk["text"])
                )

                metadatas.append({
                    "document_id": str(document_id),
                    "filename": filename[:255],
                    "course_id": int(course_id),
                    "owner_id": int(owner_id),
                    "visibility": COURSE_VISIBILITY,
                    "retrieval_enabled": bool(retrieval_enabled),
                    "page": (
                        int(chunk["page"])
                        if chunk.get("page")
                        else -1
                    ),
                    "heading": str(chunk.get("heading") or "")[:200],
                    "chunk_index": int(chunk.get("chunk_index", i)),
                })

            collection.add(
                ids=ids,
                documents=[chunk["text"] for chunk in batch],
                embeddings=vectors,
                metadatas=metadatas,
            )

            total += len(batch)
    except Exception:
        collection.delete(where=where)
        raise
    finally:
        keyword_index.invalidate_course(course_id)

    return total

# Update the retrieval status of a document's chunks.
def set_document_retrieval(
    document_id: str,
    course_id: int,
    enabled: bool,
) -> int:
    """Enable or disable retrieval for every chunk of a document."""
    collection = get_collection()

    result = collection.get(
        where={
            "$and": [
                {"document_id": str(document_id)},
                {"course_id": int(course_id)},
            ]
        },
        include=["metadatas"],
    )

    ids = result.get("ids", [])
    metadatas = result.get("metadatas") or []

    # Update in slices: Chroma rejects a single write above its max batch size.
    step = 1000
    for start in range(0, len(ids), step):
        collection.update(
            ids=ids[start:start + step],
            metadatas=[
                {**dict(m or {}), "retrieval_enabled": bool(enabled)}
                for m in metadatas[start:start + step]
            ],
        )
    keyword_index.invalidate_course(course_id)

    return len(ids)

# Get the retrieval status of a document's chunks.
def get_document_retrieval_status(document_id: str) -> dict:
    """Report how many chunks of a document are searchable (for verification)."""
    collection = get_collection()

    result = collection.get(
        where={"document_id": str(document_id)},
        include=["metadatas"],
    )

    metadatas = result.get("metadatas") or []
    enabled = sum(1 for m in metadatas if (m or {}).get("retrieval_enabled", True))

    return {
        "total_chunks": len(metadatas),
        "enabled_chunks": enabled,
        "disabled_chunks": len(metadatas) - enabled,
    }

# Generate a report of all documents in the vector store and their retrieval status.
def get_retrieval_report() -> list[dict]:
    """List every document in ChromaDB with its searchable chunk counts.

    Compare this with Moodle's local_ragassistant_documents table to find
    vectors that are still enabled but not managed (or mismatched) in Moodle.
    """
    collection = get_collection()
    docs: dict[str, dict] = {}
    offset = 0

    while True:
        result = collection.get(limit=1000, offset=offset, include=["metadatas"])
        ids = result.get("ids", [])

        if not ids:
            break

        for metadata in result.get("metadatas") or []:
            metadata = metadata or {}
            doc_id = str(metadata.get("document_id", "?"))
            entry = docs.setdefault(doc_id, {
                "document_id": doc_id,
                "filename": metadata.get("filename"),
                "course_id": metadata.get("course_id"),
                "visibility": metadata.get("visibility"),
                "total_chunks": 0,
                "enabled_chunks": 0,
            })
            entry["total_chunks"] += 1
            if metadata.get("retrieval_enabled", True):
                entry["enabled_chunks"] += 1

        offset += len(ids)

    return sorted(
        docs.values(),
        key=lambda d: (
            d["course_id"] if isinstance(d["course_id"], int) else 0,
            int(d["document_id"]) if d["document_id"].isdigit() else 10**9,
        ),
    )

# Delete every vector belonging to a document.
def delete_document(document_id: str, course_id: int) -> int:
    """Delete every vector belonging to a document."""
    collection = get_collection()

    found = collection.get(
        where={
            "$and": [
                {"document_id": str(document_id)},
                {"course_id": int(course_id)},
            ]
        },
        include=[],
    )

    ids = found.get("ids", [])

    if ids:
        collection.delete(ids=ids)
    keyword_index.invalidate_course(course_id)

    return len(ids)

# Search only enabled chunks the user is allowed to access.
def search_course(
    question: str,
    course_id: int,
    include_course_documents: bool,
    top_k: int,
) -> list[dict]:
    """Search only enabled chunks the user is allowed to access.

    Returns up to ``top_k`` (max 20) rows ``{"chunk_id", "text", "metadata",
    "distance"}``, nearest first; ``distance`` is cosine distance (lower is closer).
    """
    # Course documents are the only retrievable kind. When the caller excludes
    # them there is nothing the user may search, so do not query at all.
    if not include_course_documents:
        return []

    collection = get_collection()

    # Chunks indexed as "personal" before student uploads were removed fail the
    # visibility clause, so they can never be returned to anyone.
    where = {
        "$and": [
            {"course_id": int(course_id)},
            {"retrieval_enabled": True},
            {"visibility": COURSE_VISIBILITY},
        ]
    }

    embedding_started = time.perf_counter()
    embedding = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
    logger.info(
        "Retrieval query embedding finished provider=%s model=%s duration_ms=%.2f",
        get_settings().embedding_provider,
        get_settings().resolved_embedding_model,
        (time.perf_counter() - embedding_started) * 1000,
    )

    query_started = time.perf_counter()
    result = collection.query(
        query_embeddings=[embedding],
        n_results=max(1, min(top_k, 20)),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    logger.info(
        "Retrieval query finished course=%s results=%s duration_ms=%.2f",
        course_id, len(result["ids"][0]), (time.perf_counter() - query_started) * 1000,
    )

    rows = []

    for chunk_id, document, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        rows.append({
            "chunk_id": chunk_id,
            "text": document,
            "metadata": metadata or {},
            "distance": distance,
        })

    return rows


# --------------------------------------------------------------------------- hybrid retrieval
# Reciprocal rank fusion constant (the usual 60): fuses ranked lists without
# having to make BM25 scores and cosine distances comparable.
RRF_K = 60
# Two chunks whose word sets overlap at least this much are duplicates (e.g. the
# same file uploaded twice, or repeated slide headers); only the better one is kept.
DUPLICATE_JACCARD = 0.8
MAX_QUERIES = 3
MAX_CANDIDATES = 50


def _course_where(course_id: int, *extra: dict) -> dict:
    """The access filter every retrieval read uses, plus optional extra clauses."""
    return {"$and": [
        {"course_id": int(course_id)},
        {"retrieval_enabled": True},
        {"visibility": COURSE_VISIBILITY},
        *extra,
    ]}


def _cosine_distance(a, b) -> float:
    """Cosine distance (1 - cosine similarity), the metric of the Chroma collection."""
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        x, y = float(x), float(y)
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if not norm_a or not norm_b:
        return 1.0
    return 1.0 - dot / math.sqrt(norm_a * norm_b)


def _keyword_text(document: str, metadata: dict) -> str:
    """What BM25 indexes: file title, heading and chunk text."""
    return " ".join(filter(None, (
        _document_title(str(metadata.get("filename", ""))), str(metadata.get("heading", "")), document,
    )))


def _load_course_corpus(collection, course_id: int) -> tuple[list[str], list[str]]:
    """All searchable chunks of a course, for the keyword index (paged reads)."""
    ids: list[str] = []
    texts: list[str] = []
    offset = 0
    while True:
        result = collection.get(where=_course_where(course_id), limit=2000, offset=offset,
                                include=["documents", "metadatas"])
        batch = result.get("ids", [])
        if not batch:
            break
        for chunk_id, document, metadata in zip(batch, result.get("documents") or [],
                                                result.get("metadatas") or []):
            ids.append(chunk_id)
            texts.append(_keyword_text(document or "", metadata or {}))
        offset += len(batch)
    return ids, texts


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def hybrid_search(
    queries: list[str],
    course_id: int,
    include_course_documents: bool,
    top_k: int,
) -> list[dict]:
    """Dense + BM25 search over the user's course, fused with reciprocal rank fusion.

    Each of up to 3 queries (the search query plus sub-queries) runs a dense and
    a keyword search for a candidate pool of ``max(RETRIEVAL_CANDIDATES, 2*top_k)``
    (at most 50). Results are fused, near-duplicates are dropped, and every
    candidate gets ``distance`` = its smallest cosine distance to any query, so
    the relevance threshold applies to keyword hits too. Returns the pool,
    best first; the caller narrows it to ``top_k``.

    Rows are ``{"chunk_id", "text", "metadata", "distance", "score"}``. Every row
    is read from Chroma with the course/retrieval/visibility filter: keyword
    ranking alone never makes a chunk visible.
    """
    if not include_course_documents:
        return []
    queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))[:MAX_QUERIES]
    if not queries:
        return []
    settings = get_settings()
    pool = max(1, min(MAX_CANDIDATES, max(settings.retrieval_candidates, 2 * top_k)))
    collection = get_collection()
    where = _course_where(course_id)

    started = time.perf_counter()
    query_vectors = embed_texts(queries, task_type="RETRIEVAL_QUERY")
    keyword = keyword_index.get_course_index(
        settings.embedding_identity, course_id,
        lambda: _load_course_corpus(collection, course_id),
    )

    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for query, vector in zip(queries, query_vectors):
        result = collection.query(query_embeddings=[vector], n_results=pool, where=where,
                                  include=["documents", "metadatas", "embeddings"])
        dense_ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        embeddings = result["embeddings"][0] if result.get("embeddings") is not None else []
        for rank, chunk_id in enumerate(dense_ids, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            if chunk_id not in rows:
                rows[chunk_id] = {"chunk_id": chunk_id, "text": documents[rank - 1],
                                  "metadata": metadatas[rank - 1] or {},
                                  "embedding": embeddings[rank - 1]}
        for rank, (chunk_id, _) in enumerate(keyword.search(query, pool), 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(scores, key=scores.get, reverse=True)[:pool]
    missing = [chunk_id for chunk_id in ranked if chunk_id not in rows]
    if missing:
        # Keyword-only hits: re-read with the access filter (drops anything stale).
        fetched = collection.get(ids=missing, where=where,
                                 include=["documents", "metadatas", "embeddings"])
        for i, chunk_id in enumerate(fetched.get("ids", [])):
            rows[chunk_id] = {"chunk_id": chunk_id, "text": fetched["documents"][i],
                              "metadata": fetched["metadatas"][i] or {},
                              "embedding": fetched["embeddings"][i]}

    results: list[dict] = []
    kept_words: list[set[str]] = []
    for chunk_id in ranked:
        row = rows.get(chunk_id)
        if row is None:
            continue
        words = _word_set(row["text"])
        if any(words and len(words & other) / len(words | other) >= DUPLICATE_JACCARD
               for other in kept_words):
            continue
        kept_words.append(words)
        embedding = row.pop("embedding")
        row["distance"] = (min(_cosine_distance(vector, embedding) for vector in query_vectors)
                           if embedding is not None and len(embedding) else None)
        row["score"] = scores[chunk_id]
        results.append(row)
    logger.info(
        "Hybrid retrieval finished course=%s queries=%s candidates=%s keyword_hits=%s duration_ms=%.2f",
        course_id, len(queries), len(results), len(missing),
        (time.perf_counter() - started) * 1000,
    )
    return results


def merge_chunk_texts(first: str, second: str) -> str:
    """Join consecutive chunks, dropping the sentence overlap they share."""
    longest = min(len(first), len(second), 1000)
    for size in range(longest, 9, -1):
        # Only a whole-word overlap counts, so a coincidental suffix is not merged.
        at_boundary = size == len(first) or first[-size - 1].isspace()
        if at_boundary and first.endswith(second[:size]):
            return first + second[size:]
    return first + "\n" + second


def _chunk_index(row: dict) -> int | None:
    value = row["metadata"].get("chunk_index")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _fetch_chunks(course_id: int, document_id: str, *extra: dict) -> list[dict]:
    """Searchable chunks of one document (access filter applied), in document order."""
    result = get_collection().get(
        where=_course_where(course_id, {"document_id": str(document_id)}, *extra),
        include=["documents", "metadatas"],
    )
    rows = [{"chunk_id": chunk_id, "text": document, "metadata": metadata or {}}
            for chunk_id, document, metadata in zip(result.get("ids", []),
                                                    result.get("documents") or [],
                                                    result.get("metadatas") or [])]
    rows = [row for row in rows if _chunk_index(row) is not None]
    return sorted(rows, key=_chunk_index)


def expand_with_neighbours(matches: list[dict], course_id: int, neighbour_hits: int) -> list[dict]:
    """Merge each hit with its adjacent chunks so the model sees whole explanations.

    The top ``neighbour_hits`` hits also pull in chunk ``index-1`` and ``index+1``
    of the same document; hits that are adjacent to each other are merged into
    one passage. A passage keeps the metadata, citation page and score of its
    best-ranked hit, and ``chunk_count`` says how many chunks it contains.
    Chunks indexed before ``chunk_index`` existed are left as is.
    """
    wanted: dict[str, set[int]] = {}
    for position, match in enumerate(matches):
        index = _chunk_index(match)
        if index is None:
            continue
        document_id = str(match["metadata"].get("document_id"))
        indexes = wanted.setdefault(document_id, set())
        indexes.add(index)
        if position < neighbour_hits:
            indexes.update({index - 1, index + 1})
    if not wanted:
        return matches

    texts: dict[tuple[str, int], str] = {}
    for document_id, indexes in wanted.items():
        needed = sorted(i for i in indexes if i >= 0)
        for row in _fetch_chunks(course_id, document_id, {"chunk_index": {"$in": needed}}):
            texts[(document_id, _chunk_index(row))] = row["text"]

    runs: dict[tuple[str, int], tuple[int, ...]] = {}
    for document_id in wanted:
        present = sorted(i for (doc, i) in texts if doc == document_id)
        run: list[int] = []
        for index in present + [None]:
            if run and (index is None or index != run[-1] + 1):
                for member in run:
                    runs[(document_id, member)] = tuple(run)
                run = []
            if index is not None:
                run.append(index)

    expanded: list[dict] = []
    emitted: set[tuple[str, tuple[int, ...]]] = set()
    for match in matches:
        index = _chunk_index(match)
        document_id = str(match["metadata"].get("document_id"))
        run = runs.get((document_id, index)) if index is not None else None
        if run is None:
            expanded.append(match)
            continue
        if (document_id, run) in emitted:
            continue                     # already merged into a better-ranked passage
        emitted.add((document_id, run))
        text = texts[(document_id, run[0])]
        for member in run[1:]:
            text = merge_chunk_texts(text, texts[(document_id, member)])
        expanded.append({**match, "text": text, "chunk_count": len(run)})
    return expanded


def ordered_document_chunks(document_id: str, course_id: int,
                            heading: str | None = None) -> list[dict]:
    """A document's searchable chunks in reading order, optionally one heading only.

    Used for summaries and study guides of a whole document or section.
    """
    extra = ({"heading": heading},) if heading else ()
    return _fetch_chunks(course_id, document_id, *extra)
