import hashlib
import threading
from typing import Optional
import chromadb
from app.config import get_settings
from app.services.openai_client import embed_texts

_client = None
_collection = None
_retrieval_metadata_migrated = False 
_lock = threading.Lock()


class DocumentCourseConflictError(ValueError):
    """Raised when a document ID is already owned by another course."""

# Return the persistent ChromaDB collection, initializing it if necessary.
def get_collection():
    """Return the persistent ChromaDB collection."""
    global _client, _collection, _retrieval_metadata_migrated

    with _lock:
        if _collection is None:
            settings = get_settings()
            _client = chromadb.PersistentClient(
                path=settings.chroma_absolute_path
            )
            _collection = _client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )

        # Preserve retrieval access for chunks indexed before this feature.
        if not _retrieval_metadata_migrated:
            _migrate_retrieval_metadata(_collection)
            _retrieval_metadata_migrated = True

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
    digest = hashlib.sha256(
        f"{document_id}:{index}:{text}".encode()
    ).hexdigest()[:20]

    return f"{document_id}_{index}_{digest}"

# Index a document in the vector store.
def index_document(
    document_id: str,
    filename: str,
    course_id: int,
    owner_id: int,
    visibility: str,
    chunks: list[dict],
    retrieval_enabled: Optional[bool] = None,
) -> int:
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
        for offset in range(0, len(chunks), 32):
            batch = chunks[offset:offset + 32]
            vectors = embed_texts([chunk["text"] for chunk in batch])

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
                    "visibility": visibility,
                    "retrieval_enabled": bool(retrieval_enabled),
                    "page": (
                        int(chunk["page"])
                        if chunk.get("page")
                        else -1
                    ),
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

    return len(ids)

# Search only enabled chunks the user is allowed to access.
def search_course(
    question: str,
    course_id: int,
    user_id: int,
    include_course_documents: bool,
    top_k: int,
) -> list[dict]:
    """Search only enabled chunks the user is allowed to access."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    visibility = [
        {
            "$and": [
                {"visibility": "personal"},
                {"owner_id": int(user_id)},
            ]
        }
    ]

    if include_course_documents:
        visibility.append({"visibility": "course"})

    where = {
        "$and": [
            {"course_id": int(course_id)},
            {"retrieval_enabled": True},
            {"$or": visibility},
        ]
    }

    embedding = embed_texts([question])[0]

    result = collection.query(
        query_embeddings=[embedding],
        n_results=max(1, min(top_k, 20)),
        where=where,
        include=["documents", "metadatas", "distances"],
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