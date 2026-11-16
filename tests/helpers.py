"""Plain helpers shared by test modules (importable as `helpers`)."""
from app.services.vector_store import get_collection, index_document


def chat_body(question: str, course_id: int = 1, user_id: int = 10, **extra) -> dict:
    body = {
        "question": question,
        "course_id": course_id,
        "user_id": user_id,
        "provider": "openai",
        "model": "test-model",
    }
    body.update(extra)
    return body


def index_text(document_id: str, text: str, *, course_id: int = 1, owner_id: int = 10,
               filename: str | None = None, page: int | None = 1,
               retrieval_enabled: bool | None = None) -> int:
    """Index one chunk of text directly through the vector store."""
    return index_document(
        document_id,
        filename or f"{document_id}.txt",
        course_id,
        owner_id,
        [{"page": page, "text": text}],
        retrieval_enabled=retrieval_enabled,
    )


def add_legacy_personal_chunk(document_id: str, text: str, *, course_id: int = 1,
                              owner_id: int = 10) -> None:
    """Simulate a student upload indexed before personal documents were removed.

    The current code can no longer create these, but existing Chroma stores may
    still contain them, so retrieval must keep excluding them.
    """
    from conftest import fake_vector

    get_collection().add(
        ids=[f"{document_id}_legacy"],
        documents=[text],
        embeddings=[fake_vector(text)],
        metadatas=[{
            "document_id": document_id, "filename": f"{document_id}.txt",
            "course_id": course_id, "owner_id": owner_id, "visibility": "personal",
            "retrieval_enabled": True, "page": 1,
        }],
    )
