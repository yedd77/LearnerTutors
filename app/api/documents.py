import logging
import time
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.schemas import DeleteResponse, IndexResponse
from app.security import RequireApiKey
from app.services.chunker import chunk_sections
from app.services.document_parser import DocumentParseError, extract_sections
from app.services.suggestion_generator import generate_document_suggestions
from app.services.vector_store import (
    DocumentCourseConflictError,
    delete_document,
    index_document,
    set_document_retrieval,
)

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[RequireApiKey])
logger = logging.getLogger(__name__)


@router.post("/index", response_model=IndexResponse, status_code=201)
def index_upload(
    file: Annotated[UploadFile, File()],
    document_id: Annotated[str, Form(min_length=1, max_length=100)],
    course_id: Annotated[int, Form(gt=0)],
    owner_id: Annotated[int, Form(gt=0)],
    visibility: Annotated[str, Form()] = "personal",
    retrieval_enabled: Annotated[bool, Form()] = True,
    generate_suggestions: Annotated[bool, Form()] = False,
):
    s = get_settings()
    if visibility not in {"personal", "course"}:
        raise HTTPException(400, "visibility must be 'personal' or 'course'.")

    started = time.perf_counter()
    filename = Path(file.filename or "upload").name
    logger.info(
        "document.index.start document=%s course=%s filename=%s visibility=%s",
        document_id, course_id, filename, visibility,
    )
    if Path(filename).suffix.lower() not in {".pdf", ".docx", ".pptx", ".txt", ".md"}:
        raise HTTPException(415, "Unsupported file type.")

    content_length = file.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise HTTPException(400, "Invalid Content-Length header.") from exc
        if declared_size > s.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File exceeds {s.max_upload_mb} MB.")

    content = file.file.read(s.max_upload_mb * 1024 * 1024 + 1)
    file.file.close()
    if len(content) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {s.max_upload_mb} MB.")
    if not content:
        raise HTTPException(400, "Uploaded file is empty.")

    parse_started = time.perf_counter()
    try:
        sections = extract_sections(filename, content)
        chunks = chunk_sections(sections)
        logger.info(
            "document.parse.finish document=%s sections=%s chunks=%s duration_ms=%.2f",
            document_id, len(sections), len(chunks),
            (time.perf_counter() - parse_started) * 1000,
        )
    except DocumentParseError as exc:
        logger.warning("document.parse.error document=%s filename=%s error=%s", document_id, filename, exc)
        raise HTTPException(422, str(exc)) from exc

    if len(chunks) > s.max_chunks:
        raise HTTPException(413, f"File produces more than {s.max_chunks} chunks.")

    index_started = time.perf_counter()
    logger.info("document.vector_index.start document=%s chunks=%s", document_id, len(chunks))
    try:
        count = index_document(
            document_id,
            filename,
            course_id,
            owner_id,
            visibility,
            chunks,
            retrieval_enabled=retrieval_enabled,
        )
    except DocumentCourseConflictError as exc:
        logger.warning("document.index.conflict document=%s course=%s error=%s", document_id, course_id, exc)
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        logger.exception("document.vector_index.error document=%s course=%s", document_id, course_id)
        raise HTTPException(502, f"Indexing failed: {exc}") from exc

    logger.info(
        "document.vector_index.finish document=%s indexed_chunks=%s duration_ms=%.2f",
        document_id, count, (time.perf_counter() - index_started) * 1000,
    )

    # Suggestions run as a separate Moodle background task so slow LLM
    # generation never holds up indexing completion or the document status.
    suggestions = []
    if generate_suggestions:
        suggestions_started = time.perf_counter()
        try:
            suggestions = generate_document_suggestions(filename, sections)
            logger.info(
                "document.suggestions.finish document=%s suggestions=%s duration_ms=%.2f",
                document_id, len(suggestions),
                (time.perf_counter() - suggestions_started) * 1000,
            )
        except Exception:
            logger.exception("document.suggestions.error document=%s", document_id)

    logger.info(
        "document.index.finish document=%s total_ms=%.2f",
        document_id, (time.perf_counter() - started) * 1000,
    )
    return IndexResponse(
        document_id=document_id,
        filename=filename,
        course_id=course_id,
        indexed_chunks=count,
        status="ready",
        suggestions=suggestions,
    )


class SuggestionResponse(BaseModel):
    document_id: str
    suggestions: list[str]


@router.post("/suggestions", response_model=SuggestionResponse)
def regenerate_suggestions(
    file: Annotated[UploadFile, File()],
    document_id: Annotated[str, Form(min_length=1, max_length=100)],
):
    """Generate starter questions for a document without changing its vectors."""
    settings = get_settings()
    started = time.perf_counter()
    filename = Path(file.filename or "upload").name
    if Path(filename).suffix.lower() not in {".pdf", ".docx", ".pptx", ".txt", ".md"}:
        raise HTTPException(415, "Unsupported file type.")

    content = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    file.file.close()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB.")
    if not content:
        raise HTTPException(400, "Uploaded file is empty.")

    try:
        sections = extract_sections(filename, content)
    except DocumentParseError as exc:
        logger.warning("document.suggestions.parse_error document=%s filename=%s error=%s",
                       document_id, filename, exc)
        raise HTTPException(422, str(exc)) from exc

    suggestions = generate_document_suggestions(filename, sections)
    logger.info(
        "document.suggestions_only.finish document=%s suggestions=%s duration_ms=%.2f",
        document_id, len(suggestions), (time.perf_counter() - started) * 1000,
    )
    return SuggestionResponse(document_id=document_id, suggestions=suggestions)


@router.delete("/{document_id}", response_model=DeleteResponse)
def remove_document(document_id: str, course_id: int):
    try:
        count = delete_document(document_id, course_id)
    except Exception as exc:
        raise HTTPException(500, f"Deletion failed: {exc}") from exc
    if count == 0:
        raise HTTPException(404, "Document not found in this course.")
    return DeleteResponse(document_id=document_id, deleted_chunks=count, status="deleted")


class RetrievalToggleRequest(BaseModel):
    enabled: bool


@router.patch("/{document_id}/retrieval")
def toggle_retrieval(
    document_id: str,
    course_id: int,
    request: RetrievalToggleRequest,
):
    try:
        count = set_document_retrieval(
            document_id,
            course_id,
            request.enabled,
        )
    except Exception as exc:
        raise HTTPException(
            500,
            f"Failed to update retrieval setting: {exc}",
        ) from exc
    if count == 0:
        raise HTTPException(404, "Document not found in this course.")

    return {
        "document_id": document_id,
        "retrieval_enabled": request.enabled,
        "updated_chunks": count,
    }
