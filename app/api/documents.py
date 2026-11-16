"""Document endpoints: index uploads, generate suggestions, delete, toggle retrieval.

Moodle is the only caller (authenticated with X-RAG-API-Key). Only course-wide
documents are supported; see DECISION-003 in conductor/decision-log.md.
"""
import json
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
from app.services.document_parser import SUPPORTED_EXTENSIONS, DocumentParseError, extract_sections
from app.services.suggestion_generator import (
    generate_document_suggestions,
    generate_one_document_suggestion,
)
from app.services.vector_store import (
    DocumentCourseConflictError,
    delete_document,
    index_document,
    set_document_retrieval,
)

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[RequireApiKey])
logger = logging.getLogger(__name__)


def _upload_filename(file: UploadFile) -> str:
    """Return the uploaded file's base name, dropping any client-supplied directories."""
    return Path(file.filename or "upload").name


def _read_upload(file: UploadFile, filename: str) -> bytes:
    """Validate an upload's type and size, then return its bytes.

    Raises 415 for unsupported extensions, 400 for a malformed Content-Length or
    an empty file, and 413 when the file is larger than ``MAX_UPLOAD_MB``. Never
    reads more than the limit plus one byte, so oversized uploads are rejected
    without loading them fully into memory.
    """
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, "Unsupported file type.")

    max_mb = get_settings().max_upload_mb
    max_bytes = max_mb * 1024 * 1024
    # Reject early when the client declares an oversized part.
    content_length = file.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise HTTPException(400, "Invalid Content-Length header.") from exc
        if declared_size > max_bytes:
            raise HTTPException(413, f"File exceeds {max_mb} MB.")

    content = file.file.read(max_bytes + 1)
    file.file.close()
    if len(content) > max_bytes:
        raise HTTPException(413, f"File exceeds {max_mb} MB.")
    if not content:
        raise HTTPException(400, "Uploaded file is empty.")
    return content


@router.post("/index", response_model=IndexResponse, status_code=201)
def index_upload(
    file: Annotated[UploadFile, File()],
    document_id: Annotated[str, Form(min_length=1, max_length=100)],
    course_id: Annotated[int, Form(gt=0)],
    owner_id: Annotated[int, Form(gt=0)],
    visibility: Annotated[str | None, Form()] = None,
    retrieval_enabled: Annotated[bool, Form()] = True,
    generate_suggestions: Annotated[bool, Form()] = False,
):
    """Parse, chunk, embed and store one course document, replacing any earlier version.

    Optionally generates three starter-question suggestions afterwards; a
    suggestion failure is logged but never fails the indexing request.
    Errors: 400 (visibility/empty), 409 (document_id belongs to another course),
    413 (size or chunk limit), 415 (type), 422 (unparseable), 502 (embedding/store).
    """
    s = get_settings()
    # Personal (student-owned) uploads were removed. Require an explicit course
    # visibility: an omitted value used to mean "personal", so silently treating
    # it as course-wide could expose a student's file to the whole course.
    if visibility != "course":
        raise HTTPException(
            400,
            "Personal document uploads are not supported; visibility must be 'course'.",
        )

    started = time.perf_counter()
    filename = _upload_filename(file)
    logger.info(
        "document.index.start document=%s course=%s filename=%s visibility=%s",
        document_id, course_id, filename, visibility,
    )
    content = _read_upload(file, filename)

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
    """Response of ``POST /documents/suggestions``."""
    document_id: str
    suggestions: list[str]


@router.post("/suggestions", response_model=SuggestionResponse)
def regenerate_suggestions(
    file: Annotated[UploadFile, File()],
    document_id: Annotated[str, Form(min_length=1, max_length=100)],
):
    """Generate starter questions for a document without changing its vectors.

    Falls back to deterministic questions when the LLM fails, so this only
    errors on invalid uploads (400/413/415) or unparseable files (422).
    """
    started = time.perf_counter()
    filename = _upload_filename(file)
    content = _read_upload(file, filename)

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


@router.post("/{document_id}/suggestions/{slot}")
def generate_one_suggestion(
    document_id: str,
    slot: int,
    file: Annotated[UploadFile, File()],
    existing_suggestions: Annotated[str, Form()] = "[]",
):
    """Generate one document-grounded suggestion without modifying indexed vectors.

    Used when a lecturer replaces a single suggestion card (``slot`` 1-3).
    ``existing_suggestions`` is a JSON array of strings the new question must
    differ from. Unlike the bulk endpoint there is no fallback: LLM or
    validation failures return 502.
    """
    if slot < 1 or slot > 3:
        raise HTTPException(422, "slot must be between 1 and 3.")

    filename = _upload_filename(file)
    content = _read_upload(file, filename)

    try:
        sections = extract_sections(filename, content)
        existing = json.loads(existing_suggestions or "[]")
        if not isinstance(existing, list) or not all(isinstance(item, str) for item in existing):
            raise ValueError("existing_suggestions must be a JSON array of strings.")
        suggestion = generate_one_document_suggestion(filename, sections, existing)
    except DocumentParseError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(502, f"Suggestion generation failed: {exc}") from exc
    except Exception as exc:
        logger.exception("Single suggestion generation failed for document %s slot %s", document_id, slot)
        raise HTTPException(502, f"Suggestion generation failed: {exc}") from exc

    return {"document_id": document_id, "slot": slot, "suggestion": suggestion}


@router.delete("/{document_id}", response_model=DeleteResponse)
def remove_document(document_id: str, course_id: int):
    """Delete every chunk of a document, scoped to its course (404 if none matched)."""
    try:
        count = delete_document(document_id, course_id)
    except Exception as exc:
        raise HTTPException(500, f"Deletion failed: {exc}") from exc
    if count == 0:
        raise HTTPException(404, "Document not found in this course.")
    return DeleteResponse(document_id=document_id, deleted_chunks=count, status="deleted")


class RetrievalToggleRequest(BaseModel):
    """Body of ``PATCH /documents/{id}/retrieval``."""
    enabled: bool


@router.patch("/{document_id}/retrieval")
def toggle_retrieval(
    document_id: str,
    course_id: int,
    request: RetrievalToggleRequest,
):
    """Enable or disable a document for chat retrieval without deleting its vectors."""
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
