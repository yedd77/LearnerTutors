"""Download endpoint for temporary generated files (study sheet PDFs).

Protected by the API key: only the Moodle server calls it, through a plugin
endpoint that has already checked the logged-in user and course. The file is
served only when course_id and user_id match the file's owner.
"""
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.security import RequireApiKey
from app.services import temp_files

router = APIRouter(prefix="/generated-files", tags=["generated-files"],
                   dependencies=[RequireApiKey])
logger = logging.getLogger(__name__)


@router.get("/{file_id}", response_class=FileResponse)
def download_generated_file(
    file_id: str,
    course_id: int = Query(..., gt=0),
    user_id: int = Query(..., gt=0),
):
    """Return the file. 404 unknown or not this user's, 410 expired (and now deleted)."""
    try:
        path, record = temp_files.open_for(file_id, course_id=course_id, user_id=user_id)
    except temp_files.GeneratedFileExpired:
        logger.info("generated_file.expired id=%s course=%s", file_id[:32], course_id)
        raise HTTPException(410, "This file has expired and was deleted. Generate it again.")
    except temp_files.GeneratedFileNotFound:
        raise HTTPException(404, "File not found.")
    return FileResponse(
        path,
        media_type=record["content_type"],
        filename=record["filename"],
        headers={"Cache-Control": "private, no-store"},
    )
