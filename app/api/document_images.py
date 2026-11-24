"""Images extracted from course documents, for answers that show them.

Protected by the API key: only the Moodle server calls it, through a plugin
endpoint that has already checked the logged-in user's access to the course.
The image is served only if it belongs to that course's document and the
document is still searchable (not deleted, disabled or replaced).
"""
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.security import RequireApiKey
from app.services import document_images, vector_store

router = APIRouter(prefix="/document-images", tags=["document-images"],
                   dependencies=[RequireApiKey])
logger = logging.getLogger(__name__)


@router.get("/{image_id}", response_class=FileResponse)
def read_document_image(
    image_id: str,
    course_id: int = Query(..., gt=0),
    document_id: str = Query(..., min_length=1, max_length=255),
):
    """Return the PNG. 404 when unknown, not in this course document, or not searchable."""
    path = document_images.image_path(image_id, course_id, document_id, vector_store.document_visible)
    if path is None:
        logger.info("document_image.not_found course=%s", course_id)
        raise HTTPException(404, "Image not found.")
    return FileResponse(path, media_type="image/png", filename=f"{image_id}.png",
                        headers={"Cache-Control": "private, no-store"})
