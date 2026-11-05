from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.services.illustrations import serve_illustration

router = APIRouter(prefix="/illustrations", tags=["illustrations"])


@router.get("/{image_id}", response_class=FileResponse)
def read_illustration(
    image_id: str,
    expires: int = Query(..., ge=1),
    signature: str = Query(..., min_length=64, max_length=64),
):
    """Serve a generated image through an expiring signed URL."""
    return serve_illustration(image_id, expires, signature)
