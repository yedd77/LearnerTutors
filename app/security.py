import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from app.config import get_settings


# Security dependency to require a valid API key in the X-RAG-API-Key header for protected endpoints.
def verify_api_key(
    x_rag_api_key: Annotated[str | None, Header(alias="X-RAG-API-Key")] = None,
) -> None:
    expected = get_settings().rag_api_key

    # Require a sufficiently long random shared secret rather than accepting
    # short human passwords as service-to-service credentials.
    if len(expected) < 32:
        raise HTTPException(
            503,
            "RAG_API_KEY is not configured or is too short. Use at least 32 characters.",
        )

    if not x_rag_api_key or not secrets.compare_digest(x_rag_api_key, expected):
        raise HTTPException(401, "Invalid or missing API key.")


RequireApiKey = Depends(verify_api_key)
