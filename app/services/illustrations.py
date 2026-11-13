import base64
import hashlib
import hmac
import time
import uuid
from pathlib import Path

import requests
from fastapi import HTTPException
from fastapi.responses import FileResponse

from app.config import get_settings

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "storage" / "illustrations"
IMAGE_TTL = 60 * 60 * 24 * 365


def _signature(image_id: str, expires: int) -> str:
    secret = get_settings().rag_api_key
    if not secret:
        raise RuntimeError("RAG_API_KEY is not configured.")
    message = f"{image_id}:{expires}".encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def generate_illustration(question: str, answer: str, context: str) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    if not settings.image_model:
        raise RuntimeError("Image generation model is not configured in Moodle settings.")

    prompt = (
        "Create an accurate educational infographic for a university student. "
        "Use a clear visual hierarchy, diagrams, arrows, stages and concise labels. "
        "Avoid paragraphs, tiny text, logos, watermarks and unsupported facts. "
        "Use the supplied course material as the factual source.\n\n"
        f"Question: {question[:2000]}\n\n"
        f"Explanation: {answer[:5000]}\n\n"
        f"Course material: {context[:9000]}"
    )
    response = requests.post(
        f"{settings.openai_base_url.rstrip('/')}/images/generations",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json={
            "model": settings.image_model,
            "prompt": prompt,
            "size": "1024x1024",
            "quality": "medium",
            "output_format": "png",
            "n": 1,
        },
        timeout=max(settings.openai_timeout_seconds, 180),
    )
    try:
        response.raise_for_status()
        encoded = response.json()["data"][0]["b64_json"]
        image_bytes = base64.b64decode(encoded, validate=True)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        detail = exc.response.text[:800] if exc.response is not None else str(exc)
        raise RuntimeError(f"Image generation returned HTTP {status}: {detail}") from exc
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Image generation failed: {exc}") from exc

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_id = uuid.uuid4().hex
    (IMAGE_DIR / f"{image_id}.png").write_bytes(image_bytes)
    expires = int(time.time()) + IMAGE_TTL
    signature = _signature(image_id, expires)
    return f"/api/v1/illustrations/{image_id}?expires={expires}&signature={signature}"


def serve_illustration(image_id: str, expires: int, signature: str) -> FileResponse:
    if len(image_id) != 32 or any(c not in "0123456789abcdef" for c in image_id):
        raise HTTPException(404, "Illustration not found.")
    now = int(time.time())
    if expires < now or expires > now + IMAGE_TTL + 60:
        raise HTTPException(403, "Illustration link expired.")
    expected = _signature(image_id, expires)
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(403, "Invalid illustration signature.")
    path = IMAGE_DIR / f"{image_id}.png"
    if not path.is_file():
        raise HTTPException(404, "Illustration not found.")
    return FileResponse(path, media_type="image/png", filename="explanation.png")


async def generate_illustration_async(question: str, answer: str, context: str) -> str:
    """Generate an illustration using a cancellable asynchronous HTTP request."""
    import httpx

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    if not settings.image_model:
        raise RuntimeError("Image generation model is not configured in Moodle settings.")

    prompt = (
        "Create an accurate educational infographic for a university student. "
        "Use a clear visual hierarchy, diagrams, arrows, stages and concise labels. "
        "Avoid paragraphs, tiny text, logos, watermarks and unsupported facts. "
        "Use the supplied course material as the factual source.\n\n"
        f"Question: {question[:2000]}\n\n"
        f"Explanation: {answer[:5000]}\n\n"
        f"Course material: {context[:9000]}"
    )
    timeout = max(settings.openai_timeout_seconds, 180)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{settings.openai_base_url.rstrip('/')}/images/generations",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model": settings.image_model,
                "prompt": prompt,
                "size": "1024x1024",
                "quality": "medium",
                "output_format": "png",
                "n": 1,
            },
        )
        response.raise_for_status()
        encoded = response.json()["data"][0]["b64_json"]
        image_bytes = base64.b64decode(encoded, validate=True)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_id = uuid.uuid4().hex
    (IMAGE_DIR / f"{image_id}.png").write_bytes(image_bytes)
    expires = int(time.time()) + IMAGE_TTL
    signature = _signature(image_id, expires)
    return f"/api/v1/illustrations/{image_id}?expires={expires}&signature={signature}"
