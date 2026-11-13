import asyncio
import base64
import hashlib
import hmac
import time
import uuid
from pathlib import Path
from typing import Any

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


def _build_prompt(question: str, answer: str, context: str) -> str:
    return (
        "Create an accurate educational infographic for a university student. "
        "Use a clear visual hierarchy, diagrams, arrows, stages and concise labels. "
        "Avoid paragraphs, tiny text, logos, watermarks and unsupported facts. "
        "Use the supplied course material as the factual source.\n\n"
        f"Question: {question[:2000]}\n\n"
        f"Explanation: {answer[:5000]}\n\n"
        f"Course material: {context[:9000]}"
    )


def _selected_image_key(settings: Any) -> str:
    provider = (settings.image_provider or "openai").lower()
    if provider not in {"openai", "gemini"}:
        raise RuntimeError(f"Unsupported image generation provider: {provider}")

    if settings.reuse_chat_api_key:
        if settings.llm_provider != provider:
            raise RuntimeError(
                "The chat API key can only be reused when chat and image providers match."
            )
        key = settings.openai_api_key if provider == "openai" else settings.gemini_api_key
    else:
        key = (
            settings.image_openai_api_key
            if provider == "openai"
            else settings.image_gemini_api_key
        )

    if not key:
        credential_name = "OPENAI" if provider == "openai" else "GEMINI"
        raise RuntimeError(
            f"No {credential_name} image API key is configured. Enter an image-provider key "
            "or enable reuse of the matching chat-provider key."
        )
    return key


def _generate_openai_image(prompt: str, settings: Any, api_key: str) -> bytes:
    if not settings.image_model:
        raise RuntimeError("Image generation model is not configured in Moodle settings.")
    response = requests.post(
        f"{settings.openai_base_url.rstrip('/')}/images/generations",
        headers={"Authorization": f"Bearer {api_key}"},
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
        return base64.b64decode(encoded, validate=True)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        detail = exc.response.text[:800] if exc.response is not None else str(exc)
        raise RuntimeError(f"OpenAI image generation returned HTTP {status}: {detail}") from exc
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"OpenAI image generation failed: {exc}") from exc


def _generate_gemini_image(prompt: str, settings: Any, api_key: str) -> bytes:
    if not settings.image_model:
        raise RuntimeError("Image generation model is not configured in Moodle settings.")
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=settings.image_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                inline_data = getattr(part, "inline_data", None)
                image_data = getattr(inline_data, "data", None)
                if isinstance(image_data, bytes) and image_data:
                    return image_data
                if isinstance(image_data, str) and image_data:
                    return base64.b64decode(image_data, validate=True)
        raise RuntimeError("Gemini returned no image data. Check that the selected model supports image output.")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Gemini image generation failed: {type(exc).__name__}: {exc}") from exc


def _generate_image_bytes(question: str, answer: str, context: str) -> bytes:
    settings = get_settings()
    provider = (settings.image_provider or "openai").lower()
    api_key = _selected_image_key(settings)
    prompt = _build_prompt(question, answer, context)

    if provider == "openai":
        return _generate_openai_image(prompt, settings, api_key)
    if provider == "gemini":
        return _generate_gemini_image(prompt, settings, api_key)
    raise RuntimeError(f"Unsupported image generation provider: {provider}")


def _store_image(image_bytes: bytes) -> str:
    if not image_bytes:
        raise RuntimeError("Image provider returned an empty image.")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_id = uuid.uuid4().hex
    (IMAGE_DIR / f"{image_id}.png").write_bytes(image_bytes)
    expires = int(time.time()) + IMAGE_TTL
    signature = _signature(image_id, expires)
    return f"/api/v1/illustrations/{image_id}?expires={expires}&signature={signature}"


def generate_illustration(question: str, answer: str, context: str) -> str:
    return _store_image(_generate_image_bytes(question, answer, context))


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
    """Generate an illustration without blocking the event loop."""
    image_bytes = await asyncio.to_thread(_generate_image_bytes, question, answer, context)
    return _store_image(image_bytes)
