import logging
from typing import Any
import requests
from app.config import get_settings
from app.prompt_builder import build_prompt_bundle
from app.prompt_router import RouteResult


logger = logging.getLogger(__name__)
Message = dict[str, str]


class OllamaError(RuntimeError):
    pass

# construct the full URL for Ollama API endpoints
def _url(path: str, base_url: str) -> str:
    return f"{base_url.rstrip('/')}{path}"

# log the complete chat prompt sent to Ollama
def log_prompt(messages: list[Message]) -> None:
    """Log the complete chat prompt sent to Ollama."""
    logger.debug(
        "Ollama prompt sent:\n%s",
        "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}"
            for message in messages
        ),
    )

# send a POST request to the Ollama API with the given payload
def _post(path: str, payload: dict[str, Any]) -> requests.Response:
    settings = get_settings()
    return requests.post(
        _url(path, settings.ollama_base_url),
        json=payload,
        timeout=settings.ollama_timeout_seconds,
    )

# build a text representation of the last few messages in the conversation history
def _history_text(history: list[dict[str, Any]]) -> str:
    entries = [
        f"{item.get('role', 'user').upper()}: "
        f"{str(item.get('content', ''))[:2000]}"
        for item in history[-6:]
        if item.get("role") in ("user", "assistant")
    ]
    return "\n".join(entries) or "(No previous conversation.)"

# parse the embedding response from Ollama and ensure it matches the expected count
def _embedding_response(response: requests.Response, expected_count: int) -> list[list[float]]:
    embeddings = response.json().get("embeddings")
    if not embeddings or len(embeddings) != expected_count:
        raise OllamaError("Unexpected embedding response from Ollama.")
    return embeddings

# parse the answer response from Ollama and ensure it is not empty
def _answer_response(response: requests.Response) -> str:
    answer = response.json().get("message", {}).get("content", "").strip()
    if not answer:
        raise OllamaError("Ollama returned an empty answer.")
    return answer

# generate embeddings for a list of texts using the Ollama API
def embed_texts(texts: list[str], prefix: str = "") -> list[list[float]]:
    if not texts:
        return []
    settings = get_settings()
    inputs = [f"{prefix}{text}" for text in texts]
    try:
        response = _post(
            "/api/embed",
            {"model": settings.ollama_embed_model, "input": inputs},
        )
        response.raise_for_status()
        return _embedding_response(response, len(texts))
    except requests.RequestException as exc:
        raise OllamaError(
            f"Ollama embedding request failed. Check model "
            f"'{settings.ollama_embed_model}' and service status."
        ) from exc

# generate an answer from the Ollama API based on the provided route, context, and conversation history
def generate_answer(
    route: RouteResult,
    context: str,
    history: list[dict] | None = None,
) -> str:
    settings = get_settings()
    bundle = build_prompt_bundle(
        route=route,
        context=context,
        history=_history_text(history or []),
    )

    try:
        messages = bundle["messages"]
        if settings.log_prompts:
            log_prompt(messages)
        response = _post(
            "/api/chat",
            {
                "model": settings.ollama_chat_model,
                "messages": messages,
                "stream": False,
                "think": "low",
                "options": {"temperature": 0.2, "num_ctx": settings.num_ctx},
            },
        )
        response.raise_for_status()
        return _answer_response(response)
    except requests.HTTPError as exc:
        response = exc.response
        status = response.status_code if response is not None else "unknown"
        body = response.text[:1000] if response is not None else str(exc)
        raise OllamaError(f"Ollama chat returned HTTP {status}: {body}") from exc
    except requests.RequestException as exc:
        raise OllamaError(
            f"Ollama chat request failed: {type(exc).__name__}: {exc}"
        ) from exc