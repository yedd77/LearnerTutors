import logging
import os
from pathlib import Path
from typing import Any
import requests
from dotenv import load_dotenv
from app.prompt_builder import build_prompt_bundle
from app.prompt_router import RouteResult


logger = logging.getLogger(__name__)
Message = dict[str, str]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Maximum number of texts sent to the embeddings endpoint in one request.
EMBED_REQUEST_SIZE = 100


class OpenAIError(RuntimeError):
    pass

# log the complete chat prompt sent to the chat model
def log_prompt(messages: list[Message]) -> None:
    """Log the complete chat prompt sent to the chat model."""
    logger.debug(
        "Chat prompt sent:\n%s",
        "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}"
            for message in messages
        ),
    )

# send a POST request to the OpenAI API with the given payload
def _post(path: str, payload: dict[str, Any]) -> requests.Response:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise OpenAIError("OPENAI_API_KEY is not configured on the backend.")
    return requests.post(
        f"{os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')}{path}",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=int(os.getenv("OPENAI_TIMEOUT_SECONDS", "120")),
    )

# convert a failed HTTP response into an OpenAIError with a short body excerpt
def _http_error(label: str, exc: requests.HTTPError) -> OpenAIError:
    response = exc.response
    status = response.status_code if response is not None else "unknown"
    body = response.text[:1000] if response is not None else str(exc)
    return OpenAIError(f"OpenAI {label} returned HTTP {status}: {body}")

# build a text representation of the last few messages in the conversation history
def _history_text(history: list[dict[str, Any]]) -> str:
    entries = [
        f"{item.get('role', 'user').upper()}: "
        f"{str(item.get('content', ''))[:2000]}"
        for item in history[-6:]
        if item.get("role") in ("user", "assistant")
    ]
    return "\n".join(entries) or "(No previous conversation.)"

# parse the embedding response from OpenAI and ensure it matches the expected count
def _embedding_response(response: requests.Response, expected_count: int) -> list[list[float]]:
    try:
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        embeddings = [row["embedding"] for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise OpenAIError("Unexpected embedding response from OpenAI.") from exc
    if len(embeddings) != expected_count:
        raise OpenAIError("Unexpected embedding response from OpenAI.")
    return embeddings

# parse the chat completion response from OpenAI and ensure the answer is not empty
def _answer_response(response: requests.Response) -> str:
    
    try:
        answer = (response.json()["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise OpenAIError("Unexpected chat response from OpenAI.") from exc
    if not answer:
        raise OpenAIError("OpenAI returned an empty answer.")
    return answer

# generate embeddings for a list of texts using the OpenAI API
def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(texts), EMBED_REQUEST_SIZE):
            batch = texts[start:start + EMBED_REQUEST_SIZE]
            response = _post(
                "/embeddings",
                {
                    "model": os.getenv(
                        "OPENAI_EMBED_MODEL", "text-embedding-3-small"
                    ),
                    "input": batch,
                },
            )
            response.raise_for_status()
            vectors.extend(_embedding_response(response, len(batch)))
        return vectors
    except requests.HTTPError as exc:
        raise _http_error("embeddings", exc) from exc
    except requests.RequestException as exc:
        raise OpenAIError(
            f"OpenAI embedding request failed: {type(exc).__name__}: {exc}"
        ) from exc

# generate an answer from the OpenAI API based on the provided route, context, and conversation history
def generate_answer(
    route: RouteResult,
    context: str,
    history: list[dict] | None = None,
) -> str:
    bundle = build_prompt_bundle(
        route=route,
        context=context,
        history=_history_text(history or []),
    )
    messages = bundle["messages"]
    if os.getenv("LOG_PROMPTS", "false").strip().lower() in {"1", "true", "yes", "on"}:
        log_prompt(messages)

    payload: dict[str, Any] = {
        "model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "messages": messages,
    }
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    if temperature >= 0:
        payload["temperature"] = temperature
    max_completion_tokens = int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", "0"))
    if max_completion_tokens > 0:
        payload["max_completion_tokens"] = max_completion_tokens

    try:
        response = _post("/chat/completions", payload)
        response.raise_for_status()
        return _answer_response(response)
    except requests.HTTPError as exc:
        raise _http_error("chat", exc) from exc
    except requests.RequestException as exc:
        raise OpenAIError(
            f"OpenAI chat request failed: {type(exc).__name__}: {exc}"
        ) from exc