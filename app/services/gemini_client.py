import logging
from functools import lru_cache
from typing import Any
from google import genai
from google.genai import errors, types
from app.config import get_settings
from app.prompt_builder import build_prompt_bundle
from app.prompt_router import RouteResult


logger = logging.getLogger(__name__)
Message = dict[str, str]

# Maximum number of texts sent to the embeddings endpoint in one request.
EMBED_REQUEST_SIZE = 100


class GeminiError(RuntimeError):
    pass

# strip an optional "models/" prefix so the model name is consistent
def _model_id(model: str) -> str:
    return model.removeprefix("models/")

# create the Gemini SDK client once and reuse it
@lru_cache
def _client() -> genai.Client:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise GeminiError("GEMINI_API_KEY is not configured on the backend.")
    return genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=settings.gemini_timeout_seconds * 1000),
    )

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

# build a text representation of the last few messages in the conversation history
def _history_text(history: list[dict[str, Any]]) -> str:
    entries = [
        f"{item.get('role', 'user').upper()}: "
        f"{str(item.get('content', ''))[:2000]}"
        for item in history[-6:]
        if item.get("role") in ("user", "assistant")
    ]
    return "\n".join(entries) or "(No previous conversation.)"

# convert chat messages into a system instruction string and Gemini contents
def _to_gemini_contents(messages: list[Message]) -> tuple[str | None, list[types.Content]]:
    system_parts: list[str] = []
    contents: list[types.Content] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
        else:
            role = "model" if message["role"] == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=message["content"])])
            )
    return ("\n\n".join(system_parts) or None), contents

# turn an SDK API error into a GeminiError with the HTTP status and message
def _api_error(label: str, exc: errors.APIError) -> GeminiError:
    return GeminiError(
        f"Gemini {label} returned HTTP {exc.code} ({exc.status}): "
        f"{str(exc.message)[:1000]}"
    )

# extract the answer text from a generate_content response and ensure it is not empty
def _answer_text(response: types.GenerateContentResponse) -> str:
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason:
        reason = getattr(feedback.block_reason, "name", feedback.block_reason)
        raise GeminiError(f"Gemini blocked the prompt ({reason}).")
    if not response.candidates:
        raise GeminiError("Gemini returned no answer candidates.")

    logger.debug("Gemini model version used: %s", response.model_version)
    candidate = response.candidates[0]
    parts = (candidate.content.parts if candidate.content else None) or []
    answer = "".join(
        part.text for part in parts if part.text and not part.thought
    ).strip()
    if not answer:
        reason = getattr(candidate.finish_reason, "name", candidate.finish_reason) or "unknown"
        raise GeminiError(f"Gemini returned an empty answer (finish reason: {reason}).")
    return answer

# generate embeddings for a list of texts using the Gemini API
# task_type is RETRIEVAL_DOCUMENT when indexing and RETRIEVAL_QUERY when searching
def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    if not texts:
        return []
    settings = get_settings()
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=settings.gemini_embed_dimensions or None,
    )
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(texts), EMBED_REQUEST_SIZE):
            batch = texts[start:start + EMBED_REQUEST_SIZE]
            response = _client().models.embed_content(
                model=_model_id(settings.gemini_embed_model),
                contents=batch,
                config=config,
            )
            embeddings = [item.values for item in (response.embeddings or [])]
            if len(embeddings) != len(batch) or any(not item for item in embeddings):
                raise GeminiError("Unexpected embedding response from Gemini.")
            vectors.extend(embeddings)
        return vectors
    except errors.APIError as exc:
        raise _api_error("embeddings", exc) from exc
    except GeminiError:
        raise
    except Exception as exc:
        raise GeminiError(
            f"Gemini embedding request failed: {type(exc).__name__}: {exc}"
        ) from exc

# generate an answer from the Gemini API based on the provided route, context, and conversation history
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
    messages = bundle["messages"]
    if settings.log_prompts:
        log_prompt(messages)

    system_instruction, contents = _to_gemini_contents(messages)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=settings.gemini_temperature if settings.gemini_temperature >= 0 else None,
        max_output_tokens=settings.gemini_max_output_tokens or None,
    )
    try:
        response = _client().models.generate_content(
            model=_model_id(settings.gemini_chat_model),
            contents=contents,
            config=config,
        )
        return _answer_text(response)
    except errors.APIError as exc:
        raise _api_error("chat", exc) from exc
    except GeminiError:
        raise
    except Exception as exc:
        raise GeminiError(
            f"Gemini chat request failed: {type(exc).__name__}: {exc}"
        ) from exc