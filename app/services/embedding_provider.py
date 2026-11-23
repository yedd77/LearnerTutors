"""Text embeddings for indexing and retrieval (OpenAI, Gemini or Ollama).

The provider/model comes from Moodle (``POST /admin/config``) when it sets one,
otherwise from ``.env`` (EMBEDDING_PROVIDER, EMBEDDING_MODEL). The API key is
the chat key for that provider when ``embedding_reuse_chat_api_key`` is on;
otherwise the Moodle-managed embedding key if stored, then the chat key
(Moodle, then ``.env``). Vectors from different models are not
comparable, so a provider/model change selects a different Chroma collection
and requires re-indexing.
"""
import logging
import time

import requests
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Inputs sent per provider request.
_OPENAI_BATCH = 100
_OLLAMA_BATCH = 32
_GEMINI_BATCH = 100

# nomic-embed-text was trained with task prefixes and retrieves noticeably worse
# without them; Ollama does not add them. Changing this changes the vectors, so
# the prefix is part of Settings.embedding_identity (a new collection + reindex).
_NOMIC_PREFIXES = {"RETRIEVAL_DOCUMENT": "search_document: ", "RETRIEVAL_QUERY": "search_query: "}


def uses_nomic_prefixes(provider: str, model: str) -> bool:
    """True for Ollama nomic-embed models, which need ``search_query:``/``search_document:``."""
    return provider.lower() == "ollama" and "nomic-embed" in model.lower()


class EmbeddingError(RuntimeError):
    """Embedding failed: bad configuration, a provider error, or an unexpected response."""


def _batches(texts: list[str], size: int):
    """Yield consecutive slices of ``texts`` with at most ``size`` items each."""
    for start in range(0, len(texts), size):
        yield texts[start:start + size]


def _check_count(vectors: list, batch: list[str], provider_label: str) -> list:
    """Ensure the provider returned exactly one vector per input text."""
    if len(vectors) != len(batch):
        raise EmbeddingError(f"{provider_label} returned an unexpected embedding count.")
    return vectors


def chat_key_reuse_error(embedding_provider: str, llm_provider: str) -> str | None:
    """Why the chat key cannot be reused for embeddings, or None if it can.

    Shared by ``POST /admin/config`` (422) and the configuration test (fail).
    """
    embedding_provider = (embedding_provider or "").lower()
    llm_provider = (llm_provider or "").lower()
    if "ollama" in (embedding_provider, llm_provider):
        return "The chat API key cannot be reused with Ollama, which uses no API key."
    if embedding_provider != llm_provider:
        return "The chat API key can only be reused when chat and embedding providers are the same."
    return None


def embedding_api_key(provider: str, settings: Settings) -> str:
    """The key used for embeddings.

    With ``embedding_reuse_chat_api_key`` on: the chat key for ``provider``,
    ignoring any stored embedding key. Otherwise the Moodle embedding key, else
    the chat key (Moodle, then ``.env``).
    """
    if provider == "openai":
        chat_key, embedding_key = settings.openai_api_key, settings.embedding_openai_api_key
    elif provider == "gemini":
        chat_key, embedding_key = settings.gemini_api_key, settings.embedding_gemini_api_key
    else:
        return ""
    if settings.embedding_reuse_chat_api_key:
        return chat_key
    return embedding_key or chat_key


def _embed_openai(texts: list[str], model: str, settings: Settings) -> list[list[float]]:
    """Embed with the OpenAI-compatible ``/embeddings`` endpoint, preserving input order."""
    api_key = embedding_api_key("openai", settings)
    if not api_key:
        raise EmbeddingError("No OpenAI API key is configured for embeddings.")
    vectors: list[list[float]] = []
    for batch in _batches(texts, _OPENAI_BATCH):
        response = requests.post(
            f"{settings.openai_base_url.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": batch},
            timeout=settings.openai_timeout_seconds,
        )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        vectors.extend(_check_count([row["embedding"] for row in rows], batch, "OpenAI"))
    return vectors


def _embed_ollama(texts: list[str], model: str, settings: Settings,
                  task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed with a local Ollama server's ``/api/embed`` endpoint (nomic task prefixes added)."""
    if uses_nomic_prefixes("ollama", model):
        prefix = _NOMIC_PREFIXES.get(task_type, _NOMIC_PREFIXES["RETRIEVAL_DOCUMENT"])
        texts = [prefix + text for text in texts]
    vectors: list[list[float]] = []
    for batch in _batches(texts, _OLLAMA_BATCH):
        response = requests.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/embed",
            json={"model": model, "input": batch},
            timeout=settings.ollama_timeout_seconds,
        )
        response.raise_for_status()
        vectors.extend(_check_count(response.json().get("embeddings", []), batch, "Ollama"))
    return vectors


def _embed_gemini(texts: list[str], model: str, settings: Settings,
                  task_type: str) -> list[list[float]]:
    """Embed with the Gemini SDK, using ``task_type`` and the configured output dimensions."""
    from google import genai
    from google.genai import types

    api_key = embedding_api_key("gemini", settings)
    if not api_key:
        raise EmbeddingError("No Gemini API key is configured for embeddings.")
    client = genai.Client(api_key=api_key)
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=settings.gemini_embed_dimensions or None,
    )
    vectors: list[list[float]] = []
    for batch in _batches(texts, _GEMINI_BATCH):
        result = client.models.embed_content(model=model.removeprefix("models/"), contents=batch, config=config)
        vectors.extend(_check_count([item.values for item in (result.embeddings or [])], batch, "Gemini"))
    return vectors


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT",
                settings: Settings | None = None) -> list[list[float]]:
    """Return one embedding vector per text, using the configured embedding provider.

    ``task_type`` is ``RETRIEVAL_DOCUMENT`` for indexing and ``RETRIEVAL_QUERY``
    for questions; Gemini receives it as a parameter and Ollama nomic models as a
    text prefix. ``settings`` overrides the stored
    configuration (the configuration test passes unsaved values). All failures
    are raised as EmbeddingError (HTTP errors include up to 1000 characters of
    the provider's response).
    """
    if not texts:
        return []
    settings = settings or get_settings()
    provider = settings.embedding_provider.lower()
    model = settings.resolved_embedding_model
    if provider not in {"openai", "gemini", "ollama"}:
        raise EmbeddingError(f"Unsupported embedding provider: {provider}")
    if not model:
        raise EmbeddingError(f"No embedding model configured for provider '{provider}'.")
    total_started = time.perf_counter()
    logger.info(
        "embedding.start provider=%s model=%s items=%s task=%s",
        provider, model, len(texts), task_type,
    )
    try:
        if provider == "openai":
            vectors = _embed_openai(texts, model, settings)
        elif provider == "ollama":
            vectors = _embed_ollama(texts, model, settings, task_type)
        else:
            vectors = _embed_gemini(texts, model, settings, task_type)
        logger.info(
            "embedding.finish provider=%s model=%s vectors=%s duration_ms=%.2f",
            provider, model, len(vectors), (time.perf_counter() - total_started) * 1000,
        )
        return vectors
    except EmbeddingError:
        logger.exception("embedding.failed provider=%s model=%s", provider, model)
        raise
    except requests.HTTPError as exc:
        logger.exception("embedding.http_error provider=%s model=%s", provider, model)
        body = exc.response.text[:1000] if exc.response is not None else str(exc)
        raise EmbeddingError(f"{provider} embeddings failed: {body}") from exc
    except Exception as exc:
        logger.exception("embedding.error provider=%s model=%s", provider, model)
        raise EmbeddingError(f"{provider} embedding request failed: {type(exc).__name__}: {exc}") from exc
