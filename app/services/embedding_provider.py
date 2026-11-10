import requests
from app.config import get_settings


class EmbeddingError(RuntimeError):
    pass


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    if not texts:
        return []
    settings = get_settings()
    provider = settings.embedding_provider.lower()
    model = settings.resolved_embedding_model
    if provider not in {"openai", "gemini", "ollama"}:
        raise EmbeddingError(f"Unsupported embedding provider: {provider}")
    if not model:
        raise EmbeddingError(f"No embedding model configured for provider '{provider}'.")
    vectors: list[list[float]] = []
    try:
        if provider == "openai":
            if not settings.openai_api_key:
                raise EmbeddingError("OPENAI_API_KEY is not configured.")
            for start in range(0, len(texts), 100):
                batch = texts[start:start + 100]
                response = requests.post(
                    f"{settings.openai_base_url.rstrip('/')}/embeddings",
                    headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                    json={"model": model, "input": batch},
                    timeout=settings.openai_timeout_seconds,
                )
                response.raise_for_status()
                rows = sorted(response.json()["data"], key=lambda row: row["index"])
                batch_vectors = [row["embedding"] for row in rows]
                if len(batch_vectors) != len(batch):
                    raise EmbeddingError("OpenAI returned an unexpected embedding count.")
                vectors.extend(batch_vectors)
        elif provider == "ollama":
            for start in range(0, len(texts), 32):
                batch = texts[start:start + 32]
                response = requests.post(
                    f"{settings.ollama_base_url.rstrip('/')}/api/embed",
                    json={"model": model, "input": batch},
                    timeout=settings.ollama_timeout_seconds,
                )
                response.raise_for_status()
                batch_vectors = response.json().get("embeddings", [])
                if len(batch_vectors) != len(batch):
                    raise EmbeddingError("Ollama returned an unexpected embedding count.")
                vectors.extend(batch_vectors)
        else:
            from google import genai
            from google.genai import types
            if not settings.gemini_api_key:
                raise EmbeddingError("GEMINI_API_KEY is not configured.")
            client = genai.Client(api_key=settings.gemini_api_key)
            config = types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.gemini_embed_dimensions or None,
            )
            for start in range(0, len(texts), 100):
                batch = texts[start:start + 100]
                result = client.models.embed_content(model=model.removeprefix("models/"), contents=batch, config=config)
                batch_vectors = [item.values for item in (result.embeddings or [])]
                if len(batch_vectors) != len(batch):
                    raise EmbeddingError("Gemini returned an unexpected embedding count.")
                vectors.extend(batch_vectors)
        return vectors
    except EmbeddingError:
        raise
    except requests.HTTPError as exc:
        body = exc.response.text[:1000] if exc.response is not None else str(exc)
        raise EmbeddingError(f"{provider} embeddings failed: {body}") from exc
    except Exception as exc:
        raise EmbeddingError(f"{provider} embedding request failed: {type(exc).__name__}: {exc}") from exc
