"""Shared fixtures for the offline test suite.

Safety rules enforced here:
  * Environment overrides are applied BEFORE `app` is imported, so values from the
    developer's real `.env` (provider keys, RAG_API_KEY, ...) are never used.
  * Chroma, the runtime config file, the illustration and generated-file
    directories all live in
    temporary locations. The real `storage/` directory is never touched.
  * Provider calls (embeddings, chat generation, images) are stubbed. No network.
"""
import atexit
import hashlib
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_API_KEY = "test-api-key-" + "x" * 40  # >= 32 characters, as the backend requires
_SESSION_DIR = Path(tempfile.mkdtemp(prefix="learnertutors-tests-"))
atexit.register(shutil.rmtree, _SESSION_DIR, ignore_errors=True)  # fallback; see pytest_sessionfinish

# Empty strings deliberately override anything the real .env would supply.
os.environ.update({
    "APP_ENV": "development",
    "LOG_LEVEL": "WARNING",
    "RAG_API_KEY": TEST_API_KEY,
    "RUNTIME_CONFIG_ENCRYPTION_KEY": Fernet.generate_key().decode(),
    "CHROMA_PATH": str(_SESSION_DIR / "chroma"),
    "CHROMA_COLLECTION": "test_chunks",
    "EMBEDDING_PROVIDER": "openai",
    "EMBEDDING_MODEL": "",
    "OPENAI_EMBED_MODEL": "test-embed",
    "OPENAI_API_KEY": "",
    "GEMINI_API_KEY": "",
    "MOODLE_RUNTIME_LLM_PROVIDER": "",
    "MOODLE_RUNTIME_LLM_MODEL": "",
    "MOODLE_RUNTIME_IMAGE_MODEL": "",
    "MOODLE_RUNTIME_IMAGE_PROVIDER": "openai",
    "MOODLE_RUNTIME_REUSE_CHAT_API_KEY": "false",
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
    "TOP_K": "5",
    # Real embedding distances are model-specific; the fake embeddings are not calibrated.
    "RETRIEVAL_MAX_DISTANCE": "0",
    "CHUNK_SIZE": "900",
    "CHUNK_OVERLAP": "150",
    "MAX_UPLOAD_MB": "1",
    "MAX_PAGES": "300",
    "MAX_CHUNKS": "50",
    "RETRIEVAL_WORKERS": "2",
    # Chroma would otherwise send anonymous usage telemetry over the network.
    "ANONYMIZED_TELEMETRY": "False",
})

from fastapi.testclient import TestClient  # noqa: E402

import app.api.admin as admin_module  # noqa: E402
import app.config as config_module  # noqa: E402
import app.services.illustrations as illustrations_module  # noqa: E402
import app.services.temp_files as temp_files_module  # noqa: E402
import app.services.vector_store as vector_store  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402

# Captured at import so teardown still works when a test monkeypatches get_collection.
_real_get_collection = vector_store.get_collection


def pytest_sessionfinish(session, exitstatus):
    """Release Chroma's open SQLite handle so the temp directory can be deleted on Windows."""
    import gc

    vector_store.reset_vector_store_cache()
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:  # pragma: no cover - API differs between chromadb versions
        pass
    gc.collect()
    shutil.rmtree(_SESSION_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# Offline guard: any non-loopback connection attempt fails the test immediately.
# ---------------------------------------------------------------------------
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    import socket

    real_connect = socket.socket.connect

    def guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK:
            raise AssertionError(f"Test attempted a network connection to {address!r}")
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


# ---------------------------------------------------------------------------
# Settings isolation
# ---------------------------------------------------------------------------
def reset_settings_cache() -> None:
    config_module._settings_cache = None
    config_module._settings_cache_mtime = None


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    """Per-test runtime config file, fresh settings cache, empty vector store."""
    runtime_path = tmp_path / "runtime_config.json"
    monkeypatch.setattr(config_module, "RUNTIME_CONFIG_PATH", runtime_path)
    monkeypatch.setattr(admin_module, "RUNTIME_CONFIG_PATH", runtime_path)
    monkeypatch.setattr(illustrations_module, "IMAGE_DIR", tmp_path / "illustrations")
    monkeypatch.setattr(temp_files_module, "GENERATED_DIR", tmp_path / "generated")
    reset_settings_cache()
    yield runtime_path
    reset_settings_cache()
    # Keep one shared Chroma client (fast) but start every test empty.
    collection = _real_get_collection()
    ids = collection.get(include=[])["ids"]
    if ids:
        collection.delete(ids=ids)


@pytest.fixture
def configure(monkeypatch):
    """Override settings via environment variables, e.g. configure(CHUNK_SIZE="300")."""
    def apply(**env: str) -> None:
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        reset_settings_cache()
    return apply


@pytest.fixture
def runtime_path(isolated_runtime) -> Path:
    return isolated_runtime


# ---------------------------------------------------------------------------
# Fake embeddings: hashed bag of words, so similar texts have similar vectors.
# ---------------------------------------------------------------------------
_DIMS = 64


def fake_vector(text: str) -> list[float]:
    vector = [0.0] * _DIMS
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIMS
        vector[bucket] += 1.0
    vector[0] += 0.01  # never an all-zero vector (undefined cosine distance)
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    calls: list[dict] = []

    def fake_embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
        calls.append({"texts": list(texts), "task_type": task_type})
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(vector_store, "embed_texts", fake_embed_texts)
    return calls


# ---------------------------------------------------------------------------
# HTTP client and generation stubs
# ---------------------------------------------------------------------------
@pytest.fixture
def client() -> TestClient:
    # Not used as a context manager: startup warm-up is not needed here.
    return TestClient(fastapi_app)


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-RAG-API-Key": TEST_API_KEY}


@pytest.fixture
def stub_generation(monkeypatch):
    """Replace chat generation. Records every call; set `.error` to simulate failure."""
    import app.services.llm_provider as llm_provider

    class Stub:
        def __init__(self):
            self.calls: list[dict] = []
            self.answer = "Stubbed answer [S1]."
            self.error: Exception | None = None

    stub = Stub()

    async def fake_generate_answer_async(route, context, history=None, provider=None, model=None,
                                         *, fallback=None):
        stub.calls.append({
            "route": route, "context": context, "history": history,
            "provider": provider, "model": model, "fallback": fallback,
        })
        if stub.error is not None:
            raise stub.error
        return stub.answer

    monkeypatch.setattr(llm_provider, "generate_answer_async", fake_generate_answer_async)
    return stub


@pytest.fixture
def stub_illustration(monkeypatch):
    import app.api.chat as chat_module

    class Stub:
        def __init__(self):
            self.calls = 0
            self.error: Exception | None = None
            self.url = "/api/v1/illustrations/" + "a" * 32 + "?expires=1&signature=" + "b" * 64
            # The "would an image help?" check; says yes unless changed.
            self.needs_image = True
            self.check_error: Exception | None = None
            self.checks: list[dict] = []

    stub = Stub()

    async def fake_generate_illustration_async(question, answer, context):
        stub.calls += 1
        if stub.error is not None:
            raise stub.error
        return stub.url

    real_structured = chat_module.llm_provider.generate_structured

    async def fake_structured(messages, schema, provider=None, model=None, **kwargs):
        if schema is not chat_module.IllustrationDecision:
            return await real_structured(messages, schema, provider, model, **kwargs)
        stub.checks.append({"messages": messages, "provider": provider, "model": model,
                            "fallback": kwargs.get("fallback")})
        if stub.check_error is not None:
            raise stub.check_error
        return schema(needs_image=stub.needs_image, reason="stub")

    monkeypatch.setattr(chat_module, "generate_illustration_async", fake_generate_illustration_async)
    monkeypatch.setattr(chat_module.llm_provider, "generate_structured", fake_structured)
    return stub
