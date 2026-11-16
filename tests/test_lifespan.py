"""App lifespan: Chroma warm-up on startup, retrieval pool shutdown on exit."""
import logging

from fastapi.testclient import TestClient

import app.services.retrieval as retrieval
import app.services.vector_store as vector_store
from app.main import app as fastapi_app
from helpers import chat_body, index_text


def messages(caplog):
    return [r.getMessage() for r in caplog.records]


def test_no_deprecated_startup_hooks():
    assert fastapi_app.router.on_startup == [] and fastapi_app.router.on_shutdown == []


def test_startup_warms_store_and_shutdown_stops_pool(caplog, auth, stub_generation):
    index_text("bio", "Photosynthesis converts light energy into chemical energy.")
    with caplog.at_level(logging.INFO, logger="app.requests"):
        with TestClient(fastapi_app) as client:
            assert any(m.startswith("startup.vector_store.ready chunks=1") for m in messages(caplog))
            response = client.post("/api/v1/chat", headers=auth, json=chat_body("What is photosynthesis?"))
            assert response.status_code == 200
            pool = retrieval.get_retrieval_executor()
        assert pool._shutdown
        assert "shutdown.retrieval_executor stopped=True" in messages(caplog)


def test_app_can_restart_in_the_same_process(auth, stub_generation):
    index_text("bio", "Photosynthesis converts light energy into chemical energy.")
    for _ in range(2):
        with TestClient(fastapi_app) as client:
            response = client.post("/api/v1/chat", headers=auth, json=chat_body("What is photosynthesis?"))
            assert response.status_code == 200 and response.json()["retrieved_chunks"] == 1


def test_warmup_failure_is_not_fatal(monkeypatch, caplog):
    def broken():
        raise RuntimeError("chroma is unavailable")

    monkeypatch.setattr(vector_store, "get_collection", broken)
    with caplog.at_level(logging.INFO, logger="app.requests"):
        with TestClient(fastapi_app) as client:
            assert any(m.startswith("startup.vector_store.failed") for m in messages(caplog))
            assert client.get("/health").status_code == 200
            ready = client.get("/ready")
            assert ready.status_code == 503 and ready.json()["status"] == "not_ready"
