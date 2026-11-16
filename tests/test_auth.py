import pytest

from helpers import chat_body

# (method, path, kwargs) for every API-key protected route.
PROTECTED = [
    ("GET", "/api/v1/chat/tools", {}),
    ("POST", "/api/v1/chat", {"json": chat_body("What is osmosis?")}),
    ("GET", "/api/v1/admin/config", {}),
    ("POST", "/api/v1/admin/connection-test", {}),
    ("POST", "/api/v1/admin/config", {"json": {"llm_provider": "ollama", "llm_model": "m"}}),
    ("DELETE", "/api/v1/documents/doc1?course_id=1", {}),
    ("PATCH", "/api/v1/documents/doc1/retrieval?course_id=1", {"json": {"enabled": False}}),
]


@pytest.mark.parametrize("method,path,kwargs", PROTECTED)
def test_missing_key_is_rejected(client, method, path, kwargs):
    assert client.request(method, path, **kwargs).status_code == 401


@pytest.mark.parametrize("method,path,kwargs", PROTECTED)
def test_wrong_key_is_rejected(client, method, path, kwargs):
    response = client.request(method, path, headers={"X-RAG-API-Key": "wrong" * 10}, **kwargs)
    assert response.status_code == 401


def test_key_that_is_only_a_prefix_is_rejected(client, auth):
    prefix = auth["X-RAG-API-Key"][:-1]
    assert client.get("/api/v1/chat/tools", headers={"X-RAG-API-Key": prefix}).status_code == 401


def test_valid_key_is_accepted(client, auth):
    response = client.get("/api/v1/chat/tools", headers=auth)
    assert response.status_code == 200
    assert {tool["id"] for tool in response.json()["tools"]} >= {"ask", "exam_generation"}


def test_connection_test_endpoint(client, auth):
    response = client.post("/api/v1/admin/connection-test", headers=auth)
    assert response.json() == {"status": "ok", "authenticated": True}


@pytest.mark.parametrize("short_key", ["", "too-short"])
def test_unconfigured_or_short_server_key_returns_503(client, configure, short_key):
    configure(RAG_API_KEY=short_key)
    # Even a request that "matches" the configured key must not be accepted.
    response = client.get("/api/v1/chat/tools", headers={"X-RAG-API-Key": short_key})
    assert response.status_code == 503


def test_health_is_public(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_chunk_count(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "indexed_chunks": 0}


def test_ready_returns_503_without_leaking_details(client, monkeypatch):
    import app.services.vector_store as vector_store

    def boom():
        raise RuntimeError("secret internal path C:/very/private")

    monkeypatch.setattr(vector_store, "get_collection", boom)
    response = client.get("/ready")
    assert response.status_code == 503
    assert "private" not in response.text
    assert response.json()["status"] == "not_ready"


def test_every_response_carries_a_request_id(client):
    assert client.get("/health").headers.get("X-Request-ID")
