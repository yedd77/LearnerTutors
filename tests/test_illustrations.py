import re
import time
import uuid

import pytest

import app.services.illustrations as illustrations

PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"


def stored_url() -> str:
    return illustrations._store_image(PNG)


def parts(url: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"/api/v1/illustrations/([0-9a-f]{32})\?expires=(\d+)&signature=([0-9a-f]{64})", url)
    assert match, url
    return match.groups()


def test_valid_signed_url_serves_the_image_without_an_api_key(client):
    response = client.get(stored_url())  # deliberately no X-RAG-API-Key header
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == PNG


def test_url_shape_and_expiry_window():
    image_id, expires, _ = parts(stored_url())
    remaining = int(expires) - int(time.time())
    assert 0 < remaining <= illustrations.IMAGE_TTL
    assert (illustrations.IMAGE_DIR / f"{image_id}.png").read_bytes() == PNG


def test_signature_depends_on_the_server_secret(configure):
    image_id, expires, signature = parts(stored_url())
    assert illustrations._signature(image_id, int(expires)) == signature
    configure(RAG_API_KEY="a-different-secret-" + "k" * 30)
    assert illustrations._signature(image_id, int(expires)) != signature


def test_tampered_signature_is_rejected(client):
    image_id, expires, signature = parts(stored_url())
    flipped = ("0" if signature[0] != "0" else "1") + signature[1:]
    response = client.get(f"/api/v1/illustrations/{image_id}?expires={expires}&signature={flipped}")
    assert response.status_code == 403


def test_signature_for_another_image_is_rejected(client):
    first_id, expires, _ = parts(stored_url())
    _, _, other_signature = parts(stored_url())
    response = client.get(f"/api/v1/illustrations/{first_id}?expires={expires}&signature={other_signature}")
    assert response.status_code == 403


def test_extending_expiry_invalidates_the_signature(client):
    image_id, expires, signature = parts(stored_url())
    response = client.get(f"/api/v1/illustrations/{image_id}?expires={int(expires) + 100}&signature={signature}")
    assert response.status_code == 403


def test_expired_link_is_rejected_even_with_a_valid_signature(client):
    image_id, _, _ = parts(stored_url())
    past = int(time.time()) - 10
    signature = illustrations._signature(image_id, past)
    response = client.get(f"/api/v1/illustrations/{image_id}?expires={past}&signature={signature}")
    assert response.status_code == 403
    assert "expired" in response.json()["detail"]


def test_implausibly_distant_expiry_is_rejected(client):
    image_id, _, _ = parts(stored_url())
    far = int(time.time()) + illustrations.IMAGE_TTL + 10_000
    signature = illustrations._signature(image_id, far)
    assert client.get(f"/api/v1/illustrations/{image_id}?expires={far}&signature={signature}").status_code == 403


@pytest.mark.parametrize("image_id", ["../../etc/passwd", "g" * 32, "A" * 32, "a" * 31, "a" * 33])
def test_malformed_image_ids_are_404_and_never_touch_the_filesystem(client, image_id):
    signature = "0" * 64
    response = client.get(f"/api/v1/illustrations/{image_id}?expires={int(time.time()) + 100}&signature={signature}")
    assert response.status_code == 404


def test_valid_signature_but_missing_file_is_404(client):
    image_id = uuid.uuid4().hex
    expires = int(time.time()) + 1000
    signature = illustrations._signature(image_id, expires)
    response = client.get(f"/api/v1/illustrations/{image_id}?expires={expires}&signature={signature}")
    assert response.status_code == 404


@pytest.mark.parametrize("signature", ["short", "g" * 64 + "x", ""])
def test_wrong_length_signature_fails_validation(client, signature):
    image_id, expires, _ = parts(stored_url())
    response = client.get(f"/api/v1/illustrations/{image_id}?expires={expires}&signature={signature}")
    assert response.status_code == 422


def test_missing_query_parameters_fail_validation(client):
    assert client.get(f"/api/v1/illustrations/{'a' * 32}").status_code == 422


def test_empty_image_data_is_refused():
    with pytest.raises(RuntimeError, match="empty image"):
        illustrations._store_image(b"")


def test_signing_requires_a_configured_secret(configure):
    configure(RAG_API_KEY="")
    with pytest.raises(RuntimeError, match="RAG_API_KEY"):
        illustrations._signature("a" * 32, int(time.time()) + 100)


def test_unsupported_image_provider_is_reported(configure):
    configure(MOODLE_RUNTIME_IMAGE_PROVIDER="midjourney")
    with pytest.raises(RuntimeError):
        illustrations._generate_image_bytes("q", "a", "c")
