import pytest

import app.api.documents as documents_api
import app.services.vector_store as vs

URL = "/api/v1/documents/index"
TEXT = "Photosynthesis converts light energy into chemical energy in chloroplasts."


def upload(client, auth, *, name="bio.txt", content=TEXT.encode(), document_id="doc-1",
           course_id=1, owner_id=10, visibility="course", **extra):
    """Upload a course document. Pass visibility=None to omit the field entirely."""
    data = {"document_id": document_id, "course_id": str(course_id), "owner_id": str(owner_id)}
    if visibility is not None:
        data["visibility"] = visibility
    data.update({k: str(v) for k, v in extra.items()})
    return client.post(URL, headers=auth, files={"file": (name, content, "text/plain")}, data=data)


def test_index_success_returns_201_and_makes_the_document_searchable(client, auth):
    response = upload(client, auth)
    assert response.status_code == 201
    body = response.json()
    assert body == {
        "document_id": "doc-1", "filename": "bio.txt", "course_id": 1,
        "indexed_chunks": 1, "status": "ready", "suggestions": [],
    }
    rows = vs.search_course("photosynthesis", 1, True, 5)
    assert [r["metadata"]["document_id"] for r in rows] == ["doc-1"]


def test_course_upload_is_visible_to_every_student_in_the_course(client, auth):
    upload(client, auth, owner_id=10)
    [row] = vs.search_course("photosynthesis", 1, True, 5)
    assert row["metadata"]["visibility"] == "course"
    assert row["metadata"]["owner_id"] == 10


@pytest.mark.parametrize("visibility", ["personal", None])
def test_personal_uploads_are_rejected_and_nothing_is_indexed(client, auth, visibility):
    # None = field omitted, which used to default to "personal". It must not be
    # silently reinterpreted as a course-wide upload.
    response = upload(client, auth, visibility=visibility)
    assert response.status_code == 400
    assert "Personal document uploads are not supported" in response.json()["detail"]
    assert vs.get_document_retrieval_status("doc-1")["total_chunks"] == 0


def test_path_components_in_filename_are_stripped(client, auth):
    response = upload(client, auth, name="../../secret/bio.txt")
    assert response.json()["filename"] == "bio.txt"


@pytest.mark.parametrize("name", ["virus.exe", "photo.png", "noextension"])
def test_unsupported_file_type_is_415(client, auth, name):
    assert upload(client, auth, name=name).status_code == 415


def test_empty_file_is_400(client, auth):
    assert upload(client, auth, content=b"").status_code == 400


def test_oversized_file_is_413(client, auth):
    assert upload(client, auth, content=b"a" * (1024 * 1024 + 1)).status_code == 413


def test_file_without_extractable_text_is_422(client, auth):
    response = upload(client, auth, content=b"   \n\t  ")
    assert response.status_code == 422
    assert "No extractable text" in response.json()["detail"]


def test_too_many_chunks_is_413(client, auth):
    response = upload(client, auth, content=("word " * 20000).encode())  # ~100k chars > 50 chunks
    assert response.status_code == 413
    assert "chunks" in response.json()["detail"]


@pytest.mark.parametrize("visibility", ["public", "Course", "", " course"])
def test_any_visibility_other_than_course_is_400(client, auth, visibility):
    assert upload(client, auth, visibility=visibility).status_code == 400


@pytest.mark.parametrize("override", [{"course_id": 0}, {"owner_id": 0}, {"document_id": ""}])
def test_invalid_form_values_are_422(client, auth, override):
    assert upload(client, auth, **override).status_code == 422


def test_missing_form_fields_are_422(client, auth):
    response = client.post(URL, headers=auth, files={"file": ("a.txt", b"hello", "text/plain")},
                           data={"visibility": "course"})
    assert response.status_code == 422


def test_same_document_id_in_another_course_is_409(client, auth):
    assert upload(client, auth, course_id=1).status_code == 201
    response = upload(client, auth, course_id=2, content=b"hijack attempt text")
    assert response.status_code == 409
    assert vs.search_course("hijack", 2, True, 5) == []


def test_reindexing_replaces_content(client, auth):
    upload(client, auth, content=b"first version about mitosis")
    upload(client, auth, content=b"second version about osmosis")
    rows = vs.search_course("version", 1, True, 5)
    assert [r["text"] for r in rows] == ["second version about osmosis"]


def test_retrieval_disabled_at_upload(client, auth):
    upload(client, auth, retrieval_enabled="false")
    assert vs.search_course("photosynthesis", 1, True, 5) == []


def test_indexing_failure_is_502(client, auth, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("embedding provider down")

    monkeypatch.setattr(documents_api, "index_document", boom)
    response = upload(client, auth)
    assert response.status_code == 502
    assert "Indexing failed" in response.json()["detail"]


# --------------------------------------------------------------------------- suggestions
def test_suggestions_are_not_generated_by_default(client, auth, monkeypatch):
    called = []
    monkeypatch.setattr(documents_api, "generate_document_suggestions", lambda *a: called.append(a) or ["q"])
    assert upload(client, auth).json()["suggestions"] == []
    assert called == []


def test_suggestions_returned_when_requested(client, auth, monkeypatch):
    monkeypatch.setattr(documents_api, "generate_document_suggestions", lambda *a: ["What is X?", "Why Y?"])
    response = upload(client, auth, generate_suggestions="true")
    assert response.status_code == 201
    assert response.json()["suggestions"] == ["What is X?", "Why Y?"]


def test_suggestion_failure_does_not_fail_indexing(client, auth, monkeypatch):
    def boom(*args):
        raise RuntimeError("llm down")

    monkeypatch.setattr(documents_api, "generate_document_suggestions", boom)
    response = upload(client, auth, generate_suggestions="true")
    assert response.status_code == 201
    assert response.json()["suggestions"] == []
    assert vs.get_document_retrieval_status("doc-1")["total_chunks"] == 1


def test_regenerate_suggestions_does_not_touch_vectors(client, auth, monkeypatch):
    monkeypatch.setattr(documents_api, "generate_document_suggestions", lambda *a: ["Q1"])
    response = client.post(
        "/api/v1/documents/suggestions", headers=auth, data={"document_id": "doc-1"},
        files={"file": ("bio.txt", TEXT.encode(), "text/plain")},
    )
    assert response.status_code == 200
    assert response.json() == {"document_id": "doc-1", "suggestions": ["Q1"]}
    assert vs.get_document_retrieval_status("doc-1")["total_chunks"] == 0


@pytest.mark.parametrize("slot", [0, 4, -1])
def test_single_suggestion_slot_must_be_1_to_3(client, auth, slot):
    response = client.post(
        f"/api/v1/documents/doc-1/suggestions/{slot}", headers=auth,
        files={"file": ("bio.txt", TEXT.encode(), "text/plain")},
    )
    assert response.status_code == 422


def test_single_suggestion_success(client, auth, monkeypatch):
    seen = {}

    def fake(filename, sections, existing):
        seen.update(filename=filename, existing=existing)
        return "A fresh question?"

    monkeypatch.setattr(documents_api, "generate_one_document_suggestion", fake)
    response = client.post(
        "/api/v1/documents/doc-1/suggestions/2", headers=auth,
        data={"existing_suggestions": '["Old one?"]'},
        files={"file": ("bio.txt", TEXT.encode(), "text/plain")},
    )
    assert response.status_code == 200
    assert response.json() == {"document_id": "doc-1", "slot": 2, "suggestion": "A fresh question?"}
    assert seen == {"filename": "bio.txt", "existing": ["Old one?"]}


# --------------------------------------------------------------------------- delete / toggle
def test_delete_removes_vectors(client, auth):
    upload(client, auth)
    response = client.delete("/api/v1/documents/doc-1?course_id=1", headers=auth)
    assert response.status_code == 200
    assert response.json() == {"document_id": "doc-1", "deleted_chunks": 1, "status": "deleted"}
    assert vs.search_course("photosynthesis", 1, True, 5) == []


def test_delete_in_wrong_course_is_404_and_keeps_data(client, auth):
    upload(client, auth)
    assert client.delete("/api/v1/documents/doc-1?course_id=2", headers=auth).status_code == 404
    assert vs.get_document_retrieval_status("doc-1")["total_chunks"] == 1


def test_delete_requires_course_id(client, auth):
    assert client.delete("/api/v1/documents/doc-1", headers=auth).status_code == 422


def test_retrieval_toggle_round_trip(client, auth):
    upload(client, auth)
    off = client.patch("/api/v1/documents/doc-1/retrieval?course_id=1", headers=auth, json={"enabled": False})
    assert off.status_code == 200
    assert off.json() == {"document_id": "doc-1", "retrieval_enabled": False, "updated_chunks": 1}
    assert vs.search_course("photosynthesis", 1, True, 5) == []
    on = client.patch("/api/v1/documents/doc-1/retrieval?course_id=1", headers=auth, json={"enabled": True})
    assert on.json()["retrieval_enabled"] is True
    assert len(vs.search_course("photosynthesis", 1, True, 5)) == 1


def test_retrieval_toggle_in_wrong_course_is_404(client, auth):
    upload(client, auth)
    response = client.patch("/api/v1/documents/doc-1/retrieval?course_id=2", headers=auth, json={"enabled": False})
    assert response.status_code == 404
    assert len(vs.search_course("photosynthesis", 1, True, 5)) == 1
