"""Past exam paper uploads: structure only, never searchable, scoped to the course."""
import pytest

from app.services import exam_blueprint as eb
from app.services import llm_provider
import app.services.vector_store as vs

URL = "/api/v1/documents/index"
PAPER = (b"Section A. Answer ALL questions. 1. Which organelle carries out photosynthesis? "
         b"(1 mark)\nSection B. Answer any THREE questions. (a) Explain ... [20 marks]")
TEXT = b"Photosynthesis converts light energy into chemical energy in chloroplasts."


def upload(client, auth, *, content=PAPER, document_id="paper-1", course_id=1, name="past.txt",
           **extra):
    data = {"document_id": document_id, "course_id": str(course_id), "owner_id": "10",
            "visibility": "course"}
    data.update({k: str(v) for k, v in extra.items()})
    return client.post(URL, headers=auth, files={"file": (name, content, "text/plain")}, data=data)


def paper_blueprint() -> eb.ExamBlueprint:
    return eb.ExamBlueprint.model_validate({"sections": [
        {"title": "Section A", "question_type": "mcq", "question_count": 20, "marks_per_question": 1},
        {"title": "Section B", "question_type": "structured", "question_count": 5,
         "answer_count": 3, "marks_per_question": 20, "subparts_per_question": 3},
    ], "duration_minutes": 120})


@pytest.fixture
def analysis(monkeypatch, configure):
    """Stub the analysis LLM call; records how often it ran."""
    configure(MOODLE_RUNTIME_LLM_PROVIDER="openai", MOODLE_RUNTIME_LLM_MODEL="gpt-test")
    state = {"calls": 0, "result": eb.BlueprintAnalysis(is_exam_paper=True,
                                                        blueprint=paper_blueprint())}

    def fake(messages, schema, provider=None, model=None, **kwargs):
        state["calls"] += 1
        if isinstance(state["result"], Exception):
            raise state["result"]
        return state["result"]

    monkeypatch.setattr(llm_provider, "generate_structured_sync", fake)
    return state


def test_past_paper_stores_structure_and_is_never_searchable(client, auth, analysis):
    response = upload(client, auth, document_role="past_exam")
    assert response.status_code == 201
    body = response.json()
    assert body["indexed_chunks"] == 0 and body["document_role"] == "past_exam"
    assert body["suggestions"] == []
    summary = body["exam_blueprint"]
    assert summary["question_count"] == 25 and summary["total_marks"] == 80
    assert summary["duration_minutes"] == 120 and len(summary["sections"]) == 2
    assert "Section B: 5 structured questions, candidates answer 3" in summary["description"]
    assert vs.get_document_retrieval_status("paper-1")["total_chunks"] == 0
    assert vs.search_course("organelle photosynthesis", 1, True, 5) == []
    assert eb.course_plan(1).question_count() == 25


def test_ordinary_upload_response_is_unchanged(client, auth):
    body = upload(client, auth, content=TEXT, document_id="doc-1").json()
    assert set(body) == {"document_id", "filename", "course_id", "indexed_chunks", "status",
                         "suggestions"}
    body = upload(client, auth, content=TEXT, document_id="doc-2",
                  document_role="course_material").json()
    assert body["indexed_chunks"] == 1 and "exam_blueprint" not in body


def test_unknown_role_is_400(client, auth):
    assert upload(client, auth, document_role="answers").status_code == 400


def test_unchanged_paper_is_not_analysed_again(client, auth, analysis):
    upload(client, auth, document_role="past_exam")
    upload(client, auth, document_role="past_exam")
    assert analysis["calls"] == 1
    upload(client, auth, document_role="past_exam", content=PAPER + b" (revised)")
    assert analysis["calls"] == 2


def test_suggestions_are_never_generated_for_a_past_paper(client, auth, analysis, monkeypatch):
    import app.api.documents as documents_api
    monkeypatch.setattr(documents_api, "generate_document_suggestions",
                        lambda *a: pytest.fail("suggestions would expose past questions"))
    body = upload(client, auth, document_role="past_exam", generate_suggestions="true").json()
    assert body["suggestions"] == []


def test_switching_roles_removes_the_other_copy(client, auth, analysis):
    upload(client, auth, content=TEXT, document_id="doc-1")
    assert vs.get_document_retrieval_status("doc-1")["total_chunks"] == 1
    upload(client, auth, document_id="doc-1", document_role="past_exam")
    assert vs.get_document_retrieval_status("doc-1")["total_chunks"] == 0
    assert eb.load_record(1, "doc-1") is not None
    upload(client, auth, content=TEXT, document_id="doc-1")
    assert eb.load_record(1, "doc-1") is None
    assert vs.get_document_retrieval_status("doc-1")["total_chunks"] == 1


def test_not_an_exam_paper_is_422_and_nothing_is_stored(client, auth, analysis):
    analysis["result"] = eb.BlueprintAnalysis(is_exam_paper=False)
    response = upload(client, auth, document_role="past_exam")
    assert response.status_code == 422 and "does not look like an exam paper" in response.json()["detail"]
    assert eb.course_blueprints(1) == []


def test_analysis_failure_is_502_without_provider_details(client, auth, analysis):
    analysis["result"] = llm_provider.ProviderError("secret upstream detail sk-123")
    response = upload(client, auth, document_role="past_exam")
    assert response.status_code == 502
    assert "sk-123" not in response.text and "upstream" not in response.text
    assert eb.course_blueprints(1) == []


def test_document_id_owned_by_another_course_is_409(client, auth, analysis):
    upload(client, auth, document_role="past_exam", course_id=1)
    assert upload(client, auth, document_role="past_exam", course_id=2).status_code == 409
    assert upload(client, auth, content=TEXT, course_id=2).status_code == 409
    upload(client, auth, content=TEXT, document_id="doc-9", course_id=1)
    assert upload(client, auth, document_id="doc-9", document_role="past_exam",
                  course_id=2).status_code == 409
    assert eb.course_blueprints(2) == []


def test_disable_and_delete_apply_to_past_papers(client, auth, analysis):
    upload(client, auth, document_role="past_exam")
    path = "/api/v1/documents/paper-1/retrieval?course_id=1"
    assert client.patch(path, headers=auth, json={"enabled": False}).status_code == 200
    assert eb.course_plan(1) is None
    assert client.patch(path, headers=auth, json={"enabled": True}).status_code == 200
    assert eb.course_plan(1) is not None
    # Another course cannot touch it.
    assert client.patch("/api/v1/documents/paper-1/retrieval?course_id=2", headers=auth,
                        json={"enabled": False}).status_code == 404
    assert client.delete("/api/v1/documents/paper-1?course_id=2", headers=auth).status_code == 404
    response = client.delete("/api/v1/documents/paper-1?course_id=1", headers=auth)
    assert response.status_code == 200 and response.json()["deleted_chunks"] == 0
    assert eb.course_plan(1) is None
    assert client.delete("/api/v1/documents/paper-1?course_id=1", headers=auth).status_code == 404


def test_paper_uploaded_disabled_does_not_shape_exams(client, auth, analysis):
    upload(client, auth, document_role="past_exam", retrieval_enabled="false")
    assert eb.load_record(1, "paper-1")["enabled"] is False
    assert eb.course_plan(1) is None


def test_conflict_found_at_save_time_is_409_and_keeps_this_courses_data(client, auth, analysis,
                                                                          monkeypatch):
    upload(client, auth, content=TEXT, document_id="doc-1", course_id=1)
    # Simulate another course winning the claim between the early check and the save.
    def lose(*args, **kwargs):
        raise eb.BlueprintConflict("Document 'doc-1' belongs to another course.")
    monkeypatch.setattr(eb, "save_blueprint", lose)
    response = upload(client, auth, document_id="doc-1", course_id=1, document_role="past_exam")
    assert response.status_code == 409
    assert vs.get_document_retrieval_status("doc-1")["total_chunks"] == 1   # nothing removed
