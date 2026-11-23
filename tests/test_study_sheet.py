"""/notes study sheet PDFs: sufficiency, the approve/decline flow, temporary downloads."""
import json
import time

import pymupdf
import pytest

import app.services.llm_provider as llm_provider
import app.services.vector_store as vector_store
from app.services import pending_actions, temp_files
from app.services import study_sheet as sheets
from app.services.llm_provider import ProviderError
from helpers import chat_body, index_text

CHAT = "/api/v1/chat"
FILES = "/api/v1/generated-files"
TOPIC = "photosynthesis light energy chloroplast glucose"


def material(n: int, course_id: int = 1, words: int = 60) -> None:
    """Index ``n`` on-topic chunks of about ``words`` words each."""
    for i in range(n):
        body = " ".join([f"Photosynthesis fact {i}:"] + ["light energy chloroplast glucose"] * (words // 4))
        index_text(f"c{course_id}doc{i}", body, course_id=course_id, filename=f"Biology_{i}.pdf" if course_id == 1 else f"Course{course_id}_{i}.pdf", page=i + 1)


def sample_sheet(**overrides) -> sheets.StudySheet:
    data = dict(
        title="Photosynthesis", coverage="sufficient",
        overview="Plants convert light energy into chemical energy stored in glucose.",
        key_concepts=["Happens in chloroplasts", "Two stages: light reactions and Calvin cycle"],
        definitions=[{"term": "Chlorophyll", "meaning": "Green pigment that absorbs light"}],
        main_points=["Water is split, releasing O₂", "CO₂ is fixed in the stroma"],
        examples=["Greenhouses add CO₂ to raise yield"],
        formulas=[{"expression": "6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂",
                   "meaning": "Overall equation"}],
        remember=["Oxygen comes from water"],
        summary=["Light energy becomes chemical energy"],
    )
    data.update(overrides)
    return sheets.StudySheet(**data)


@pytest.fixture
def sheet_llm(monkeypatch):
    """Stub the structured LLM call that writes the sheet; records each call."""

    class Stub:
        calls: list = []
        sheet = sample_sheet()
        error: Exception | None = None

    stub = Stub()
    stub.calls = []

    async def fake_structured(messages, schema, provider=None, model=None, *, timeout=None,
                              fallback=None):
        assert schema is sheets.StudySheet, f"unexpected structured call: {schema.__name__}"
        stub.calls.append({"messages": messages, "provider": provider, "model": model})
        if stub.error is not None:
            raise stub.error
        return stub.sheet

    monkeypatch.setattr(llm_provider, "generate_structured", fake_structured)
    return stub


@pytest.fixture
def searches(monkeypatch):
    """Record hybrid_search queries while running the real search."""
    calls: list[dict] = []
    real = vector_store.hybrid_search

    def spy(queries, course_id, include_course_documents, top_k):
        for query in queries:
            calls.append({"query": query, "course_id": course_id, "top_k": top_k})
        return real(queries, course_id, include_course_documents, top_k)

    monkeypatch.setattr(vector_store, "hybrid_search", spy)
    return calls


def chat(client, auth, question=f"/notes {TOPIC}", **extra):
    response = client.post(CHAT, headers=auth, json=chat_body(question, **extra))
    assert response.status_code == 200, response.text
    return response.json()


def download(client, auth, file_id, course_id=1, user_id=10):
    return client.get(f"{FILES}/{file_id}", headers=auth,
                      params={"course_id": course_id, "user_id": user_id})


def stored_files() -> list[str]:
    if not temp_files.GENERATED_DIR.is_dir():
        return []
    return sorted(p.name for p in temp_files.GENERATED_DIR.iterdir())


# --------------------------------------------------------------------------- sufficient
def test_sufficient_material_returns_a_downloadable_one_page_pdf(client, auth, sheet_llm):
    material(5)
    body = chat(client, auth)
    item = body["download"]
    assert item["content_type"] == "application/pdf" and item["pages"] == 1
    assert item["filename"] == "Photosynthesis_study_sheet.pdf"
    assert item["ttl_minutes"] == 60 and item["expires_at"] > time.time()
    assert "pending_action" not in body
    assert body["answer"].startswith("Your study sheet is ready!")
    assert "not stored permanently" in body["answer"] and "1 hour" in body["answer"]
    assert body["sources"] and body["retrieved_chunks"] == 5

    response = download(client, auth, item["file_id"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "no-store" in response.headers["cache-control"]
    assert response.content.startswith(b"%PDF")
    doc = pymupdf.open(stream=response.content, filetype="pdf")
    assert doc.page_count == 1
    text = doc[0].get_text()
    for expected in ("Photosynthesis", "Topic overview", "Key concepts", "Important definitions",
                     "Formulas & equations", "Remember this", "Quick revision summary",
                     "Biology_0.pdf", "Page 1 of 1", "LearnerTutors"):
        assert expected in text, expected


def test_sheet_prompt_is_grounded_in_retrieved_course_text(client, auth, sheet_llm):
    material(4)
    material(1, course_id=2)  # another course's text must never reach the prompt
    index_text("other", "Secret course two photosynthesis glucose", course_id=2)
    chat(client, auth)
    [call] = sheet_llm.calls
    system, user = call["messages"][0]["content"], call["messages"][1]["content"]
    assert "Do not invent facts" in system           # core grounding rules are included
    assert "Biology_0.pdf" in user and "Secret course two" not in user
    assert TOPIC in user and "/notes" not in user
    assert call["provider"] == "openai" and call["model"] == "test-model"


def test_other_tools_are_unchanged(client, auth, stub_generation, sheet_llm):
    material(5)
    body = chat(client, auth, f"/summarize {TOPIC}")
    assert body["answer"] == "Stubbed answer [S1]." and "download" not in body
    assert sheet_llm.calls == [] and stored_files() == []
    assert stub_generation.calls[0]["route"].tool.id == "summarization"


# --------------------------------------------------------------------------- insufficient
def test_too_little_material_asks_permission_without_calling_the_llm(client, auth, sheet_llm):
    index_text("tiny", "Photosynthesis uses light.")
    body = chat(client, auth)
    assert "download" not in body and stored_files() == []
    assert sheet_llm.calls == []
    assert "limited information" in body["answer"] and "Would you like me" in body["answer"]
    action = body["pending_action"]
    assert action["type"] == "expand_notes_search"
    assert action["approve_label"] and action["decline_label"]
    assert body["sources"] == [] and body["retrieved_contexts"] == []


def test_model_judging_coverage_limited_also_asks_permission(client, auth, sheet_llm):
    material(5)
    sheet_llm.sheet = sample_sheet(coverage="limited")
    body = chat(client, auth)
    assert body["pending_action"]["type"] == "expand_notes_search"
    assert "download" not in body and stored_files() == []


def test_approval_widens_the_search_in_the_same_course_and_makes_the_pdf(
        client, auth, sheet_llm, searches, configure):
    configure(NOTES_MIN_CHUNKS="11", NOTES_MIN_CONTEXT_CHARS="0")
    material(12)
    material(3, course_id=2)
    first = chat(client, auth)
    assert first["pending_action"] and searches[-1]["top_k"] == 10

    second = chat(client, auth, "Yes, search more widely",
                  pending_action_token=first["pending_action"]["token"],
                  pending_decision="approve")
    assert second["download"]["pages"] == 1
    assert searches[-1] == {"query": TOPIC, "course_id": 1, "top_k": 20}
    assert second["retrieved_chunks"] == 12   # course 2's chunks were never searched
    assert download(client, auth, second["download"]["file_id"]).status_code == 200


def test_decline_cancels_and_generates_nothing(client, auth, sheet_llm, searches):
    index_text("tiny", "Photosynthesis uses light.")
    first = chat(client, auth)
    count = len(searches)
    body = chat(client, auth, "No, cancel", pending_action_token=first["pending_action"]["token"],
                pending_decision="decline")
    assert "cancelled" in body["answer"]
    assert "download" not in body and "pending_action" not in body
    assert len(searches) == count and sheet_llm.calls == [] and stored_files() == []


def test_widened_search_still_insufficient_is_reported_honestly(client, auth, sheet_llm):
    index_text("tiny", "Photosynthesis uses light.")
    first = chat(client, auth)
    body = chat(client, auth, "Yes", pending_action_token=first["pending_action"]["token"],
                pending_decision="approve")
    assert "still isn't enough material" in body["answer"]
    assert "download" not in body and "pending_action" not in body  # no endless re-asking
    assert stored_files() == []


def test_free_text_yes_is_not_consent(client, auth, sheet_llm, stub_generation, searches):
    index_text("tiny", "Photosynthesis uses light.")
    chat(client, auth)
    body = chat(client, auth, "yes please",
                history=[{"role": "assistant", "content": sheets.ask_to_expand_message(TOPIC)}])
    assert "download" not in body and stored_files() == []
    assert all(call["top_k"] != 20 for call in searches)


@pytest.mark.parametrize("change", ["other_user", "other_course", "tampered", "expired", "garbage"])
def test_invalid_tokens_never_run_the_request(client, auth, sheet_llm, searches, change,
                                              monkeypatch):
    material(5)
    token, _ = pending_actions.issue(sheets.EXPAND_ACTION, course_id=1, user_id=10,
                                     question=TOPIC, tool_id="note_making")
    extra = {"course_id": 1, "user_id": 10}
    if change == "other_user":
        extra["user_id"] = 11
    elif change == "other_course":
        extra["course_id"] = 2
    elif change == "tampered":
        body, sig = token.rsplit(".", 1)
        token = body[:-2] + ("AA" if body[-2:] != "AA" else "BB") + "." + sig
    elif change == "expired":
        later = time.time() + 31 * 60  # default TTL is 30 minutes
        monkeypatch.setattr(pending_actions.time, "time", lambda: later)
    elif change == "garbage":
        token = "not-a-token"
    body = chat(client, auth, "Yes", pending_action_token=token, pending_decision="approve",
                **extra)
    assert "expired or is no longer valid" in body["answer"]
    assert searches == [] and sheet_llm.calls == [] and stored_files() == []


def test_token_and_decision_must_come_together(client, auth):
    response = client.post(CHAT, headers=auth, json=chat_body("Yes", pending_decision="approve"))
    assert response.status_code == 422


# --------------------------------------------------------------------------- failures
def test_provider_failure_while_writing_the_sheet_is_a_502(client, auth, sheet_llm):
    material(5)
    sheet_llm.error = ProviderError("openai is unavailable", retryable=True)
    response = client.post(CHAT, headers=auth, json=chat_body(f"/notes {TOPIC}"))
    assert response.status_code == 502 and stored_files() == []


def test_render_failure_returns_a_polite_message(client, auth, sheet_llm, monkeypatch):
    import app.pdf.study_sheet as renderer

    def broken(*args, **kwargs):
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr(renderer, "render_study_sheet", broken)
    material(5)
    body = chat(client, auth)
    assert "couldn't create the study sheet" in body["answer"]
    assert "download" not in body and stored_files() == []


# --------------------------------------------------------------------------- downloads
@pytest.fixture
def stored(client, auth, sheet_llm):
    material(5)
    return chat(client, auth)["download"]["file_id"]


def test_another_student_cannot_download(client, auth, stored):
    assert download(client, auth, stored, user_id=11).status_code == 404
    assert download(client, auth, stored, course_id=2).status_code == 404
    assert download(client, auth, stored).status_code == 200


def test_download_requires_the_api_key(client, stored):
    response = client.get(f"{FILES}/{stored}", params={"course_id": 1, "user_id": 10})
    assert response.status_code == 401


@pytest.mark.parametrize("file_id", ["../../runtime_config", "x" * 32, "A" * 32, "0" * 31])
def test_malformed_or_unknown_ids_are_not_found(client, auth, stored, file_id):
    assert download(client, auth, file_id).status_code in (404, 405)


def test_expired_file_is_refused_and_deleted(client, auth, stored):
    record_path = temp_files.GENERATED_DIR / f"{stored}.json"
    record = json.loads(record_path.read_text())
    record["expires_at"] = int(time.time()) - 1
    record_path.write_text(json.dumps(record))
    response = download(client, auth, stored)
    assert response.status_code == 410 and "expired" in response.json()["detail"]
    assert stored_files() == []
    assert download(client, auth, stored).status_code == 404


def test_cleanup_removes_expired_and_orphaned_files_only():
    keep = temp_files.store(b"%PDF-keep", course_id=1, user_id=10, filename="a", ttl_seconds=3600)
    old = temp_files.store(b"%PDF-old", course_id=1, user_id=10, filename="b", ttl_seconds=3600)
    orphan = temp_files.GENERATED_DIR / ("f" * 32 + ".pdf")
    orphan.write_bytes(b"%PDF-orphan")
    now = time.time()
    record_path = temp_files.GENERATED_DIR / f"{old.file_id}.json"
    record = json.loads(record_path.read_text())
    record["expires_at"] = int(now) - 5
    record_path.write_text(json.dumps(record))

    removed = temp_files.cleanup_expired(now=now + 120)  # past the orphan grace period
    assert removed == 2
    assert stored_files() == sorted([f"{keep.file_id}.json", f"{keep.file_id}.pdf"])


def test_store_uses_unique_ids_and_safe_names():
    a = temp_files.store(b"%PDF-1", course_id=1, user_id=10, filename="../../etc/passwd",
                         ttl_seconds=60)
    b = temp_files.store(b"%PDF-1", course_id=1, user_id=10, filename="Cell <Biology> 101",
                         ttl_seconds=60)
    assert a.file_id != b.file_id
    assert "/" not in a.filename and ".." not in a.filename and a.filename.endswith(".pdf")
    assert b.filename == "Cell_Biology_101.pdf"


def test_ttl_is_configurable(client, auth, sheet_llm, configure):
    configure(STUDY_SHEET_TTL_MINUTES="15")
    material(5)
    body = chat(client, auth)
    assert body["download"]["ttl_minutes"] == 15 and "15 minutes" in body["answer"]
    assert body["download"]["expires_at"] - time.time() <= 15 * 60 + 5


def test_app_startup_sweeps_expired_files():
    from fastapi.testclient import TestClient

    from app.main import app as fastapi_app

    old = temp_files.store(b"%PDF-old", course_id=1, user_id=10, filename="b", ttl_seconds=3600)
    record_path = temp_files.GENERATED_DIR / f"{old.file_id}.json"
    record = json.loads(record_path.read_text())
    record["expires_at"] = int(time.time()) - 5
    record_path.write_text(json.dumps(record))
    with TestClient(fastapi_app):
        deadline = time.time() + 5
        while stored_files() and time.time() < deadline:
            time.sleep(0.05)
    assert stored_files() == []
