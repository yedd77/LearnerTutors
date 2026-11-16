"""Access-control and lifecycle tests for the Chroma-backed vector store.

search_course() is the single choke point that enforces course isolation, the exclusion
of legacy personal uploads and the per-document retrieval switch, so it is tested hard.
"""
import pytest

import app.services.vector_store as vs
from helpers import add_legacy_personal_chunk, index_text

QUERY = "photosynthesis"


def doc_ids(rows) -> set[str]:
    return {row["metadata"]["document_id"] for row in rows}


@pytest.fixture
def corpus():
    """Two courses of course documents, plus legacy personal uploads left on disk."""
    index_text("c1-course", "photosynthesis converts light energy", course_id=1, owner_id=99)
    index_text("c2-course", "photosynthesis in course two", course_id=2, owner_id=99)
    add_legacy_personal_chunk("c1-u10", "photosynthesis personal notes ten", course_id=1, owner_id=10)
    add_legacy_personal_chunk("c2-u10", "photosynthesis user ten course two", course_id=2, owner_id=10)


# --------------------------------------------------------------------------- isolation
def test_user_sees_only_their_own_courses_documents(corpus):
    rows = vs.search_course(QUERY, course_id=1, include_course_documents=True, top_k=20)
    assert doc_ids(rows) == {"c1-course"}


def test_other_courses_are_never_returned(corpus):
    for course_id in (1, 2):
        rows = vs.search_course(QUERY, course_id, True, 20)
        assert rows and all(r["metadata"]["course_id"] == course_id for r in rows)
    assert doc_ids(vs.search_course(QUERY, 2, True, 20)) == {"c2-course"}


def test_legacy_personal_uploads_are_never_returned_even_to_their_owner(corpus):
    for course_id in (1, 2):
        rows = vs.search_course(QUERY, course_id, True, 20)
        assert all(r["metadata"]["visibility"] == "course" for r in rows)
        assert not {"c1-u10", "c2-u10"} & doc_ids(rows)


def test_course_documents_excluded_returns_nothing_without_querying(corpus, fake_embeddings):
    """ERR-002 regression: this used to raise ValueError (HTTP 422 from /chat)."""
    calls_before = len(fake_embeddings)
    assert vs.search_course(QUERY, 1, False, 20) == []
    assert len(fake_embeddings) == calls_before  # no embedding or Chroma query


def test_unknown_course_gets_nothing(corpus):
    assert vs.search_course(QUERY, 999, True, 20) == []


def test_indexed_chunks_are_always_course_visibility():
    index_text("doc", "photosynthesis text", owner_id=10)
    [row] = vs.search_course(QUERY, 1, True, 5)
    assert row["metadata"]["visibility"] == "course"
    assert row["metadata"]["owner_id"] == 10  # uploader is still recorded


def test_owner_id_does_not_affect_retrieval():
    index_text("a", "photosynthesis from lecturer a", owner_id=1)
    index_text("b", "photosynthesis from lecturer b", owner_id=2)
    assert doc_ids(vs.search_course(QUERY, 1, True, 20)) == {"a", "b"}


# --------------------------------------------------------------------------- retrieval switch
def test_disabled_documents_are_excluded_and_can_be_re_enabled(corpus):
    assert vs.set_document_retrieval("c1-course", 1, False) == 1
    assert vs.search_course(QUERY, 1, True, 20) == []
    assert vs.set_document_retrieval("c1-course", 1, True) == 1
    assert doc_ids(vs.search_course(QUERY, 1, True, 20)) == {"c1-course"}


def test_toggle_is_scoped_to_the_course(corpus):
    assert vs.set_document_retrieval("c1-course", 2, False) == 0  # wrong course: no-op
    assert "c1-course" in doc_ids(vs.search_course(QUERY, 1, True, 20))


def test_document_indexed_as_disabled_is_not_retrievable():
    index_text("hidden", "photosynthesis hidden", retrieval_enabled=False)
    assert vs.search_course(QUERY, 1, True, 20) == []
    assert vs.get_document_retrieval_status("hidden") == {
        "total_chunks": 1, "enabled_chunks": 0, "disabled_chunks": 1,
    }


def test_reindex_keeps_existing_retrieval_flag_when_not_specified():
    index_text("doc", "photosynthesis v1")
    vs.set_document_retrieval("doc", 1, False)
    index_text("doc", "photosynthesis v2")  # retrieval_enabled=None -> inherit
    assert vs.get_document_retrieval_status("doc")["enabled_chunks"] == 0
    index_text("doc", "photosynthesis v3", retrieval_enabled=True)  # explicit wins
    assert vs.get_document_retrieval_status("doc")["enabled_chunks"] == 1


def test_legacy_chunks_without_flag_are_backfilled_as_enabled():
    collection = vs.get_collection()
    collection.add(
        ids=["legacy-1"], documents=["photosynthesis legacy"],
        embeddings=[__import__("conftest").fake_vector("photosynthesis legacy")],
        metadatas=[{"document_id": "legacy", "course_id": 1, "owner_id": 5,
                    "visibility": "course", "page": 1, "filename": "legacy.txt"}],
    )
    # Before migration the retrieval_enabled filter hides it...
    assert vs.search_course(QUERY, 1, True, 20) == []
    vs._migrate_retrieval_metadata(collection)
    # ...afterwards it is searchable.
    assert doc_ids(vs.search_course(QUERY, 1, True, 20)) == {"legacy"}


# --------------------------------------------------------------------------- delete / conflict / reindex
def test_delete_is_scoped_to_course_and_document(corpus):
    assert vs.delete_document("c1-course", course_id=2) == 0  # wrong course
    assert "c1-course" in doc_ids(vs.search_course(QUERY, 1, True, 20))
    assert vs.delete_document("c1-course", course_id=1) == 1
    assert vs.search_course(QUERY, 1, True, 20) == []
    assert "c2-course" in doc_ids(vs.search_course(QUERY, 2, True, 20))


def test_reindexing_a_document_id_into_another_course_is_rejected():
    index_text("shared-id", "photosynthesis original", course_id=1)
    with pytest.raises(vs.DocumentCourseConflictError):
        index_text("shared-id", "photosynthesis hijack", course_id=2)
    assert vs.search_course(QUERY, 2, True, 20) == []
    rows = vs.search_course(QUERY, 1, True, 20)
    assert [r["text"] for r in rows] == ["photosynthesis original"]


def test_reindex_replaces_old_chunks():
    vs.index_document("doc", "doc.txt", 1, 10,
                      [{"page": 1, "text": f"photosynthesis part {i}"} for i in range(3)])
    assert vs.get_document_retrieval_status("doc")["total_chunks"] == 3
    count = vs.index_document("doc", "doc.txt", 1, 10, [{"page": 1, "text": "photosynthesis only"}])
    assert count == 1
    assert vs.get_document_retrieval_status("doc")["total_chunks"] == 1


def test_failed_indexing_leaves_no_partial_chunks(monkeypatch):
    calls = {"n": 0}

    def flaky_embed(texts, task_type="RETRIEVAL_DOCUMENT"):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("embedding provider down")
        return [__import__("conftest").fake_vector(t) for t in texts]

    monkeypatch.setattr(vs, "embed_texts", flaky_embed)
    chunks = [{"page": 1, "text": f"photosynthesis chunk {i}"} for i in range(150)]  # two batches
    with pytest.raises(RuntimeError, match="provider down"):
        vs.index_document("big", "big.txt", 1, 10, chunks)
    assert vs.get_document_retrieval_status("big")["total_chunks"] == 0


# --------------------------------------------------------------------------- search behaviour
def test_nearest_chunk_ranks_first():
    index_text("bio", "mitosis is cell division", course_id=1)
    index_text("chem", "covalent bonds share electrons", course_id=1)
    rows = vs.search_course("mitosis cell division", 1, True, 5)
    assert rows[0]["metadata"]["document_id"] == "bio"
    assert rows[0]["distance"] < rows[1]["distance"]


def test_top_k_is_clamped_to_between_one_and_twenty():
    vs.index_document("many", "many.txt", 1, 10,
                      [{"page": 1, "text": f"photosynthesis item {i}"} for i in range(25)])
    assert len(vs.search_course(QUERY, 1, True, 100)) == 20
    assert len(vs.search_course(QUERY, 1, True, 0)) == 1
    assert len(vs.search_course(QUERY, 1, True, 3)) == 3


def test_result_rows_expose_text_metadata_and_distance(corpus):
    row = vs.search_course(QUERY, 1, True, 1)[0]
    assert set(row) == {"chunk_id", "text", "metadata", "distance"}
    assert {"document_id", "course_id", "owner_id", "visibility", "page", "filename",
            "retrieval_enabled"} <= set(row["metadata"])


def test_missing_page_is_stored_as_minus_one():
    index_text("nopage", "photosynthesis undated", page=None)
    assert vs.search_course(QUERY, 1, True, 1)[0]["metadata"]["page"] == -1


def test_embedding_task_types(fake_embeddings):
    index_text("doc", "photosynthesis text")
    vs.search_course(QUERY, 1, True, 5)
    assert [c["task_type"] for c in fake_embeddings] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    assert fake_embeddings[-1]["texts"] == [QUERY]


def test_retrieval_report_lists_documents_with_counts(corpus):
    vs.set_document_retrieval("c2-course", 2, False)
    report = {d["document_id"]: d for d in vs.get_retrieval_report()}
    assert set(report) == {"c1-course", "c2-course", "c1-u10", "c2-u10"}
    assert report["c2-course"]["enabled_chunks"] == 0 and report["c2-course"]["total_chunks"] == 1
    assert report["c1-course"]["course_id"] == 1


def test_retrieval_report_surfaces_legacy_personal_uploads_for_cleanup(corpus):
    legacy = {d["document_id"] for d in vs.get_retrieval_report() if d["visibility"] == "personal"}
    assert legacy == {"c1-u10", "c2-u10"}


def test_legacy_personal_uploads_can_still_be_deleted(corpus):
    assert vs.delete_document("c1-u10", course_id=1) == 1
    assert "c1-u10" not in {d["document_id"] for d in vs.get_retrieval_report()}


# --------------------------------------------------------------------------- embedding-space isolation
def test_collection_name_changes_with_embedding_identity(monkeypatch):
    base = vs.get_collection().name
    try:
        with monkeypatch.context() as m:
            m.setenv("EMBEDDING_PROVIDER", "gemini")
            m.setenv("EMBEDDING_MODEL", "gemini-embedding-001")
            __import__("conftest").reset_settings_cache()
            vs.reset_vector_store_cache()
            gemini_name = vs.get_collection().name
            m.setenv("GEMINI_EMBED_DIMENSIONS", "256")
            __import__("conftest").reset_settings_cache()
            vs.reset_vector_store_cache()
            other_dims_name = vs.get_collection().name
        assert len({base, gemini_name, other_dims_name}) == 3
        assert all(len(n) <= 63 for n in (base, gemini_name, other_dims_name))
    finally:
        # Restore the shared collection for the remaining tests.
        __import__("conftest").reset_settings_cache()
        vs.reset_vector_store_cache()
    assert vs.get_collection().name == base
