"""Retrieval pipeline: sentence chunking, headings, hybrid search, neighbours, rerank, sub-queries.

All offline: embeddings are the conftest bag-of-words fakes; LLM calls are stubbed.
"""
from io import BytesIO

import fitz
import pytest
import requests
from docx import Document
from pptx import Presentation
from pptx.util import Inches

import app.services.embedding_provider as embeddings
import app.services.llm_provider as llm_provider
import app.services.vector_store as vs
from app.config import get_settings
from app.rag_graph import nodes
from app.services import keyword_index, retrieval
from app.services import study_sheet as sheets
from app.services.chunker import chunk_sections
from app.services.document_parser import extract_sections
from helpers import chat_body, index_text

CHAT = "/api/v1/chat"


def ask(client, auth, question, **extra):
    response = client.post(CHAT, headers=auth, json=chat_body(question, **extra))
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def aux(monkeypatch):
    """Stub for the structured aux calls (classify/rewrite/rerank)."""
    class Aux:
        calls: list[tuple[str, list]] = []
        rewrite = None          # nodes.SearchQuery to return
        ranking: list[int] = []
        error: Exception | None = None

    stub = Aux()
    stub.calls = []

    async def fake_structured(messages, schema, provider=None, model=None, *, timeout=None,
                              fallback=None):
        stub.calls.append((schema.__name__, messages))
        if stub.error is not None:
            raise stub.error
        if schema is nodes.Ranking:
            return schema(ranking=stub.ranking)
        if schema is nodes.SearchQuery:
            return stub.rewrite
        return schema(intent="course_question")

    monkeypatch.setattr(llm_provider, "generate_structured", fake_structured)
    return stub


def index_chunks(document_id, texts, *, course_id=1, filename=None, headings=None,
                 retrieval_enabled=None, start=0):
    chunks = [{"page": 1, "text": text, "chunk_index": start + i,
               "heading": (headings[i] if headings else None)} for i, text in enumerate(texts)]
    return vs.index_document(document_id, filename or f"{document_id}.txt", course_id, 10, chunks,
                             retrieval_enabled=retrieval_enabled)


# --------------------------------------------------------------------------- chunking
def test_chunks_end_at_sentence_boundaries(configure):
    configure(CHUNK_SIZE="200", CHUNK_OVERLAP="60")
    sentences = [f"Sentence number {i} explains one pharmacology idea clearly." for i in range(12)]
    chunks = chunk_sections([{"page": 1, "text": " ".join(sentences)}])
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 200
        assert chunk["text"].startswith("Sentence number") and chunk["text"].endswith("clearly.")
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_overlap_repeats_whole_sentences(configure):
    configure(CHUNK_SIZE="200", CHUNK_OVERLAP="60")
    sentences = [f"Fact {i} is stated here in full." for i in range(20)]
    chunks = chunk_sections([{"page": 1, "text": " ".join(sentences)}])
    for earlier, later in zip(chunks, chunks[1:]):
        first_sentence = later["text"].split(". ")[0] + "."
        assert first_sentence in earlier["text"]          # carried over, not cut mid-sentence


def test_overlong_sentence_is_split_at_spaces(configure):
    configure(CHUNK_SIZE="200", CHUNK_OVERLAP="0")
    chunks = chunk_sections([{"page": 1, "text": "word " * 120}])
    assert all(len(c["text"]) <= 200 and not c["text"].endswith("wor") for c in chunks)


def test_headings_and_document_wide_indexes():
    chunks = chunk_sections([
        {"page": 1, "text": "Intro text.", "heading": "Introduction"},
        {"page": 2, "text": "Method text."},
    ])
    assert [(c["heading"], c["chunk_index"]) for c in chunks] == [("Introduction", 0), (None, 1)]


def test_line_breaks_are_kept_for_lists():
    chunks = chunk_sections([{"page": 1, "text": "Goals:\n- first item\n- second item"}])
    assert chunks[0]["text"] == "Goals:\n- first item\n- second item"


# --------------------------------------------------------------------------- headings in parsing
def test_pdf_larger_font_lines_become_headings():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Pharmacogenomics", fontsize=20)
    page.insert_text((72, 110), "Genes change how patients respond to drugs.", fontsize=11)
    page.insert_text((72, 130), "Testing guides the choice of medicine.", fontsize=11)
    page = doc.new_page()
    page.insert_text((72, 72), "More body text on the next page.", fontsize=11)
    data = doc.tobytes()
    sections = extract_sections("a.pdf", data)
    assert sections[0]["heading"] == "Pharmacogenomics" and sections[0]["page"] == 1
    assert "Genes change" in sections[0]["text"]
    assert sections[1] == {"page": 2, "text": "More body text on the next page.",
                           "heading": "Pharmacogenomics"}        # carried over


def test_docx_heading_styles_split_sections():
    doc = Document()
    doc.add_heading("Chapter 1: Formulary", level=1)
    doc.add_paragraph("A formulary lists approved medicines.")
    doc.add_heading("Chapter 2: Safety", level=1)
    doc.add_paragraph("Safety goals reduce errors.")
    buffer = BytesIO()
    doc.save(buffer)
    sections = extract_sections("a.docx", buffer.getvalue())
    assert [s["heading"] for s in sections] == ["Chapter 1: Formulary", "Chapter 2: Safety"]
    assert "approved medicines" in sections[0]["text"]


def test_pptx_slide_title_is_the_heading():
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])   # title + content
    slide.shapes.title.text = "Rational drug use"
    slide.placeholders[1].text = "Right drug, right dose."
    buffer = BytesIO()
    presentation.save(buffer)
    [section] = extract_sections("a.pptx", buffer.getvalue())
    assert section["heading"] == "Rational drug use" and "Right drug" in section["text"]


def test_markdown_sections():
    sections = extract_sections("n.md", b"Preface.\n\n# One\nAlpha.\n## Two\nBeta.")
    assert [(s.get("heading"), s["text"]) for s in sections] == [
        (None, "Preface."), ("One", "# One\nAlpha."), ("Two", "## Two\nBeta.")]


# --------------------------------------------------------------------------- indexing metadata
def test_heading_and_index_are_stored_and_embedded_in_context(fake_embeddings):
    index_chunks("doc", ["Activated mainly by the liver enzyme."], headings=["Clopidogrel"],
                 filename="week_3-pharmacogenomics.pdf")
    stored = vs.get_collection().get(where={"document_id": "doc"}, include=["metadatas", "documents"])
    assert stored["metadatas"][0]["heading"] == "Clopidogrel"
    assert stored["metadatas"][0]["chunk_index"] == 0
    assert stored["documents"][0] == "Activated mainly by the liver enzyme."     # stored plain
    embedded = fake_embeddings[-1]["texts"][0]
    assert embedded.startswith("Document: week 3 pharmacogenomics\nSection: Clopidogrel\n\n")


# --------------------------------------------------------------------------- keyword index
def test_tokenizer_keeps_codes_and_drops_stopwords():
    assert keyword_index.tokenize("What is the role of CYP2C19?") == ["role", "cyp2c19"]


def test_bm25_prefers_the_rare_exact_term():
    index = keyword_index.KeywordIndex.build(
        ["a", "b", "c"],
        ["drug therapy overview", "CYP2C19 metabolises clopidogrel", "drug drug drug therapy"])
    assert index.search("cyp2c19 drug", 3)[0][0] == "b"
    assert index.search("unrelated", 3) == []


# --------------------------------------------------------------------------- hybrid search
def test_keyword_hit_outside_the_dense_pool_is_found(monkeypatch, configure):
    configure(RETRIEVAL_CANDIDATES="1")
    index_chunks("rare", ["Zymogen activation occurs in the pancreas."])
    index_chunks("common", ["Photosynthesis light energy plants."])
    real_query = vs.get_collection().query

    def dense_only_common(*args, **kwargs):          # dense search never sees "rare"
        kwargs["where"] = {"$and": [kwargs["where"], {"document_id": "common"}]}
        return real_query(*args, **kwargs)

    monkeypatch.setattr(vs.get_collection(), "query", dense_only_common)
    rows = vs.hybrid_search(["zymogen activation"], 1, True, top_k=1)
    rare = next(row for row in rows if row["metadata"]["document_id"] == "rare")
    assert rare["distance"] is not None              # threshold still applies to keyword hits


def test_keyword_ranking_never_exposes_filtered_chunks(monkeypatch):
    index_chunks("mine", ["Insulin regulates glucose."], course_id=1)
    index_chunks("other", ["Insulin secret notes."], course_id=2)
    index_chunks("off", ["Insulin disabled document."], course_id=1, retrieval_enabled=False)
    foreign = vs.get_collection().get(where={"document_id": {"$in": ["other", "off"]}})["ids"]
    monkeypatch.setattr(keyword_index.KeywordIndex, "search",
                        lambda self, query, limit: [(chunk_id, 9.9) for chunk_id in foreign])
    rows = vs.hybrid_search(["insulin"], 1, True, top_k=5)
    assert {row["metadata"]["document_id"] for row in rows} == {"mine"}


def test_deleted_document_is_not_returned_from_a_cached_keyword_index():
    index_chunks("gone", ["Warfarin interacts with vitamin K."])
    assert vs.hybrid_search(["warfarin"], 1, True, top_k=5)
    vs.delete_document("gone", 1)
    assert vs.hybrid_search(["warfarin"], 1, True, top_k=5) == []


def test_duplicate_chunks_are_dropped():
    index_chunks("copy1", ["The essential medicines list guides procurement decisions."])
    index_chunks("copy2", ["The essential medicines list guides procurement decisions."])
    rows = vs.hybrid_search(["essential medicines list"], 1, True, top_k=5)
    assert len(rows) == 1


def test_each_sub_query_contributes_results():
    index_chunks("cyp", ["CYP2C19 enzyme genotype variants."])
    index_chunks("plt", ["Platelet aggregation inhibitor therapy."])
    for n in range(6):
        index_chunks(f"noise{n}", [f"Unrelated filler topic number {n} about hospital logistics."])
    rows = vs.hybrid_search(["CYP2C19 genotype", "platelet inhibitor therapy"], 1, True, top_k=1)
    found = {row["metadata"]["document_id"] for row in rows[:2]}
    assert found == {"cyp", "plt"}


def test_excluded_course_documents_mean_no_search(fake_embeddings):
    index_chunks("doc", ["Anything at all."])
    before = len(fake_embeddings)
    assert vs.hybrid_search(["anything"], 1, False, top_k=5) == []
    assert len(fake_embeddings) == before


# --------------------------------------------------------------------------- neighbours
def test_neighbours_are_merged_without_repeating_the_overlap():
    index_chunks("doc", ["First part. Shared sentence.", "Shared sentence. Second part.",
                         "Third part.", "Unrelated far chunk."], start=0)
    hit = vs.hybrid_search(["second part"], 1, True, top_k=1)[0]
    assert hit["metadata"]["chunk_index"] == 1
    [passage] = vs.expand_with_neighbours([hit], 1, neighbour_hits=1)
    assert passage["text"] == "First part. Shared sentence. Second part.\nThird part."
    assert passage["chunk_count"] == 3 and passage["chunk_id"] == hit["chunk_id"]


def test_adjacent_hits_merge_into_one_passage():
    index_chunks("doc", ["Alpha topic one.", "Alpha topic two."])
    hits = vs.hybrid_search(["alpha topic"], 1, True, top_k=2)
    passages = vs.expand_with_neighbours(hits, 1, neighbour_hits=0)
    assert len(passages) == 1 and passages[0]["chunk_count"] == 2
    assert retrieval.chunk_count(passages) == 2


def test_neighbours_of_a_disabled_document_are_not_fetched():
    index_chunks("doc", ["Before text.", "Hit text beta.", "After text."])
    hit = vs.hybrid_search(["beta"], 1, True, top_k=1)[0]
    vs.set_document_retrieval("doc", 1, False)
    [passage] = vs.expand_with_neighbours([hit], 1, neighbour_hits=1)
    # The access filter excludes every chunk of the disabled document: nothing is added.
    assert passage == hit


def test_legacy_chunks_without_index_are_left_alone():
    from conftest import fake_vector

    vs.get_collection().add(              # indexed before chunk_index existed
        ids=["old_0"], documents=["Legacy chunk text."], embeddings=[fake_vector("Legacy chunk text.")],
        metadatas=[{"document_id": "old", "filename": "old.txt", "course_id": 1, "owner_id": 10,
                    "visibility": "course", "retrieval_enabled": True, "page": 1}])
    keyword_index.invalidate_course()
    hits = vs.hybrid_search(["legacy chunk"], 1, True, top_k=1)
    assert vs.expand_with_neighbours(hits, 1, neighbour_hits=3) == hits


def test_notes_sufficiency_counts_merged_chunks(configure):
    configure(NOTES_MIN_CHUNKS="3", NOTES_MIN_CONTEXT_CHARS="0")
    assert sheets.enough_material([{"text": "x", "chunk_count": 3}], get_settings())
    assert not sheets.enough_material([{"text": "x"}], get_settings())


# --------------------------------------------------------------------------- document-level
def test_summary_reads_the_dominant_section_in_order(client, auth, stub_generation):
    index_chunks("book", ["Ch1 opening.", "Ch1 middle.", "Ch1 end.", "Ch2 text."],
                 headings=["Chapter 1", "Chapter 1", "Chapter 1", "Chapter 2"], filename="book.pdf")
    index_chunks("other", ["Unrelated other document."])
    ask(client, auth, "/summarize Ch1 opening middle end")
    context = stub_generation.calls[0]["context"]
    assert context.index("Ch1 opening") < context.index("Ch1 middle") < context.index("Ch1 end")
    assert "Ch2 text" not in context and "Unrelated" not in context
    assert "section: Chapter 1" in context


def test_summary_falls_back_when_hits_are_spread_out():
    matches = [{"metadata": {"document_id": d, "chunk_index": 0}} for d in ("a", "b", "c")]
    assert retrieval.document_scope(matches) is None


# --------------------------------------------------------------------------- rerank
def test_rerank_reorders_and_cuts_to_top_k(client, auth, stub_generation, aux, configure):
    configure(RAG_RERANK="true", TOP_K="2")
    for name in ("alpha", "beta", "gamma"):
        index_chunks(name, [f"Photosynthesis note {name}."])
    aux.ranking = [3, 99, 3, 1]                      # out-of-range and repeats are ignored
    body = ask(client, auth, "What is photosynthesis note?")
    assert [name for name, _ in aux.calls] == ["Ranking"]
    candidates = aux.calls[0][1][1]["content"]
    third = candidates.split("[3] ")[1].split("\n")[1].strip()
    assert body["retrieved_contexts"][0] == third and body["retrieved_chunks"] == 2


def test_rerank_failure_keeps_the_fused_order(client, auth, stub_generation, aux, configure):
    configure(RAG_RERANK="true", TOP_K="1")
    index_chunks("a", ["Photosynthesis alpha."])
    index_chunks("b", ["Photosynthesis beta."])
    aux.error = RuntimeError("timeout")
    body = ask(client, auth, "What is photosynthesis?")
    assert body["answer"] == "Stubbed answer [S1]." and body["retrieved_chunks"] == 1


def test_no_rerank_when_candidates_fit(client, auth, stub_generation, aux, configure):
    configure(RAG_RERANK="true")
    index_chunks("a", ["Photosynthesis alpha."])
    ask(client, auth, "What is photosynthesis?")
    assert aux.calls == []


# --------------------------------------------------------------------------- sub-queries
@pytest.mark.parametrize("question, compound", [
    ("What is CYP2C19, and how does it affect clopidogrel?", True),
    ("What is X? How is Y measured?", True),
    ("How does photosynthesis convert light energy?", False),
    ("Compare aspirin and ibuprofen", False),
])
def test_is_compound(question, compound):
    assert nodes.is_compound(question) is compound


def test_compound_first_turn_is_split_into_sub_queries(client, auth, stub_generation, aux,
                                                       monkeypatch):
    seen = []
    real = vs.hybrid_search

    def spy(queries, *args):
        seen.append(list(queries))
        return real(queries, *args)

    monkeypatch.setattr(vs, "hybrid_search", spy)
    index_chunks("doc", ["CYP2C19 metabolises clopidogrel."])
    aux.rewrite = nodes.SearchQuery(query="CYP2C19 role",
                                    sub_queries=["clopidogrel effect", " ", "extra one", "dropped"])
    ask(client, auth, "What is CYP2C19, and how does it affect clopidogrel?")
    assert seen == [["CYP2C19 role", "clopidogrel effect", "extra one"]]
    # Generation still uses the student's own wording.
    assert stub_generation.calls[0]["route"].question.startswith("What is CYP2C19")


# --------------------------------------------------------------------------- nomic prefixes
def test_ollama_nomic_texts_get_task_prefixes(monkeypatch, configure):
    configure(EMBEDDING_PROVIDER="ollama", EMBEDDING_MODEL="nomic-embed-text")
    sent = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embeddings": [[0.1, 0.2]] * len(sent[-1])}

    def fake_post(url, json=None, timeout=None, **kwargs):
        sent.append(json["input"])
        return Response()

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    embeddings.embed_texts(["doc text"], task_type="RETRIEVAL_DOCUMENT")
    embeddings.embed_texts(["question?"], task_type="RETRIEVAL_QUERY")
    assert sent == [["search_document: doc text"], ["search_query: question?"]]


def test_other_ollama_models_are_sent_unchanged(monkeypatch, configure):
    configure(EMBEDDING_PROVIDER="ollama", EMBEDDING_MODEL="mxbai-embed-large")
    sent = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embeddings": [[0.1]]}

    monkeypatch.setattr(embeddings.requests, "post",
                        lambda url, json=None, timeout=None, **k: sent.append(json["input"]) or Response())
    embeddings.embed_texts(["plain"], task_type="RETRIEVAL_QUERY")
    assert sent == [["plain"]]


def test_nomic_prefix_is_part_of_the_embedding_identity():
    base = get_settings()
    nomic = base.model_copy(update={"embedding_provider": "ollama", "embedding_model": "nomic-embed-text"})
    other = base.model_copy(update={"embedding_provider": "ollama", "embedding_model": "mxbai-embed-large"})
    assert nomic.embedding_identity == "ollama:nomic-embed-text:default+prefixed"
    assert other.embedding_identity == "ollama:mxbai-embed-large:default"


# --------------------------------------------------------------------------- admin status
@pytest.mark.parametrize("env, status, threshold", [
    ({"RETRIEVAL_MAX_DISTANCE": "0"}, "off", 0.0),
    ({"RETRIEVAL_MAX_DISTANCE": "0.6"}, "override", 0.6),
])
def test_relevance_filter_status_is_reported(client, auth, configure, env, status, threshold):
    configure(**env)
    body = client.get("/api/v1/admin/config", headers=auth).json()
    assert (body["relevance_filter"], body["relevance_threshold"]) == (status, threshold)


def test_calibrated_and_uncalibrated_models():
    base = get_settings().model_copy(update={"retrieval_max_distance": None})
    calibrated = base.model_copy(update={"embedding_provider": "openai",
                                         "embedding_model": "text-embedding-3-small"})
    assert calibrated.relevance_filter_status == "calibrated"
    assert base.model_copy(update={"embedding_provider": "gemini"}).relevance_filter_status == "off"


def test_calibration_suggestion():
    from eval.calibrate_threshold import suggest

    assert suggest([0.2, 0.5], [0.8, 0.9]) == (0.65, "Clean separation.")
    threshold, note = suggest([0.2, 0.7], [0.6, 0.9])
    assert threshold == 0.7 and note.startswith("Overlap")
