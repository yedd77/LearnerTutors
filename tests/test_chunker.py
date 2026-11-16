from app.services.chunker import chunk_sections

# Digits only, so .strip() never changes chunk boundaries.
TEXT = "0123456789" * 100  # 1000 chars


def test_short_text_is_one_chunk_with_page_preserved(configure):
    configure(CHUNK_SIZE="900", CHUNK_OVERLAP="150")
    chunks = chunk_sections([{"page": 7, "text": "Short text."}])
    assert chunks == [{"page": 7, "text": "Short text."}]


def test_windows_overlap_by_configured_amount(configure):
    configure(CHUNK_SIZE="300", CHUNK_OVERLAP="50")
    chunks = chunk_sections([{"page": 1, "text": TEXT}])
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 300 for c in chunks)
    for earlier, later in zip(chunks, chunks[1:]):
        assert later["text"][:50] == earlier["text"][-50:]


def test_chunks_cover_the_whole_text(configure):
    configure(CHUNK_SIZE="300", CHUNK_OVERLAP="50")
    chunks = chunk_sections([{"page": 1, "text": TEXT}])
    assert chunks[0]["text"] == TEXT[:300]
    assert chunks[-1]["text"].endswith(TEXT[-10:])


def test_each_section_keeps_its_own_page(configure):
    configure(CHUNK_SIZE="300", CHUNK_OVERLAP="50")
    chunks = chunk_sections([{"page": 1, "text": TEXT}, {"page": 2, "text": "second page"}])
    assert {c["page"] for c in chunks[:-1]} == {1}
    assert chunks[-1] == {"page": 2, "text": "second page"}


def test_missing_page_becomes_none():
    assert chunk_sections([{"text": "no page here"}])[0]["page"] is None


def test_whitespace_is_normalised():
    chunks = chunk_sections([{"page": 1, "text": "a   b\t\tc\n\n\n\nd"}])
    assert chunks[0]["text"] == "a b c\n\nd"


def test_blank_sections_produce_no_chunks():
    assert chunk_sections([{"page": 1, "text": "   \n\n  "}]) == []


def test_tiny_chunk_size_is_raised_to_the_minimum(configure):
    configure(CHUNK_SIZE="10", CHUNK_OVERLAP="0")
    chunks = chunk_sections([{"page": 1, "text": TEXT}])
    assert max(len(c["text"]) for c in chunks) == 200


def test_oversized_overlap_is_capped_so_windows_advance(configure):
    configure(CHUNK_SIZE="300", CHUNK_OVERLAP="299")
    chunks = chunk_sections([{"page": 1, "text": TEXT}])
    # Overlap is capped at size // 2, so this terminates with a bounded chunk count.
    assert 1 < len(chunks) < 20
