"""Images in course documents: extraction, anchoring, storage, [IMGn] markers and serving.

All offline: embeddings are the conftest fakes and generation is stubbed.
"""
from io import BytesIO

import fitz
import pytest
from docx import Document
from docx.shared import Inches as DocxInches
from PIL import Image
from pptx import Presentation
from pptx.util import Inches

from app.prompt_builder import _core_prompt
from app.services import document_images as images_service
from app.services.chunker import chunk_sections
from app.services.document_parser import extract_sections
from helpers import chat_body

CHAT = "/api/v1/chat"


def png(width=200, height=150, colour=(200, 30, 30)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, "PNG")
    return buffer.getvalue()


def pdf_with_figure(caption="Figure 1: The Krebs cycle") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The Krebs cycle releases energy from acetyl-CoA.", fontsize=11)
    page.insert_image(fitz.Rect(72, 100, 272, 250), stream=png())
    page.insert_text((72, 280), caption, fontsize=11)
    page.insert_text((72, 320), "Oxaloacetate is regenerated at the end.", fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def upload(client, auth, filename, content, document_id="lec", course_id=1):
    response = client.post(
        "/api/v1/documents/index", headers=auth,
        files={"file": (filename, content, "application/octet-stream")},
        data={"document_id": document_id, "course_id": str(course_id), "owner_id": "10",
              "visibility": "course"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def ask(client, auth, question, **extra):
    response = client.post(CHAT, headers=auth, json=chat_body(question, **extra))
    assert response.status_code == 200, response.text
    return response.json()


def occurrence(ref, offset, data=None, digest=None, caption=None, page=1):
    return {"ref": ref, "hash": digest or f"h{ref}", "data": data or png(), "offset": offset,
            "caption": caption, "page": page}


# --------------------------------------------------------------------------- extraction
def test_pdf_image_position_and_caption():
    [section] = extract_sections("a.pdf", pdf_with_figure())
    [image] = section["images"]
    before = section["text"][:image["offset"]]
    assert before.rstrip().endswith("acetyl-CoA.") and "Oxaloacetate" not in before
    assert image["caption"] == "Figure 1: The Krebs cycle" and image["page"] == 1


def test_pdf_without_caption_line_has_no_caption():
    [section] = extract_sections("a.pdf", pdf_with_figure(caption="Plain text after the image."))
    assert section["images"][0]["caption"] is None


def test_docx_inline_picture_with_alt_text():
    doc = Document()
    doc.add_paragraph("Insulin lowers blood glucose.")
    doc.add_picture(BytesIO(png()), width=DocxInches(2))
    doc.inline_shapes[0]._inline.docPr.set("descr", "Insulin signalling pathway")
    doc.add_paragraph("Glucagon raises it.")
    buffer = BytesIO()
    doc.save(buffer)
    [section] = extract_sections("a.docx", buffer.getvalue())
    [image] = section["images"]
    assert section["text"][:image["offset"]].endswith("Insulin lowers blood glucose.")
    assert image["caption"] == "Insulin signalling pathway"


def test_pptx_picture_uses_slide_title_as_caption():
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])     # title only
    slide.shapes.title.text = "Drug absorption"
    slide.shapes.add_picture(BytesIO(png()), Inches(1), Inches(2))
    buffer = BytesIO()
    presentation.save(buffer)
    [section] = extract_sections("a.pptx", buffer.getvalue())
    assert section["images"][0]["caption"] == "Drug absorption"


# --------------------------------------------------------------------------- anchoring
def test_images_anchor_between_chunks(configure):
    configure(CHUNK_SIZE="200", CHUNK_OVERLAP="0")
    x = "Chunk X sentence about enzymes. " * 6          # ~200 chars: fills one chunk
    y = "Chunk Y sentence about substrates. " * 2
    z = "Chunk Z closing sentence."
    text = x + y + z
    section = {"page": 1, "text": text, "images": [
        occurrence(1, len(x)),                          # after X, before Y
        occurrence(2, len(text)),                       # after Z
    ]}
    chunks = chunk_sections([section])
    owner = {ref: c["chunk_index"] for c in chunks for ref in c.get("image_refs", [])}
    assert "Chunk X" in chunks[owner[1]]["text"] and "Chunk Y" not in chunks[owner[1]]["text"]
    assert chunks[owner[2]]["text"].endswith("Chunk Z closing sentence.")
    assert all("image" not in c["text"].lower() for c in chunks)       # images never enter the text


def test_image_before_any_text_and_image_only_sections():
    chunks = chunk_sections([
        {"page": 1, "text": "First slide text.", "images": [occurrence(1, 0)]},
        {"page": 2, "text": "", "images": [occurrence(2, 0)]},           # picture-only slide
    ])
    assert chunks[0]["image_refs_before"] == [1]
    assert chunks[0]["image_refs"] == [2]


def test_sections_without_images_are_unchanged():
    assert chunk_sections([{"page": 1, "text": "Plain."}]) == [
        {"page": 1, "heading": None, "text": "Plain.", "chunk_index": 0}]


# --------------------------------------------------------------------------- storage
def stored(course_id, document_id):
    return images_service._manifest(course_id, document_id)


def test_store_filters_icons_logos_and_unreadable_files():
    sections = [{"page": 1, "text": "t", "images": [
        occurrence(1, 0, data=png(50, 50)),                                  # icon
        occurrence(2, 0, digest="logo"), occurrence(3, 0, digest="logo"),
        occurrence(4, 0, digest="logo"),                                     # repeated logo
        occurrence(5, 0, data=b"not an image"),                              # e.g. EMF
        occurrence(6, 0, data=png(3200, 800)),                               # kept, scaled down
    ]}]
    chunks = [{"chunk_index": 0, "image_refs": [1, 2, 3, 4, 5, 6]}]
    assert images_service.store_document_images(1, "doc", sections, chunks) == 1
    [entry] = stored(1, "doc")
    assert (entry["width"], entry["height"]) == (1600, 400) and entry["anchor"] == 0


def test_reindex_replaces_and_delete_removes_images():
    sections = [{"page": 1, "text": "t", "images": [occurrence(1, 0, digest="a")]}]
    images_service.store_document_images(1, "doc", sections, [{"chunk_index": 0, "image_refs": [1]}])
    first = stored(1, "doc")[0]["image_id"]
    sections = [{"page": 1, "text": "t", "images": [occurrence(1, 0, digest="b", data=png(colour=(0, 0, 255)))]}]
    images_service.store_document_images(1, "doc", sections, [{"chunk_index": 0, "image_refs": [1]}])
    [entry] = stored(1, "doc")
    assert entry["image_id"] != first
    assert not (images_service._folder(1, "doc") / f"{first}.png").exists()
    images_service.delete_document_images(1, "doc")
    assert stored(1, "doc") == []


def test_manifest_of_another_course_is_never_read():
    sections = [{"page": 1, "text": "t", "images": [occurrence(1, 0)]}]
    images_service.store_document_images(1, "doc", sections, [{"chunk_index": 0, "image_refs": [1]}])
    assert stored(2, "doc") == []


# --------------------------------------------------------------------------- attaching / markers
def test_attach_uses_the_anchor_window_and_offsets(monkeypatch):
    manifest = [{"image_id": "a" * 32, "anchor": 3, "caption": "before", "page": 1},
                {"image_id": "b" * 32, "anchor": 5, "caption": "inside", "page": 1},
                {"image_id": "c" * 32, "anchor": 9, "caption": None, "page": 4}]
    monkeypatch.setattr(images_service, "_manifest", lambda course, doc: manifest)
    passage = {"text": "0123456789ABCDEF", "metadata": {"document_id": "d", "filename": "d.pdf"},
               "chunk_range": (4, 6), "chunk_ends": {4: 5, 5: 10, 6: 16}}
    images = images_service.attach_images([passage], 1)
    assert [(i["marker"], i["caption"], i["offset"]) for i in images] == [
        ("IMG1", "before", 0), ("IMG2", "inside", 10)]


def test_markers_are_inserted_into_the_context_only():
    from app.services.retrieval import build_context_and_sources

    passage = {"chunk_id": "c", "text": "Before. After.", "metadata": {"document_id": "d", "page": 1}}
    context, _ = build_context_and_sources(
        [passage], [{"marker": "IMG1", "caption": "Krebs cycle", "passage": 0, "offset": 7}])
    assert "Before.\n[IMG1: Krebs cycle]\nAfter." in context
    assert passage["text"] == "Before. After."


def test_resolve_markers_keeps_only_supplied_markers_once():
    images = [{"marker": "IMG1"}, {"marker": "IMG2"}]
    answer, resolved = images_service.resolve_markers(
        "Intro.\n[IMG1: Krebs]\nMore.\n[IMG7]\n[IMG1]\nEnd.", images)
    # Dropped markers alone on their line take the line with them: no blank gap.
    assert answer == "Intro.\n[IMG1]\nMore.\nEnd."
    assert [(i["marker"], i["placed"]) for i in resolved] == [("IMG1", True), ("IMG2", False)]


def _three_images():
    return [{"marker": "IMG1"}, {"marker": "IMG2"}, {"marker": "IMG3"}]


def test_resolve_markers_keeps_only_the_first_two_in_answer_order():
    stats: dict = {}
    answer, resolved = images_service.resolve_markers(
        "Intro.\n\n[IMG1]\n\nLoading dose.\n\n[IMG3]\n\nMaintenance.\n\n[IMG2]\n\nEnd.",
        _three_images(), stats=stats)
    assert answer == "Intro.\n\n[IMG1]\n\nLoading dose.\n\n[IMG3]\n\nMaintenance.\n\nEnd."
    assert "[IMG2]" not in answer and "\n\n\n" not in answer
    assert {i["marker"]: i["placed"] for i in resolved} == {"IMG1": True, "IMG2": False, "IMG3": True}
    assert stats == {"over_limit": 1}


def test_repeated_marker_counts_once_toward_the_limit():
    stats: dict = {}
    answer, resolved = images_service.resolve_markers(
        "A.\n[IMG1]\nB.\n[IMG1]\nC.\n[IMG2]\nD.", _three_images(), stats=stats)
    assert answer == "A.\n[IMG1]\nB.\nC.\n[IMG2]\nD."
    assert [i["placed"] for i in resolved] == [True, True, False]
    assert stats == {"over_limit": 0}


def test_unknown_markers_do_not_use_a_slot():
    answer, resolved = images_service.resolve_markers(
        "A.\n[IMG9]\nB.\n[IMG2]\nC.\n[IMG3]\nD.", _three_images())
    assert answer == "A.\nB.\n[IMG2]\nC.\n[IMG3]\nD."
    assert [i["placed"] for i in resolved] == [False, True, True]


@pytest.mark.parametrize("text", [
    "No figures here.\n\nJust text.",
    "One.\n\n[IMG2]\n\nDone.",
    "One.\n\n[IMG1]\n\nTwo.\n\n[IMG3]\n\nDone.",
])
def test_answers_within_the_limit_are_unchanged(text):
    stats: dict = {}
    answer, _ = images_service.resolve_markers(text, _three_images(), stats=stats)
    assert answer == text
    assert stats == {"over_limit": 0}


def test_inline_dropped_marker_keeps_the_surrounding_text():
    answer, _ = images_service.resolve_markers(
        "See [IMG1] and [IMG2] and [IMG3] here.", _three_images())
    assert answer == "See [IMG1] and [IMG2] and  here."


def test_answer_format_prompt_explains_figure_markers():
    core = _core_prompt()
    assert "[IMG1]" in core
    assert "at most 2 figures" in core


# --------------------------------------------------------------------------- end to end
def test_answer_places_the_figure_and_moodle_can_fetch_it(client, auth, stub_generation):
    upload(client, auth, "krebs.pdf", pdf_with_figure())
    stub_generation.answer = "The cycle regenerates oxaloacetate [S1].\n[IMG1]\n[IMG5]"
    body = ask(client, auth, "How does the Krebs cycle release energy?")
    assert "[IMG1: Figure 1: The Krebs cycle]" in stub_generation.calls[0]["context"]
    assert body["answer"] == "The cycle regenerates oxaloacetate [S1].\n[IMG1]"
    [image] = body["images"]
    assert image["placed"] is True and image["page"] == 1 and image["filename"] == "krebs.pdf"
    assert image["path"] == f"/api/v1/document-images/{image['image_id']}?course_id=1&document_id=lec"
    assert all("[IMG" not in context for context in body["retrieved_contexts"])

    served = client.get(image["path"], headers=auth)
    assert served.status_code == 200 and served.headers["content-type"] == "image/png"
    assert served.headers["cache-control"] == "private, no-store"
    assert Image.open(BytesIO(served.content)).size == (200, 150)


def test_image_endpoint_enforces_api_key_course_and_visibility(client, auth, stub_generation):
    upload(client, auth, "krebs.pdf", pdf_with_figure())
    image_id = stored(1, "lec")[0]["image_id"]
    url = f"/api/v1/document-images/{image_id}"
    assert client.get(url, params={"course_id": 1, "document_id": "lec"}).status_code == 401
    assert client.get(url, headers=auth, params={"course_id": 2, "document_id": "lec"}).status_code == 404
    assert client.get(f"/api/v1/document-images/{'0' * 32}", headers=auth,
                      params={"course_id": 1, "document_id": "lec"}).status_code == 404
    assert client.get("/api/v1/document-images/..%2Fmanifest", headers=auth,
                      params={"course_id": 1, "document_id": "lec"}).status_code == 404

    client.patch("/api/v1/documents/lec/retrieval", headers=auth,
                 params={"course_id": 1}, json={"enabled": False})
    assert client.get(url, headers=auth, params={"course_id": 1, "document_id": "lec"}).status_code == 404
    client.patch("/api/v1/documents/lec/retrieval", headers=auth,
                 params={"course_id": 1}, json={"enabled": True})
    assert client.get(url, headers=auth, params={"course_id": 1, "document_id": "lec"}).status_code == 200

    assert client.delete("/api/v1/documents/lec", headers=auth, params={"course_id": 1}).status_code == 200
    assert client.get(url, headers=auth, params={"course_id": 1, "document_id": "lec"}).status_code == 404
    assert stored(1, "lec") == []


def test_other_course_never_sees_the_figure(client, auth, stub_generation):
    upload(client, auth, "krebs.pdf", pdf_with_figure(), course_id=1)
    body = ask(client, auth, "How does the Krebs cycle release energy?", course_id=2)
    assert "images" not in body


def test_answers_without_figures_keep_the_old_response_shape(client, auth, stub_generation):
    upload(client, auth, "notes.txt", b"Mitochondria produce ATP.")
    body = ask(client, auth, "Where is ATP produced?")
    assert "images" not in body


def test_unplaced_figures_are_still_listed(client, auth, stub_generation):
    upload(client, auth, "krebs.pdf", pdf_with_figure())
    body = ask(client, auth, "How does the Krebs cycle release energy?")
    assert [image["placed"] for image in body["images"]] == [False]


# --------------------------------------------------------------------------- caption text, reminder, selection
def test_caption_includes_the_description_lines_after_the_label():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Rivets join panels that cannot be reached from behind.", fontsize=11)
    page.insert_image(fitz.Rect(72, 100, 272, 250), stream=png())
    page.insert_text((72, 275), "Figure 7.5\nPOP blind rivet: standard open type\n(Tucker Fasteners Ltd)",
                     fontsize=9)
    page.insert_text((72, 340), "Next paragraph of body text.", fontsize=11)
    [section] = extract_sections("a.pdf", doc.tobytes())
    assert section["images"][0]["caption"] == (
        "Figure 7.5 POP blind rivet: standard open type (Tucker Fasteners Ltd)")


def test_caption_stops_at_the_next_caption():
    items = [("image", {}), ("text", "Figure 1", 9, None), ("text", "First figure", 9, None),
             ("text", "Figure 2", 9, None), ("text", "Second figure", 9, None)]
    from app.services.document_parser import _caption_near

    assert _caption_near(items, 0) == "Figure 1 First figure"


def test_figure_reminder_is_added_only_when_the_context_has_figures():
    from app.prompt_builder import build_prompt_bundle
    from app.prompt_router import route_message

    route = route_message("explain rivets")
    with_figures = build_prompt_bundle(route, context="[S1] text\n[IMG1: Figure 7.5 rivet]\nmore")
    plain = build_prompt_bundle(route, context="[S1] text only")
    user = with_figures["messages"][1]["content"]
    assert "## Figures" in user and "## Figures" not in plain["messages"][1]["content"]
    assert "at most 2 figures" in user
    assert user.rstrip().endswith("## Explanation")        # the closing cue stays last


def test_selection_prefers_matching_captions_and_caps_each_passage(monkeypatch):
    manifests = {
        "a": [{"image_id": f"a{i}" * 16, "anchor": 1, "caption": f"Figure 7.{i} peel rivet", "page": 1}
              for i in range(4)],
        "b": [{"image_id": "b0" * 16, "anchor": 5, "caption": "Figure 7.5 POP blind rivet open type",
               "page": 4}],
    }
    monkeypatch.setattr(images_service, "_manifest", lambda course, doc: manifests[doc])
    passages = [
        {"text": "x", "metadata": {"document_id": "a"}, "chunk_range": (1, 1), "chunk_ends": {1: 1}},
        {"text": "y", "metadata": {"document_id": "b"}, "chunk_range": (5, 5), "chunk_ends": {5: 1}},
    ]
    images = images_service.attach_images(passages, 1, limit=3, question="Explain POP blind rivet types")
    captions = [i["caption"] for i in images]
    assert "Figure 7.5 POP blind rivet open type" in captions     # best match always kept
    assert sum(c.endswith("peel rivet") for c in captions) == 2   # at most 2 from one passage
    assert [i["marker"] for i in images] == ["IMG1", "IMG2", "IMG3"]
    assert [i["passage"] for i in images] == sorted(i["passage"] for i in images)  # reading order
