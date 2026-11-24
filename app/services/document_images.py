"""Images embedded in course documents: filter, store, attach to answers, serve.

Indexing: document_parser finds embedded images and chunker anchors each one
after a chunk (``image_refs``) or before the first chunk (``image_refs_before``).
``store_document_images`` drops icons, repeated logos/backgrounds and unreadable
formats, converts the rest to PNG (max 1600 px) and writes them with a
``manifest.json`` to ``storage/document_images/<hash of course+document>/``.

Answering: ``attach_images`` picks the images anchored inside or right at the
edges of the passages the model will see and numbers them ``IMG1..``.
``retrieval.build_context_and_sources`` puts ``[IMGn: caption]`` where each
image sits; ``resolve_markers`` keeps only the markers the model may use.

Access: images are course material. They are found only through passages that
already passed the course/retrieval/visibility filter, and ``image_path``
checks the same filter before a file is served (no public links).
"""
from collections import Counter
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import re
import shutil
import uuid

from PIL import Image

from app.config import ROOT

logger = logging.getLogger(__name__)

IMAGE_ROOT = ROOT / "storage" / "document_images"
MIN_SIDE = 100                 # smaller images are icons, bullets or decorations
MAX_SIDE = 1600                # stored images are scaled down to fit this box
MAX_SOURCE_PIXELS = 40_000_000  # refuse to decode anything larger (decompression bombs)
REPEAT_LIMIT = 3               # an image used this often is a logo/background, not content
MAX_PER_DOCUMENT = 200
MAX_PER_ANSWER = 6             # candidates offered to the model, so it still has a choice
MAX_PLACED_PER_ANSWER = 2      # markers kept in the answer; matches Moodle's MAX_FIGURES_PER_ANSWER
MAX_PER_PASSAGE = 2            # so one passage cannot use up every figure slot
_IMAGE_ID = re.compile(r"^[0-9a-f]{32}$")
_MARKER = re.compile(r"\[(IMG\d+)(?::[^\]\n]*)?\]")


def _folder(course_id: int, document_id: str) -> Path:
    """Per-document folder; hashed so a Moodle document_id can never form a path."""
    digest = hashlib.sha256(f"{int(course_id)}:{document_id}".encode()).hexdigest()[:32]
    return IMAGE_ROOT / digest


def _to_png(data: bytes) -> tuple[bytes, int, int] | None:
    """Validated PNG bytes and size, or None for tiny, huge or unreadable images (e.g. EMF)."""
    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            if min(width, height) < MIN_SIDE or width * height > MAX_SOURCE_PIXELS:
                return None
            image.load()
            mode = "RGBA" if image.mode in ("RGBA", "LA", "P", "PA") else "RGB"
            converted = image.convert(mode)
        converted.thumbnail((MAX_SIDE, MAX_SIDE))
        buffer = BytesIO()
        converted.save(buffer, "PNG", optimize=True)
        return buffer.getvalue(), converted.width, converted.height
    except Exception:  # Pillow raises many types for unsupported/corrupt data
        return None


def delete_document_images(course_id: int, document_id: str) -> None:
    """Remove every stored image of a document (no error if there are none)."""
    shutil.rmtree(_folder(course_id, document_id), ignore_errors=True)


def store_document_images(course_id: int, document_id: str, sections: list[dict],
                          chunks: list[dict]) -> int:
    """Replace the document's stored images with the ones anchored in ``chunks``; return the count."""
    occurrences = {image["ref"]: image for section in sections for image in section.get("images") or []}
    uses = Counter(image["hash"] for image in occurrences.values())
    anchored: list[tuple[int, int]] = []          # (anchor, ref) in reading order
    for chunk in chunks:
        anchored += [(-1, ref) for ref in chunk.get("image_refs_before", [])]
        anchored += [(chunk["chunk_index"], ref) for ref in chunk.get("image_refs", [])]

    folder = _folder(course_id, document_id)
    staging = folder.with_name(f"{folder.name}.{uuid.uuid4().hex}.tmp")
    entries: list[dict] = []
    seen: set[str] = set()
    try:
        for anchor, ref in anchored:
            image = occurrences.get(ref)
            if image is None or uses[image["hash"]] >= REPEAT_LIMIT or image["hash"] in seen:
                continue
            if len(entries) >= MAX_PER_DOCUMENT:
                break
            converted = _to_png(image["data"])
            if converted is None:
                continue
            seen.add(image["hash"])
            png, width, height = converted
            image_id = hashlib.sha256(f"{course_id}:{document_id}:{image['hash']}".encode()).hexdigest()[:32]
            staging.mkdir(parents=True, exist_ok=True)
            (staging / f"{image_id}.png").write_bytes(png)
            entries.append({"image_id": image_id, "anchor": anchor, "caption": image.get("caption"),
                            "page": image.get("page"), "width": width, "height": height})
        delete_document_images(course_id, document_id)
        if entries:
            (staging / "manifest.json").write_text(json.dumps({
                "course_id": int(course_id), "document_id": str(document_id), "images": entries,
            }), encoding="utf-8")
            os.replace(staging, folder)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return len(entries)


def _manifest(course_id: int, document_id: str) -> list[dict]:
    path = _folder(course_id, document_id) / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if data.get("course_id") != int(course_id) or data.get("document_id") != str(document_id):
        return []
    return data.get("images", [])


def _caption(entry: dict) -> str:
    """Caption for the prompt: document text, so brackets/newlines are removed."""
    caption = re.sub(r"[\[\]\n\r]+", " ", entry.get("caption") or "").strip()
    if caption:
        return caption[:200]
    return f"image on page {entry['page']}" if entry.get("page") else "image"


def _caption_score(entry: dict, question_terms: set[str]) -> int:
    """How many question words the caption shares (0 without a caption or question)."""
    if not question_terms or not entry.get("caption"):
        return 0
    from app.services.keyword_index import tokenize

    return len(question_terms & set(tokenize(entry["caption"])))


def attach_images(passages: list[dict], course_id: int, limit: int = MAX_PER_ANSWER,
                  question: str = "") -> list[dict]:
    """Images for the passages, numbered ``IMG1..`` in passage order.

    A passage covering chunks i..j (``chunk_range``) gets the images anchored
    from i-1 to j: just before it, inside it, or right after it. Each image is
    used once. When there are more candidates than ``limit``, captions that
    share words with ``question`` win, then earlier passages; a passage keeps
    at most ``MAX_PER_PASSAGE`` images. ``offset`` is where the marker goes in
    the passage text.
    """
    from app.services.keyword_index import tokenize

    question_terms = set(tokenize(question))
    candidates: list[tuple] = []          # (score, passage position, anchor, entry, passage)
    seen: set[str] = set()
    for position, passage in enumerate(passages):
        span = passage.get("chunk_range")
        if not span:
            continue
        start, end = span
        document_id = str(passage.get("metadata", {}).get("document_id", ""))
        for entry in _manifest(course_id, document_id):
            if start - 1 <= entry["anchor"] <= end and entry["image_id"] not in seen:
                seen.add(entry["image_id"])
                candidates.append((_caption_score(entry, question_terms), position, entry["anchor"],
                                   entry, passage))

    chosen: list[tuple] = []
    per_passage: dict[int, int] = {}
    for candidate in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
        if len(chosen) >= limit:
            break
        if per_passage.get(candidate[1], 0) >= MAX_PER_PASSAGE:
            continue
        per_passage[candidate[1]] = per_passage.get(candidate[1], 0) + 1
        chosen.append(candidate)

    images: list[dict] = []
    for _, position, _, entry, passage in sorted(chosen, key=lambda c: (c[1], c[2])):
        start = passage["chunk_range"][0]
        metadata = passage.get("metadata", {})
        offset = 0 if entry["anchor"] < start else passage.get("chunk_ends", {}).get(
            entry["anchor"], len(passage["text"]))
        images.append({
            "marker": f"IMG{len(images) + 1}", "image_id": entry["image_id"],
            "document_id": str(metadata.get("document_id", "")),
            "filename": str(metadata.get("filename", "")),
            "page": entry.get("page"), "caption": _caption(entry),
            "passage": position, "offset": offset,
        })
    return images


def resolve_markers(answer: str, images: list[dict], limit: int = MAX_PLACED_PER_ANSWER,
                    stats: dict | None = None) -> tuple[str, list[dict]]:
    """Normalise ``[IMGn: ...]`` to ``[IMGn]``, drop markers that were not supplied or repeat.

    Only the first ``limit`` distinct supplied markers (in answer order) are
    kept; later ones are dropped like unknown ones. A dropped marker alone on
    its line takes the line with it. If ``stats`` is given, its ``over_limit``
    is set to the number of distinct markers dropped for the limit.
    Returns the cleaned answer and the images with ``placed`` set.
    """
    known = {image["marker"] for image in images}
    placed: list[str] = []
    over_limit: set[str] = set()
    dropped = "\x00"                      # placeholder so lone-marker lines can be removed whole

    def replace(match: re.Match) -> str:
        marker = match.group(1)
        if marker not in known or marker in placed:
            return dropped
        if len(placed) >= limit:
            over_limit.add(marker)
            return dropped
        placed.append(marker)
        return f"[{marker}]"

    cleaned = _MARKER.sub(replace, answer)
    cleaned = re.sub(rf"^[ \t]*(?:{dropped}[ \t]*)+(?:\n|$)", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.replace(dropped, "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if stats is not None:
        stats["over_limit"] = len(over_limit)
    return cleaned, [{**image, "placed": image["marker"] in placed} for image in images]


def image_path(image_id: str, course_id: int, document_id: str, document_visible) -> Path | None:
    """The stored file if the image belongs to this course document and it is searchable.

    ``document_visible(course_id, document_id)`` must apply the vector store's
    course/retrieval/visibility filter, so deleted or disabled documents fail.
    """
    if not _IMAGE_ID.match(image_id or ""):
        return None
    if not any(entry["image_id"] == image_id for entry in _manifest(course_id, document_id)):
        return None
    if not document_visible(course_id, document_id):
        return None
    path = _folder(course_id, document_id) / f"{image_id}.png"
    return path if path.is_file() else None
