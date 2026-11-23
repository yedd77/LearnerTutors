"""Temporary generated files (e.g. study sheet PDFs): store, fetch for the owner, expire.

Each file is ``<id>.<ext>`` plus a ``<id>.json`` record of its owner (course and
user), download name, content type and expiry. Files are never permanent:
an expired file is refused and deleted, and ``cleanup_expired`` sweeps the
directory (run periodically from the app lifespan).

Access is checked here, not by the URL: a file is only returned when the
caller's course_id and user_id (supplied by Moodle after its own permission
checks) match the record. A mismatch is reported exactly like a missing file,
so IDs of other students' files cannot be probed.
"""
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = ROOT / "storage" / "generated"

logger = logging.getLogger(__name__)
_ID = re.compile(r"[0-9a-f]{32}")
_EXTENSIONS = {"application/pdf": "pdf"}


class GeneratedFileNotFound(LookupError):
    """Unknown ID, or the file belongs to another course or user."""


class GeneratedFileExpired(LookupError):
    """The file existed for this owner but its expiry time has passed."""


@dataclass(frozen=True)
class StoredFile:
    """What the API tells the client about a stored file."""
    file_id: str
    filename: str
    content_type: str
    expires_at: int


def _paths(file_id: str, content_type: str) -> tuple[Path, Path]:
    """Data and record paths for ``file_id``."""
    return (GENERATED_DIR / f"{file_id}.{_EXTENSIONS[content_type]}",
            GENERATED_DIR / f"{file_id}.json")


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a temp file and rename, so readers never see a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _safe_filename(name: str, extension: str) -> str:
    """A download name made only of safe characters, ending in ``.extension``."""
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "", name).strip(" .-_")[:80] or "download"
    stem = re.sub(r"\s+", "_", stem)
    return stem if stem.lower().endswith(f".{extension}") else f"{stem}.{extension}"


def store(data: bytes, *, course_id: int, user_id: int, filename: str,
          ttl_seconds: int, content_type: str = "application/pdf",
          kind: str = "study_sheet") -> StoredFile:
    """Save ``data`` under a new random ID for this course and user; return its details."""
    if not data:
        raise ValueError("Refusing to store an empty file.")
    if content_type not in _EXTENSIONS:
        raise ValueError(f"Unsupported content type: {content_type}")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex
    expires_at = int(time.time()) + max(60, int(ttl_seconds))
    download_name = _safe_filename(filename, _EXTENSIONS[content_type])
    data_path, record_path = _paths(file_id, content_type)
    _write_atomic(data_path, data)
    record = {"file_id": file_id, "kind": kind, "course_id": int(course_id),
              "user_id": int(user_id), "filename": download_name,
              "content_type": content_type, "created_at": int(time.time()),
              "expires_at": expires_at}
    # The record is written last: a data file without a record is never served.
    _write_atomic(record_path, json.dumps(record).encode("utf-8"))
    logger.info("generated_file.stored id=%s kind=%s course=%s bytes=%s expires_at=%s",
                file_id, kind, course_id, len(data), expires_at)
    return StoredFile(file_id, download_name, content_type, expires_at)


def _read_record(file_id: str) -> dict | None:
    """The JSON record for ``file_id``, or None if missing or unreadable."""
    try:
        return json.loads((GENERATED_DIR / f"{file_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _delete(file_id: str, record: dict | None) -> None:
    """Remove a file and its record, ignoring files already gone."""
    paths = [GENERATED_DIR / f"{file_id}.json"]
    if record and record.get("content_type") in _EXTENSIONS:
        paths.insert(0, _paths(file_id, record["content_type"])[0])
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("generated_file.delete_failed id=%s path=%s", file_id, path.name)


def open_for(file_id: str, *, course_id: int, user_id: int,
             now: float | None = None) -> tuple[Path, dict]:
    """Return ``(path, record)`` if this course and user own an unexpired file.

    Raises GeneratedFileNotFound (unknown, malformed or someone else's ID) or
    GeneratedFileExpired (deleted on the spot).
    """
    if not _ID.fullmatch(file_id or ""):
        raise GeneratedFileNotFound(file_id)
    record = _read_record(file_id)
    if (record is None or record.get("course_id") != int(course_id)
            or record.get("user_id") != int(user_id)
            or record.get("content_type") not in _EXTENSIONS):
        raise GeneratedFileNotFound(file_id)
    if int(record.get("expires_at", 0)) <= (time.time() if now is None else now):
        _delete(file_id, record)
        raise GeneratedFileExpired(file_id)
    path = _paths(file_id, record["content_type"])[0]
    if not path.is_file():
        raise GeneratedFileNotFound(file_id)
    return path, record


def cleanup_expired(now: float | None = None) -> int:
    """Delete expired files, orphaned data files and stale temp files; return files removed."""
    if not GENERATED_DIR.is_dir():
        return 0
    now = time.time() if now is None else now
    removed = 0
    recorded: set[str] = set()
    for record_path in GENERATED_DIR.glob("*.json"):
        file_id = record_path.stem
        record = _read_record(file_id)
        if record is None or int(record.get("expires_at", 0)) <= now:
            _delete(file_id, record)
            removed += 1
        else:
            recorded.add(file_id)
    for path in GENERATED_DIR.iterdir():
        if not path.is_file() or path.suffix == ".json":
            continue
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        # A data file with no record is unreachable; grace period covers a store in progress.
        if path.stem.split(".")[0] not in recorded and age > 60:
            path.unlink(missing_ok=True)
            removed += 1
    if removed:
        logger.info("generated_file.cleanup removed=%s", removed)
    return removed
