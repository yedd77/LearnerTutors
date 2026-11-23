"""Signed, expiring tokens for actions that need the student's explicit approval.

The backend keeps no conversation state (Moodle sends the history), so a
pending action travels as a token: ``base64url(payload).hmac``. The payload
binds the action to one course, one user and the original request, and the
HMAC (keyed with RAG_API_KEY) stops the client from altering any of it. A token
only acts when it comes back with an explicit ``pending_decision``; a free-text
"yes" is never treated as consent.
"""
import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from app.config import get_settings

_PURPOSE = b"ragassistant-pending-action-v1:"


class PendingActionError(ValueError):
    """The token is malformed, tampered with, expired, or for another user/course/action."""


@dataclass(frozen=True)
class PendingRequest:
    """The original request a token stands for."""
    action: str
    question: str
    tool_id: str
    expires_at: int


def _sign(body: bytes) -> str:
    """HMAC-SHA256 of ``body``, hex, keyed with RAG_API_KEY and a purpose prefix."""
    secret = get_settings().rag_api_key
    if not secret:
        raise RuntimeError("RAG_API_KEY is not configured.")
    return hmac.new(secret.encode(), _PURPOSE + body, hashlib.sha256).hexdigest()


def issue(action: str, *, course_id: int, user_id: int, question: str, tool_id: str,
          ttl_seconds: int | None = None) -> tuple[str, int]:
    """Return ``(token, expires_at)`` for ``action`` on behalf of this course and user."""
    if ttl_seconds is None:
        ttl_seconds = get_settings().pending_action_ttl_minutes * 60
    expires_at = int(time.time()) + max(60, int(ttl_seconds))
    payload = {"a": action, "c": int(course_id), "u": int(user_id), "q": question,
               "t": tool_id, "e": expires_at}
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")).rstrip(b"=")
    return f"{body.decode()}.{_sign(body)}", expires_at


def verify(token: str, action: str, *, course_id: int, user_id: int,
           now: float | None = None) -> PendingRequest:
    """Return the original request if ``token`` is valid for this action, course and user."""
    try:
        body_text, signature = token.rsplit(".", 1)
        body = body_text.encode("ascii")
    except (ValueError, AttributeError, UnicodeEncodeError) as exc:
        raise PendingActionError("Malformed token.") from exc
    if not hmac.compare_digest(signature, _sign(body)):
        raise PendingActionError("Invalid token signature.")
    try:
        payload = json.loads(base64.urlsafe_b64decode(body + b"=" * (-len(body) % 4)))
        expires_at = int(payload["e"])
        question, tool_id = str(payload["q"]), str(payload["t"])
        owner = (payload["a"], int(payload["c"]), int(payload["u"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise PendingActionError("Malformed token payload.") from exc
    if owner != (action, int(course_id), int(user_id)):
        raise PendingActionError("Token is for a different action, course or user.")
    if expires_at <= (time.time() if now is None else now):
        raise PendingActionError("Token has expired.")
    return PendingRequest(action, question, tool_id, expires_at)
