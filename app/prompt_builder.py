"""Assemble chat messages from the core prompts, the tool prompt and the request."""
from functools import lru_cache
from pathlib import Path
import re
from app.prompt_registry import PROMPT_DIR
from app.prompt_router import RouteResult

CORE_DIR = PROMPT_DIR / "core" 

# Read a prompt file and trim whitespace
@lru_cache(maxsize=64)
def _read_cached(path: str, mtime: float) -> str:
    """Cache prompt text in process memory instead of rereading it per request."""
    return Path(path).read_text(encoding="utf-8").strip()

# wrapper to read a prompt file and cache it, using the file's mtime to invalidate the cache
def _read_prompt(path: str) -> str:
    """Return a prompt file's text, re-reading it only after the file is modified."""
    prompt_path = Path(path)
    return _read_cached(path, prompt_path.stat().st_mtime)

# load shared prompt files from core directory
def _core_prompt() -> str:
    """Join the shared system instructions: role, grounding rules and answer format."""
    parts = []
    for filename in ("instructions.md", "grounding.md", "answer_format.md"):
        path = CORE_DIR / filename
        if path.is_file():
            content = _read_prompt(str(path))
            if content:
                parts.append(content)
    return "\n\n".join(parts)

# Fill placeholders in a prompt template with provided values, leaving unknown placeholders unchanged
def _fill(template: str, **values: str) -> str:
    """Replace ``{name}`` placeholders with ``values``; unknown placeholders are kept.

    Substitution is a single pass over the template, so inserted values are
    never scanned again: retrieved text containing ``{question}`` or
    ``{history}`` reaches the model unchanged (fixes ERR-001).
    """
    # Replace known placeholders only; do not use format(), which would interpret
    # braces that may appear in code examples or retrieved documents.
    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            return match.group(0)
        return values[key] or ""

    return re.sub(r"\{(\w+)\}", replace, template)

# build the prompt payload for chat model
def build_prompt_bundle(
    route: RouteResult,
    context: str = "",
    history: str = "",
) -> dict:
    """Build chat messages for the chat model; context/history must already be authorized by caller."""
    tool = route.tool
    task_path = PROMPT_DIR / "tools" / tool.prompt_file
    template = _read_prompt(str(task_path))

    task_prompt = _fill(
        template,
        context=context,
        history=history,
        question=route.question,
    )

    return {
        "tool_id": tool.id,
        "tool_label": tool.label,
        "retrieval_mode": tool.retrieval_mode,
        "explicit_command": route.explicit_command,
        "question": route.question,
        "messages": [
            {"role": "system", "content": _core_prompt()},
            {"role": "user", "content": task_prompt},
        ],
    }