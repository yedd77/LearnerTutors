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
    prompt_path = Path(path)
    return _read_cached(path, prompt_path.stat().st_mtime)

# load shared prompt files from core directory
def _core_prompt() -> str:
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
    # Replace known placeholders only; do not use format(), which would interpret
    # braces that may appear in code examples or retrieved documents.
    for key, value in values.items():
        template = template.replace("{" + key + "}", value or "")
    return re.sub(r"\{(\w+)\}",
            lambda m: values.get(m.group(1), m.group(0)) or "", template)

# build the prompt payload for ollama
def build_prompt_bundle(
    route: RouteResult,
    context: str = "",
    history: str = "",
) -> dict:
    """Build messages for Ollama; context/history must already be authorized by caller."""
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