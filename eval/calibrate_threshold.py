"""Measure a relevance threshold (RETRIEVAL_MAX_DISTANCE) for the active embedding model.

Distances are model-specific, so every embedding model offered in Moodle needs
its own value in ``CALIBRATED_MAX_DISTANCE`` (app/config.py); without one,
off-topic questions are not filtered (``GET /admin/config`` reports
``relevance_filter: off``). Re-run this after changing chunking, too.

Input: a CSV with columns ``course_id``, ``question``, ``on_topic`` (1/0). Use
questions answered by the course's indexed documents (on-topic) and questions
the course does not cover (off-topic, including other courses' topics).

For every question this runs the same hybrid search as chat (against the
local Chroma store, no answer generation) and records the best candidate's
cosine distance. It prints both distributions and a suggested threshold
between the highest on-topic and the lowest off-topic distance.

Cost: one embedding request per question with the configured embedding
provider (paid for OpenAI/Gemini; tiny). No chat-model calls.

Usage: python eval/calibrate_threshold.py questions.csv
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.services import vector_store  # noqa: E402


def best_distance(question: str, course_id: int, top_k: int) -> float | None:
    """Smallest cosine distance among the hybrid-search candidates (None: nothing indexed)."""
    rows = vector_store.hybrid_search([question], course_id, True, top_k)
    distances = [row["distance"] for row in rows if row.get("distance") is not None]
    return min(distances) if distances else None


def suggest(on_topic: list[float], off_topic: list[float]) -> tuple[float | None, str]:
    """Midpoint between the distributions, and a note when they overlap."""
    if not on_topic or not off_topic:
        return None, "Need at least one on-topic and one off-topic question with results."
    highest_on, lowest_off = max(on_topic), min(off_topic)
    if highest_on < lowest_off:
        return round((highest_on + lowest_off) / 2, 2), "Clean separation."
    kept = sum(1 for d in off_topic if d <= highest_on)
    return round(highest_on, 2), (f"Overlap: keeping every on-topic question also keeps {kept} of "
                                  f"{len(off_topic)} off-topic questions. Review those questions.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", type=Path, help="CSV with course_id, question, on_topic")
    args = parser.parse_args()

    settings = get_settings()
    on_topic: list[float] = []
    off_topic: list[float] = []
    with args.questions.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            distance = best_distance(row["question"], int(row["course_id"]), settings.top_k)
            label = "on " if row["on_topic"].strip() in {"1", "true", "yes"} else "off"
            print(f"{label} {distance if distance is not None else 'n/a':>8}  {row['question'][:80]}")
            if distance is not None:
                (on_topic if label == "on " else off_topic).append(distance)

    print(f"\nEmbedding model: {settings.embedding_provider}/{settings.resolved_embedding_model}")
    for name, values in (("on-topic", on_topic), ("off-topic", off_topic)):
        if values:
            print(f"{name:>9}: n={len(values)} min={min(values):.3f} "
                  f"median={statistics.median(values):.3f} max={max(values):.3f}")
    threshold, note = suggest(on_topic, off_topic)
    print(f"\nSuggested threshold: {threshold}  ({note})")
    if threshold is not None:
        print(f'Add to CALIBRATED_MAX_DISTANCE: ("{settings.embedding_provider.lower()}", '
              f'"{settings.resolved_embedding_model}"): {threshold},')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
