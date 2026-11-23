"""Per-course BM25 keyword search, used next to dense search (hybrid retrieval).

Dense embeddings miss exact technical terms (CYP2C19, IPSG, drug names); BM25
finds them. The index is built in memory from one course's searchable chunks
and cached briefly per process.

Security: the index only *ranks chunk IDs*. Callers must re-fetch the chunks
from Chroma with the full course/visibility/retrieval filter before using
them, so a stale cache can never expose a deleted or disabled document.
"""
from collections import Counter
from dataclasses import dataclass, field
import math
import re
import threading
import time

# Rebuild a course index at most this often per process; writes in this process
# invalidate it immediately (see invalidate_course).
CACHE_TTL_SECONDS = 30.0
_K1 = 1.5
_B = 0.75
_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i if in "
    "into is it its me my of on or our so such than that the their them then there these they "
    "this to was we were what when where which who whom why will with would you your about also "
    "explain describe tell give list define please".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens without stopwords; keeps codes such as ``cyp2c19``."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


@dataclass
class KeywordIndex:
    """BM25 (Okapi) over one course's chunks."""
    ids: list[str]
    term_freqs: list[Counter]
    lengths: list[int]
    doc_freq: Counter = field(default_factory=Counter)
    avg_length: float = 0.0

    @classmethod
    def build(cls, ids: list[str], texts: list[str]) -> "KeywordIndex":
        term_freqs = [Counter(tokenize(text)) for text in texts]
        index = cls(ids=list(ids), term_freqs=term_freqs,
                    lengths=[sum(tf.values()) for tf in term_freqs])
        for tf in term_freqs:
            index.doc_freq.update(tf.keys())
        index.avg_length = (sum(index.lengths) / len(index.lengths)) if index.lengths else 0.0
        return index

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Return up to ``limit`` ``(chunk_id, score)`` pairs with a positive score, best first."""
        terms = set(tokenize(query))
        if not terms or not self.ids:
            return []
        total = len(self.ids)
        idf = {term: math.log(1 + (total - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
               for term in terms if self.doc_freq[term]}
        if not idf:
            return []
        scores = []
        for i, tf in enumerate(self.term_freqs):
            score = 0.0
            norm = _K1 * (1 - _B + _B * self.lengths[i] / (self.avg_length or 1))
            for term, weight in idf.items():
                count = tf.get(term, 0)
                if count:
                    score += weight * count * (_K1 + 1) / (count + norm)
            if score > 0:
                scores.append((self.ids[i], score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:max(0, limit)]


_lock = threading.Lock()
_cache: dict[tuple[str, int], tuple[float, KeywordIndex]] = {}


def get_course_index(identity: str, course_id: int, loader) -> KeywordIndex:
    """Return the cached index for (embedding identity, course), rebuilding it when stale.

    ``loader()`` returns ``(ids, texts)`` for the course's searchable chunks.
    """
    key = (identity, int(course_id))
    now = time.monotonic()
    with _lock:
        cached = _cache.get(key)
        if cached and now - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]
    ids, texts = loader()
    index = KeywordIndex.build(ids, texts)
    with _lock:
        _cache[key] = (now, index)
    return index


def invalidate_course(course_id: int | None = None) -> None:
    """Forget cached indexes for one course (or all) after a write in this process."""
    with _lock:
        if course_id is None:
            _cache.clear()
        else:
            for key in [key for key in _cache if key[1] == int(course_id)]:
                del _cache[key]
