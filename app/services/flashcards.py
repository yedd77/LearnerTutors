"""/flashcards revision decks: the content model, the sufficiency check and LLM generation.

Mirrors ``practice_quiz``: the LLM writes a loose ``FlashcardDraft``;
``build_deck`` turns it into a validated ``FlashcardDeck`` (3 to the requested
number of cards, distinct fronts), mapping each card's passage marker to that
passage's file and page so a source can never be invented. Moodle renders the
deck as interactive flip cards. Nothing here decides access: the chunks passed
in were already scoped to the request's course by the graph.
"""
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from app.config import Settings, get_settings
from app.prompt_builder import CORE_DIR, _fill, _read_prompt
from app.prompt_registry import PROMPT_DIR
from app.services import llm_provider
# describe_problem is shared with /quiz: the same content-free summary for retries and logs.
from app.services.practice_quiz import (_clean, _passage_sources, _source_for, _tidy, _topic,
                                        describe_problem)

FLASHCARDS_TOOL_ID = "flashcards"
PROMPT_FILE = PROMPT_DIR / "graph" / "flashcards.md"
DEFAULT_CARDS = 5
MIN_CARDS = 3
MAX_CARDS = 15
# "10 cards", "12 flashcards", "8 flash cards"; other numbers ("week 3") are ignored.
_COUNT = re.compile(r"\b(\d{1,3})\s*(?:flash\s*cards?|cards?)\b", re.IGNORECASE)


# --------------------------------------------------------------------------- content model
class FlashcardSource(BaseModel):
    """The course document page a card comes from."""
    filename: str = Field(min_length=1, max_length=255)
    page: int | None = None


class Flashcard(BaseModel):
    """One card: a prompt on the front, a self-contained answer on the back."""
    front: str = Field(min_length=1, max_length=300)   # the question / prompt
    back: str = Field(min_length=1, max_length=800)    # the answer, self-contained
    source: FlashcardSource | None = None

    @field_validator("front", "back", mode="before")
    @classmethod
    def _tidy_text(cls, value):
        """Collapse whitespace."""
        return _tidy(value)

    @model_validator(mode="after")
    def _back_is_an_answer(self) -> "Flashcard":
        """The back must not just repeat the front."""
        if self.back.casefold() == self.front.casefold():
            raise ValueError("A card's back must answer its front, not repeat it.")
        return self


class FlashcardDeck(BaseModel):
    """MIN_CARDS..MAX_CARDS cards (at most the requested number) on one topic.

    Validate with ``context={"max_cards": n}`` to cap the deck at the number
    the student asked for; ``build_deck`` does this.
    """
    title: str = Field(min_length=1, max_length=160)   # short topic title
    cards: list[Flashcard]

    @field_validator("title", mode="before")
    @classmethod
    def _tidy_title(cls, value):
        """Collapse whitespace."""
        return _tidy(value)

    @field_validator("cards")
    @classmethod
    def _check_cards(cls, cards: list[Flashcard], info: ValidationInfo) -> list[Flashcard]:
        """Between MIN_CARDS and the requested maximum, with no duplicate fronts."""
        limit = MAX_CARDS
        if info.context and info.context.get("max_cards"):
            limit = min(MAX_CARDS, int(info.context["max_cards"]))
        if not MIN_CARDS <= len(cards) <= limit:
            raise ValueError(f"The deck needs between {MIN_CARDS} and {limit} cards.")
        if len({card.front.casefold() for card in cards}) != len(cards):
            raise ValueError("The deck contains duplicate card fronts.")
        return cards


# --------------------------------------------------------------------------- LLM output
# Deliberately loose, so a bad deck is caught by build_deck (and retried)
# instead of failing as a provider error.
class FlashcardDraftCard(BaseModel):
    """A card as the model writes it; checked by ``build_deck``."""
    front: str = ""
    back: str = ""
    source: str = ""          # passage marker from the context, e.g. "S3"


class FlashcardDraft(BaseModel):
    """Structured output of the flashcards call."""
    title: str = ""
    coverage: Literal["sufficient", "limited"] = "sufficient"
    cards: list[FlashcardDraftCard] = Field(default_factory=list)


def is_flashcards(tool_id: str) -> bool:
    """True for the /flashcards tool."""
    return tool_id == FLASHCARDS_TOOL_ID


def requested_count(question: str) -> int:
    """Number of cards the student asked for, clamped to MIN_CARDS..MAX_CARDS;
    DEFAULT_CARDS when no number is given.

    Only digits followed by "cards"/"flashcards"/"flash cards" count, e.g.
    "10 cards", "12 flashcards"; other numbers ("week 3", "chapter 2") are ignored.
    """
    found = _COUNT.search(question or "")
    if not found:
        return DEFAULT_CARDS
    return max(MIN_CARDS, min(MAX_CARDS, int(found.group(1))))


def build_deck(draft: FlashcardDraft, matches: list[dict], fallback_title: str,
               requested: int) -> FlashcardDeck:
    """Validate the draft into a ``FlashcardDeck``. Raises ``ValidationError``.

    More cards than ``requested`` is an error (retried), never trimmed silently.
    A source is taken only from a marker that names a passage the model was
    shown; anything else leaves the card without a source.
    """
    sources = _passage_sources(matches, FlashcardSource)
    return FlashcardDeck.model_validate(
        {"title": _clean(draft.title) or _clean(fallback_title)[:160],
         "cards": [{"front": card.front, "back": card.back,
                    "source": _source_for(card.source, sources)} for card in draft.cards]},
        context={"max_cards": requested},
    )


def cited_pages(deck: FlashcardDeck) -> set[tuple[str, int | None]]:
    """(filename, page) of every card's source."""
    return {(card.source.filename, card.source.page) for card in deck.cards if card.source}


# --------------------------------------------------------------------------- sufficiency
def enough_material(matches: list[dict], settings: Settings | None = None) -> bool:
    """Deterministic minimum before any LLM call: enough relevant chunks and text."""
    settings = settings or get_settings()
    # Merged passages (neighbour expansion) count every chunk they contain.
    chunks = sum(int(match.get("chunk_count", 1)) for match in matches)
    chars = sum(len(match.get("text", "")) for match in matches)
    return (chunks >= max(1, settings.flashcards_min_chunks)
            and chars >= max(0, settings.flashcards_min_context_chars))


# --------------------------------------------------------------------------- generation
def build_messages(question: str, context: str, history: str, requested: int,
                   problem: str = "") -> list[dict[str, str]]:
    """Messages for the structured flashcards call: grounding rules plus the task prompt.

    ``problem`` explains why the previous attempt was rejected (retry only).
    """
    grounding = CORE_DIR / "grounding.md"
    rules = _read_prompt(str(grounding)) if grounding.is_file() else ""
    system = ("You write accurate revision flashcards for university students. "
              "Return only valid JSON.\n\n" + rules).strip()
    prompt = _fill(_read_prompt(str(PROMPT_FILE)), context=context, history=history,
                   question=question, count=str(requested))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if problem:
        messages.append({"role": "user", "content": (
            f"Your previous flashcards could not be used: {problem}\n"
            f"Write the whole deck again and follow every rule: between {MIN_CARDS} and "
            f"{requested} cards, each with a different front and a self-contained back.")})
    return messages


async def generate_draft(question: str, context: str, history: list[dict], requested: int,
                         provider: str, model: str,
                         fallback: tuple[str, str] | None = None,
                         problem: str = "") -> FlashcardDraft:
    """Ask the configured model for the deck. Raises ProviderError on failure."""
    messages = build_messages(question, context, llm_provider._history_text(history),
                              requested, problem)
    return await llm_provider.generate_structured(messages, FlashcardDraft, provider, model,
                                                  fallback=fallback)


# --------------------------------------------------------------------------- student messages
def ready_message(title: str, count: int) -> str:
    """Short line shown above the deck; states the real count. Never contains card text."""
    return f"Here are {count} flashcards on {title}. Tap a card to see the answer."


def limited_message(question: str) -> str:
    """Honest reply when the material cannot support at least MIN_CARDS cards."""
    return (f"I couldn't find enough material about \"{_topic(question)}\" in the course "
            "documents to make reliable flashcards, so I haven't created any. Try a wider "
            "topic, or your lecturer may not have uploaded material on it yet.")


def failed_message() -> str:
    """Reply when no valid deck came back after the retry."""
    return ("Sorry, I couldn't create reliable flashcards from the course material this "
            "time, so I haven't shown any. Please try /flashcards again, perhaps with a "
            "slightly different topic.")
