"""Enhanced content lane classification for PixelMem V3.

Classifies incoming text into one of seven content lanes, each with its
own extraction strategy.  This replaces the v2 classifier with improved
ordering (chit-chat is checked before episodic) and finer-grained lane
definitions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Lane enum ───────────────────────────────────────────────────────

class ContentLane(Enum):
    """The seven content lanes recognised by the V3 pipeline."""

    EPISODIC = "episodic"
    PREFERENCE = "preference"
    ASSISTANT_CONTENT = "assistant_content"
    CHIT_CHAT = "chit_chat"
    UPDATE = "update"
    WORKFLOW = "workflow"
    MIXED = "mixed"


# ── Classification result ───────────────────────────────────────────

@dataclass
class LaneResult:
    """Result of lane classification for a piece of text."""

    lane: ContentLane
    confidence: float
    sub_type: str = ""
    hints: dict = field(default_factory=dict)


# ── Per-lane extraction strategy configs ────────────────────────────

LANE_CONFIGS: dict[ContentLane, dict] = {
    ContentLane.EPISODIC: {
        "extraction": "fast",
        "add_timestamp": True,
        "priority": "high",
    },
    ContentLane.PREFERENCE: {
        "extraction": "fast",
        "add_timestamp": False,
        "priority": "high",
    },
    ContentLane.ASSISTANT_CONTENT: {
        "extraction": "llm",
        "add_timestamp": True,
        "priority": "medium",
    },
    ContentLane.CHIT_CHAT: {
        "extraction": "skip",
        "add_timestamp": False,
        "priority": "low",
    },
    ContentLane.UPDATE: {
        "extraction": "fast",
        "add_timestamp": True,
        "priority": "high",
    },
    ContentLane.WORKFLOW: {
        "extraction": "ast",
        "add_timestamp": True,
        "priority": "high",
    },
    ContentLane.MIXED: {
        "extraction": "fast+ast",
        "add_timestamp": True,
        "priority": "high",
    },
}


# ── Detection patterns (compiled once) ─────────────────────────────

_CODE_BLOCK_RE = re.compile(r"```")
_FILE_PATH_RE = re.compile(r"\b[\w/\\]+\.(py|js|ts|json|yaml|yml|toml|cfg|sh)\b")
_IMPORT_RE = re.compile(r"^\s*(import |from .+ import )", re.MULTILINE)
_FUNC_DEF_RE = re.compile(r"^\s*(def |function |const \w+ *= *\()", re.MULTILINE)
_CLASS_DEF_RE = re.compile(r"^\s*class \w+", re.MULTILINE)

_CHIT_CHAT_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|sure|bye|goodbye|"
    r"how are you|what'?s up|no worries|sounds good|got it|yep|nope"
    r"|(?:hi|hello|hey)[,!]?\s*(?:how are you|what'?s up|there))\s*[!?.]*$",
    re.IGNORECASE,
)
_CHIT_CHAT_SHORT_THRESHOLD = 12  # tokens-ish (word count)

_UPDATE_KEYWORDS = re.compile(
    r"\b(actually|correction|no longer|moved|changed|updated|rename[ds]?|"
    r"instead of|replace[ds]?|switch(?:ed)? to)\b",
    re.IGNORECASE,
)

_PREFERENCE_RE = re.compile(
    r"\b(I prefer|my (?:fav(?:ou?rite)?|preference)|always use|I like to use|"
    r"I (?:usually|always|never) )\b",
    re.IGNORECASE,
)

_ASSISTANT_RE = re.compile(
    r"\b(let me help|here(?:'?s| is) (?:a |an |the )?(?:summary|recommendation|story|"
    r"example|explanation)|I recommend|I suggest)\b",
    re.IGNORECASE,
)

_FIRST_PERSON_RE = re.compile(r"\bI\b")
_PAST_TENSE_RE = re.compile(
    r"\b(went|bought|visited|saw|met|did|was|were|had|got|made|took|came|gave|"
    r"found|told|said|felt|left|moved|started|finished|attended|traveled)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}|"
    r"last (?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"yesterday|today)\b",
    re.IGNORECASE,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _has_code_signals(text: str) -> bool:
    """Return True if *text* contains code-like content."""
    return bool(
        _CODE_BLOCK_RE.search(text)
        or _FILE_PATH_RE.search(text)
        or _IMPORT_RE.search(text)
        or _FUNC_DEF_RE.search(text)
        or _CLASS_DEF_RE.search(text)
    )


def _is_chit_chat(text: str) -> bool:
    """Return True if *text* looks like casual chit-chat."""
    stripped = text.strip()
    # Exact match on common short phrases
    if _CHIT_CHAT_RE.match(stripped):
        return True
    # Very short messages with no substance
    if len(stripped.split()) <= 3 and len(stripped) < 30:
        lowered = stripped.lower().rstrip("!?. ")
        if lowered in {
            "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
            "sure", "bye", "goodbye", "how are you", "what's up",
            "no worries", "sounds good", "got it", "yep", "nope", "yes",
            "no", "cool", "nice", "great", "awesome", "lol", "haha",
        }:
            return True
    return False


# ── Main classifier ────────────────────────────────────────────────

def classify_lane(
    text: str,
    context: Optional[dict] = None,
) -> LaneResult:
    """Classify *text* into a content lane.

    Parameters
    ----------
    text : str
        The raw text to classify.
    context : dict, optional
        Extra hints from the caller.  Recognised keys:
        ``has_code_blocks``, ``file_path``, ``is_assistant``,
        ``session_type``.

    Returns
    -------
    LaneResult
        The detected lane together with confidence, sub-type, and hints.
    """
    ctx = context or {}
    has_code = ctx.get("has_code_blocks", False) or _has_code_signals(text)
    is_assistant = ctx.get("is_assistant", False)
    has_first_person = bool(_FIRST_PERSON_RE.search(text))

    # ── 1. CHIT_CHAT (checked BEFORE episodic — fixes v2 ordering) ──
    if _is_chit_chat(text):
        return LaneResult(
            lane=ContentLane.CHIT_CHAT,
            confidence=0.95,
            sub_type="greeting" if re.match(r"^(hi|hello|hey)\b", text, re.I) else "filler",
            hints={"word_count": len(text.split())},
        )

    # ── 2. WORKFLOW — pure code / dev artefacts ─────────────────────
    if has_code and not has_first_person:
        sub = "code_block" if _CODE_BLOCK_RE.search(text) else "file_ref"
        return LaneResult(
            lane=ContentLane.WORKFLOW,
            confidence=0.90,
            sub_type=sub,
            hints={"has_code": True},
        )

    # ── 3. MIXED — first-person AND code ────────────────────────────
    if has_code and has_first_person:
        return LaneResult(
            lane=ContentLane.MIXED,
            confidence=0.85,
            sub_type="code_and_narrative",
            hints={"has_code": True, "has_first_person": True},
        )

    # ── 4. UPDATE — corrections / overrides ─────────────────────────
    if _UPDATE_KEYWORDS.search(text):
        return LaneResult(
            lane=ContentLane.UPDATE,
            confidence=0.85,
            sub_type="correction",
            hints={},
        )

    # ── 5. PREFERENCE — stated likes / dislikes ────────────────────
    if _PREFERENCE_RE.search(text):
        return LaneResult(
            lane=ContentLane.PREFERENCE,
            confidence=0.88,
            sub_type="stated_preference",
            hints={},
        )

    # ── 6. ASSISTANT_CONTENT — generated / recommendation text ──────
    if is_assistant or _ASSISTANT_RE.search(text):
        return LaneResult(
            lane=ContentLane.ASSISTANT_CONTENT,
            confidence=0.80 if is_assistant else 0.75,
            sub_type="recommendation" if _ASSISTANT_RE.search(text) else "generated",
            hints={"is_assistant": is_assistant},
        )

    # ── 7. EPISODIC — personal events / memories ───────────────────
    if has_first_person:
        conf = 0.70
        sub = "narrative"
        if _PAST_TENSE_RE.search(text):
            conf = 0.85
            sub = "past_event"
        if _DATE_RE.search(text):
            conf = min(conf + 0.10, 0.95)
            sub = "dated_event"
        return LaneResult(
            lane=ContentLane.EPISODIC,
            confidence=conf,
            sub_type=sub,
            hints={"has_date": bool(_DATE_RE.search(text))},
        )

    # ── 8. Default fallback ─────────────────────────────────────────
    # First-person => EPISODIC, otherwise ASSISTANT_CONTENT
    if has_first_person:
        return LaneResult(
            lane=ContentLane.EPISODIC,
            confidence=0.50,
            sub_type="default",
            hints={},
        )
    return LaneResult(
        lane=ContentLane.ASSISTANT_CONTENT,
        confidence=0.50,
        sub_type="default",
        hints={},
    )


def lane_config(lane: ContentLane) -> dict:
    """Return the extraction strategy config for *lane*.

    Raises ``KeyError`` if *lane* is not in ``LANE_CONFIGS``.
    """
    return LANE_CONFIGS[lane]
