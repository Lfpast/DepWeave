"""Content-Type Lanes — classify incoming text before extraction.

Routes incoming content to different extraction strategies:
  - EPISODIC: user facts, events, purchases → fast extraction + timestamp
  - PREFERENCE: user preferences, favorites → fast extraction, no timestamp
  - ASSISTANT_CONTENT: assistant-generated stories, recommendations → LLM extraction
  - CHIT_CHAT: greetings, small talk → skip extraction entirely
  - UPDATE: corrections, status changes → fast extraction + conflict check
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ContentLane(Enum):
    EPISODIC = "episodic"
    PREFERENCE = "preference"
    ASSISTANT_CONTENT = "assistant_content"
    CHIT_CHAT = "chit_chat"
    UPDATE = "update"


@dataclass
class LaneClassification:
    lane: ContentLane
    confidence: float
    hints: dict  # e.g. {"temporal": True, "first_person": True}

    def to_dict(self) -> dict:
        return {
            "lane": self.lane.value,
            "confidence": round(self.confidence, 2),
            "hints": self.hints,
        }


# Lane → extraction/storage config
LANE_CONFIGS = {
    ContentLane.EPISODIC: {
        "extraction": "fast",
        "add_timestamp": True,
        "priority": "high",
        "shard_topic": "events",
    },
    ContentLane.PREFERENCE: {
        "extraction": "fast",
        "add_timestamp": False,
        "priority": "high",
        "shard_topic": "preferences",
    },
    ContentLane.ASSISTANT_CONTENT: {
        "extraction": "llm",
        "add_timestamp": True,
        "priority": "medium",
        "shard_topic": "assistant",
    },
    ContentLane.CHIT_CHAT: {
        "extraction": "skip",
        "add_timestamp": False,
        "priority": "low",
        "shard_topic": None,
    },
    ContentLane.UPDATE: {
        "extraction": "fast",
        "add_timestamp": True,
        "priority": "high",
        "shard_topic": "updates",
    },
}


# ── Patterns ────────────────────────────────────────────────────

_PREFERENCE_PATTERNS = [
    re.compile(r'\bI (?:prefer|like|love|enjoy|always|usually|favorite)\b', re.I),
    re.compile(r'\bmy (?:favorite|preferred|go-to|usual)\b', re.I),
    re.compile(r'\bI (?:don\'t like|hate|dislike|avoid|never)\b', re.I),
    re.compile(r'\bI\'d (?:rather|prefer)\b', re.I),
]

_ASSISTANT_PATTERNS = [
    re.compile(r'^\[?assistant\]?:', re.I),
    re.compile(r'\blet me (?:help|suggest|recommend|write|create)\b', re.I),
    re.compile(r'\bhere(?:\'s| is| are)\b.*:', re.I),
    re.compile(r'\bI(?:\'d| would) recommend\b', re.I),
    re.compile(r'\b(?:once upon|chapter \d|the end)\b', re.I),  # story content
]

_CHIT_CHAT_PATTERNS = [
    re.compile(r'^(?:hi|hello|hey|thanks|thank you|bye|goodbye|ok|okay|sure|yes|no|haha|lol)\s*[.!?,]*$', re.I),
    re.compile(r'\b(?:how are you|what\'s up|good morning|good night)\b', re.I),
    re.compile(r'^(?:sounds good|great|awesome|cool|nice|perfect|got it)\s*[.!?]*$', re.I),
    re.compile(r'^(?:hi|hello|hey)\b[^.]*(?:how are you|what\'s up)', re.I),
]

_UPDATE_PATTERNS = [
    re.compile(r'\b(?:actually|correction|update|changed|moved|switched|no longer)\b', re.I),
    re.compile(r'\bI (?:now|recently|just) (?:moved|changed|switched|started|got)\b', re.I),
    re.compile(r'\bnot anymore\b|\binstead\b|\brather than\b', re.I),
]

_EPISODIC_PATTERNS = [
    re.compile(r'\bI (?:went|visited|bought|attended|watched|tried|received|took)\b', re.I),
    re.compile(r'\b(?:yesterday|last (?:week|month|night)|today|this (?:week|morning))\b', re.I),
    re.compile(r'\b(?:\d{4}-\d{2}-\d{2}|on (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun))', re.I),
    re.compile(r'\bmy (?:commute|job|apartment|car|doctor|appointment)\b', re.I),
]


def classify_content(text: str) -> LaneClassification:
    """Classify text into a content lane."""
    text_stripped = text.strip()

    # Short messages → likely chit-chat
    if len(text_stripped) < 20:
        for p in _CHIT_CHAT_PATTERNS:
            if p.search(text_stripped):
                return LaneClassification(
                    ContentLane.CHIT_CHAT, 0.9,
                    {"short": True},
                )

    # Check patterns in priority order
    hints = {
        "first_person": bool(re.search(r'\bI\b|\bmy\b|\bme\b', text, re.I)),
        "temporal": bool(re.search(r'\b(?:yesterday|last|today|ago|\d{4}-\d{2})\b', text, re.I)),
        "has_numbers": bool(re.search(r'\$?\d+', text)),
    }

    # Chit-chat checked FIRST — before episodic so "how are you" isn't
    # misclassified as first-person episodic content
    for p in _CHIT_CHAT_PATTERNS:
        if p.search(text_stripped):
            return LaneClassification(ContentLane.CHIT_CHAT, 0.85, hints)

    # Update patterns (corrections override other types)
    for p in _UPDATE_PATTERNS:
        if p.search(text):
            return LaneClassification(ContentLane.UPDATE, 0.8, hints)

    # Preference patterns
    pref_score = sum(1 for p in _PREFERENCE_PATTERNS if p.search(text))
    if pref_score >= 1:
        return LaneClassification(ContentLane.PREFERENCE, min(0.9, 0.7 + pref_score * 0.1), hints)

    # Assistant content
    for p in _ASSISTANT_PATTERNS:
        if p.search(text):
            return LaneClassification(ContentLane.ASSISTANT_CONTENT, 0.8, hints)

    # Episodic (first-person factual content)
    for p in _EPISODIC_PATTERNS:
        if p.search(text):
            return LaneClassification(ContentLane.EPISODIC, 0.8, hints)

    # Default: episodic if first-person, assistant_content otherwise
    if hints["first_person"]:
        return LaneClassification(ContentLane.EPISODIC, 0.5, hints)
    else:
        return LaneClassification(ContentLane.ASSISTANT_CONTENT, 0.4, hints)


def lane_config(lane: ContentLane) -> dict:
    """Get extraction and storage config for a lane."""
    return LANE_CONFIGS[lane]
