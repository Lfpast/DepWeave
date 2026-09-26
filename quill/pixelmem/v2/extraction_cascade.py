"""Cheap Extraction Cascade — regex/NER first, LLM only if needed.

Stage A: Fast extractor (existing _extract_triples_fast)
  - NER patterns, date extraction, preference detection
  - Handles ~70% of user conversation content

Stage B: Confidence gate
  - If extraction is high-quality and high-count, write directly
  - Score based on entity/relation pattern quality

Stage C: LLM fallback
  - Only for complex sentences, novel relations, rich assistant content
  - Uses existing extract_triples(text, use_cli=True)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from pixelmem.triple_extractor import Triple, extract_triples


_FILLER_WORDS = re.compile(
    r'\b(just|recently|actually|basically|really|also|even|still|already|finally)\b',
    re.I,
)


def _strip_fillers(text: str) -> str:
    """Strip filler words that break extraction regex patterns.

    "I just bought a car" → "I bought a car"
    "I recently moved to NYC" → "I moved to NYC"
    """
    return _FILLER_WORDS.sub('', text).replace('  ', ' ')


@dataclass
class ExtractionResult:
    """Result from the extraction cascade."""
    triples: list[Triple]
    method: str  # "fast", "llm", "skip"
    confidence: float
    latency_ms: float
    source_type: str = "cheap_extractor"  # or "llm"

    @property
    def needs_review(self) -> bool:
        return self.confidence < 0.5

    def to_dict(self) -> dict:
        return {
            "n_triples": len(self.triples),
            "method": self.method,
            "confidence": round(self.confidence, 2),
            "latency_ms": round(self.latency_ms, 1),
            "source_type": self.source_type,
            "needs_review": self.needs_review,
        }


class ExtractionCascade:
    """Three-stage extraction: fast → confidence gate → LLM fallback."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_triples: int = 1,
        skip_llm: bool = False,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_triples = min_triples
        self.skip_llm = skip_llm
        # Stats
        self._stats = {"fast": 0, "llm": 0, "skip": 0}

    def extract(
        self,
        text: str,
        content_type: str = "general",
    ) -> ExtractionResult:
        """Run the extraction cascade.

        1. Try fast extractor
        2. Score confidence
        3. If below threshold and content_type allows, try LLM
        """
        # Skip certain content types entirely
        if content_type == "chit_chat":
            self._stats["skip"] += 1
            return ExtractionResult(
                triples=[], method="skip", confidence=1.0,
                latency_ms=0, source_type="skip",
            )

        # Stage A: Fast extraction
        # Pre-process: strip filler words that break regex patterns
        # "I just bought" → "I bought", "I recently moved" → "I moved"
        cleaned = _strip_fillers(text)
        t0 = time.perf_counter()
        fast_triples = extract_triples(cleaned, fast=True)
        # If cleaning helped, also try original in case cleaning removed signal
        if not fast_triples:
            fast_triples = extract_triples(text, fast=True)
        fast_latency = (time.perf_counter() - t0) * 1000

        # Stage B: Confidence scoring
        confidence = self._score_confidence(fast_triples, text)

        # Gate: if good enough, use fast result
        if (
            len(fast_triples) >= self.min_triples
            and confidence >= self.confidence_threshold
        ):
            self._stats["fast"] += 1
            return ExtractionResult(
                triples=fast_triples,
                method="fast",
                confidence=confidence,
                latency_ms=fast_latency,
                source_type="cheap_extractor",
            )

        # Stage C: LLM fallback (only if allowed)
        if self.skip_llm or content_type == "assistant_light":
            # Return fast results even if low confidence
            self._stats["fast"] += 1
            return ExtractionResult(
                triples=fast_triples,
                method="fast",
                confidence=confidence,
                latency_ms=fast_latency,
                source_type="cheap_extractor",
            )

        t1 = time.perf_counter()
        try:
            llm_triples = extract_triples(text, use_cli=True)
        except Exception:
            llm_triples = fast_triples
        llm_latency = (time.perf_counter() - t1) * 1000

        # Use whichever produced more triples
        if len(llm_triples) > len(fast_triples):
            self._stats["llm"] += 1
            return ExtractionResult(
                triples=llm_triples,
                method="llm",
                confidence=0.85,
                latency_ms=fast_latency + llm_latency,
                source_type="llm",
            )
        else:
            self._stats["fast"] += 1
            return ExtractionResult(
                triples=fast_triples,
                method="fast",
                confidence=confidence,
                latency_ms=fast_latency + llm_latency,
                source_type="cheap_extractor",
            )

    def _score_confidence(self, triples: list[Triple], text: str) -> float:
        """Heuristic confidence scoring for extracted triples."""
        if not triples:
            # No triples from non-empty text → low confidence
            return 0.1 if len(text.strip()) > 20 else 0.8

        score = 0.5  # base

        # More triples from longer text → higher confidence
        text_len = len(text.strip())
        expected_triples = max(1, text_len // 100)
        ratio = len(triples) / expected_triples
        if ratio >= 0.5:
            score += 0.2

        # Check triple quality
        good_triples = 0
        for t in triples:
            # Good: has subject, relation, object of reasonable length
            if len(t.subject) > 1 and len(t.relation) > 1 and len(t.object) > 1:
                good_triples += 1
            # Bonus: has condition
            if t.condition:
                good_triples += 0.5

        quality = good_triples / max(1, len(triples))
        score += quality * 0.3

        return min(1.0, score)

    @property
    def stats(self) -> dict:
        total = sum(self._stats.values())
        return {
            **self._stats,
            "total": total,
            "fast_pct": round(self._stats["fast"] / max(1, total) * 100, 1),
            "llm_pct": round(self._stats["llm"] / max(1, total) * 100, 1),
            "skip_pct": round(self._stats["skip"] / max(1, total) * 100, 1),
        }
