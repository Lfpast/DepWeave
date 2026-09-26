"""Triple extraction from raw text.

Provides both a rule-based extractor and an optional caller-based extractor.
The caller is supplied by the application, so this module has no model backend.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Triple:
    """A knowledge graph triple with optional condition metadata."""
    subject: str
    relation: str
    object: str
    condition: str = ""

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.subject, self.relation, self.object, self.condition)

    def __repr__(self) -> str:
        cond = f" [{self.condition}]" if self.condition else ""
        return f"({self.subject} --{self.relation}--> {self.object}{cond})"


def extract_triples(text: str, use_llm: bool = False, use_cli: bool = False, fast: bool = False, **llm_kwargs) -> list[Triple]:
    """Extract (subject, relation, object, condition) triples from text.

    Args:
        text: Raw text input.
        use_llm: If True, use the supplied local LLM caller.
        use_cli: Legacy alias for use_llm; also uses the supplied caller.
        fast: If True, use enhanced rule-based extraction (broader patterns,
              handles first-person, faster than CLI but less accurate).
        **llm_kwargs: Passed to the LLM extractor.

    Returns:
        List of extracted triples.
    """
    if use_llm or use_cli:
        return _extract_triples_llm(text, **llm_kwargs)
    if fast:
        return _extract_triples_fast(text)
    return _extract_triples_rule_based(text)


def _extract_triples_rule_based(text: str) -> list[Triple]:
    """Simple rule-based triple extraction using pattern matching.

    Handles common patterns:
      - "X is Y" / "X is a Y"
      - "X works at Y" / "X lives in Y" (verb + preposition)
      - "X <verb>s Y"
      - "X, who is Y, ..."
    """
    triples: list[Triple] = []
    sentences = re.split(r'[.!?;]+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        triples.extend(_parse_sentence(sentence))

    return triples


def _parse_sentence(sentence: str) -> list[Triple]:
    """Extract triples from a single sentence."""
    triples = []

    # Pattern: "X is/are/was/were [a/an/the] Y"
    m = re.match(
        r'^(.+?)\s+(?:is|are|was|were)\s+(?:a|an|the\s+)?(.+?)$',
        sentence, re.IGNORECASE
    )
    if m:
        subj, obj = m.group(1).strip(), m.group(2).strip()
        if _valid_entity(subj) and _valid_entity(obj):
            triples.append(Triple(subj, "is", obj))
            return triples

    # Pattern: "X <verb>s at/in/for/with/on/from/to Y"
    m = re.match(
        r'^(.+?)\s+(\w+(?:s|ed|es|ing)?)\s+(at|in|for|with|on|from|to)\s+(.+?)$',
        sentence, re.IGNORECASE
    )
    if m:
        subj = m.group(1).strip()
        verb = m.group(2).strip()
        prep = m.group(3).strip()
        obj = m.group(4).strip()
        if _valid_entity(subj) and _valid_entity(obj):
            relation = f"{verb}_{prep}"
            triples.append(Triple(subj, relation, obj))
            return triples

    # Pattern: "X <verb>s Y" (simple SVO)
    m = re.match(
        r'^(.+?)\s+(\w+(?:s|ed|es))\s+(.+?)$',
        sentence, re.IGNORECASE
    )
    if m:
        subj = m.group(1).strip()
        verb = m.group(2).strip()
        obj = m.group(3).strip()
        if _valid_entity(subj) and _valid_entity(obj):
            triples.append(Triple(subj, verb, obj))
            return triples

    # Pattern: "X and Y are Z" (compound subject)
    m = re.match(
        r'^(.+?)\s+and\s+(.+?)\s+(?:are|were)\s+(.+?)$',
        sentence, re.IGNORECASE
    )
    if m:
        subj1 = m.group(1).strip()
        subj2 = m.group(2).strip()
        obj = m.group(3).strip()
        if _valid_entity(subj1) and _valid_entity(subj2) and _valid_entity(obj):
            triples.append(Triple(subj1, "is", obj))
            triples.append(Triple(subj2, "is", obj))
            return triples

    return triples


def _valid_entity(text: str) -> bool:
    """Basic check that a string is plausibly an entity (not empty, not too long)."""
    text = text.strip()
    return bool(text) and len(text) < 200 and len(text.split()) < 15


def _extract_triples_llm(text: str, *, llm: Callable[[str], tuple[str, int, int]]) -> list[Triple]:
    """Extract triples through the application's local LLM caller."""

    prompt = f"""Extract knowledge graph triples from the following text.
Return a JSON array where each element has keys: "subject", "relation", "object", "condition".
- "subject" and "object" are entity names (canonicalize: lowercase, remove articles).
- "relation" is the relationship type (e.g., "works_at", "is_a", "located_in").
- "condition" is optional metadata (temporal scope, source, confidence). Use "" if none.

Be aggressive about entity resolution: "Alice", "alice", "Alice Smith" should all be "alice".

Text:
{text}

Return ONLY the JSON array, no other text."""

    content = llm(prompt)[0].strip()
    # Extract JSON from response
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\n?', '', content)
        content = re.sub(r'\n?```$', '', content)

    data = json.loads(content)
    return [
        Triple(
            subject=item["subject"],
            relation=item["relation"],
            object=item["object"],
            condition=item.get("condition", ""),
        )
        for item in data
    ]


def _extract_condition(text: str) -> tuple[str, str]:
    """Extract condition/context from a text fragment.

    Returns (clean_text, condition) where condition captures:
    - Temporal: "in two weeks", "last month", "on Sunday", "since 2022"
    - Quantity: "about 45 minutes", "3.5 weeks"
    - Parenthetical: "(since 2020)", "(on Sundays)"
    - Trailing clauses: "each way", "per day"
    """
    condition = ""
    clean = text.strip()

    # Extract parenthetical conditions: "X (since 2020)"
    m = re.search(r'\(([^)]+)\)\s*$', clean)
    if m:
        condition = m.group(1).strip()
        clean = clean[:m.start()].strip()
        return clean, condition

    # Extract temporal phrases at end: "in two weeks", "in a week and a half", "last month"
    temporal = re.search(
        r'\s+(in (?:about |approximately )?(?:\w+ )?(?:[\d.]+\s+)?(?:a )?(?:week|month|day|hour|minute|year)(?:s)?(?:\s+and\s+a\s+half)?)'
        r'|(\s+last (?:week|month|year|night|time))'
        r'|(\s+on (?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)\w*)'
        r'|(\s+since (?:20\d\d|\w+))'
        r'|(\s+(?:each|per|every) (?:way|day|week|month|year|shift|rotation))'
        r'|(\s+(?:about|approximately|around) [\d.]+ (?:minutes?|hours?|weeks?|days?)(?:\s+\w+)?)',
        clean, re.IGNORECASE
    )
    if temporal:
        matched = temporal.group(0).strip()
        condition = matched
        clean = clean[:temporal.start()].strip()
        return clean, condition

    # Extract trailing quantity/duration: "45 minutes each way"
    m = re.search(r'([\d.]+\s+(?:minutes?|hours?|weeks?|days?|months?)(?:\s+\w+)?)\s*$', clean)
    if m and m.start() > 5:
        # Only extract if there's still meaningful text before
        before = clean[:m.start()].strip()
        if len(before) > 3:
            condition = m.group(1).strip()
            clean = before
            return clean, condition

    return clean, condition


def _extract_triples_fast(text: str) -> list[Triple]:
    """Enhanced rule-based extraction producing quadruples (s, r, o, c).

    Extracts conditions from temporal phrases, quantities, parentheticals.

    Examples:
      "I watched 22 MCU movies in two weeks"
        → (user, watched, 22 MCU movies, in two weeks)
      "My commute takes about 45 minutes each way"
        → (user, commute, 45 minutes, each way)
      "Admon works the 8am-4pm shift on Sunday"
        → (admon, shift, 8am-4pm, on Sunday)
      "I graduated with a degree in Business Administration"
        → (user, graduated_with, Business Administration, )
    """
    triples: list[Triple] = []
    sentences = re.split(r'[.!?\n]+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 5:
            continue

        # Strip prefixes
        sentence = re.sub(r'^\[(?:user|assistant)\]:\s*', '', sentence)
        sentence = re.sub(r'^[-*\d]+[.)]\s*', '', sentence)  # bullet points
        sentence = sentence.strip()
        if len(sentence) < 5:
            continue

        matched = False

        # === First-person patterns → subject = "user" ===

        # "I <verb> <prep> X [condition]"
        m = re.match(
            r'^I\s+(\w+)\s+(at|in|for|with|on|from|to)\s+(.+?)$',
            sentence, re.IGNORECASE
        )
        if m:
            obj, cond = _extract_condition(m.group(3).strip())
            if _valid_entity(obj):
                triples.append(Triple("user", f"{m.group(1)}_{m.group(2)}", obj, cond))
                matched = True

        # "I'm a X" / "I am a X"
        if not matched:
            m = re.match(r"^I(?:'m| am)\s+(?:a |an )?(.+?)$", sentence, re.IGNORECASE)
            if m:
                obj, cond = _extract_condition(m.group(1).strip())
                if _valid_entity(obj):
                    triples.append(Triple("user", "is", obj, cond))
                    matched = True

        # "I have/I've X"
        if not matched:
            m = re.match(r"^I(?:'ve| have)\s+(?:been\s+)?(.+?)$", sentence, re.IGNORECASE)
            if m:
                obj, cond = _extract_condition(m.group(1).strip())
                if _valid_entity(obj):
                    triples.append(Triple("user", "has", obj, cond))
                    matched = True

        # "I prefer/like/want X"
        if not matched:
            m = re.match(
                r'^I\s+(prefer|like|want|need|enjoy|love|hate|dislike)\s+(.+?)$',
                sentence, re.IGNORECASE
            )
            if m:
                obj, cond = _extract_condition(m.group(2).strip())
                if _valid_entity(obj):
                    triples.append(Triple("user", m.group(1) + "s", obj, cond))
                    matched = True

        # "I graduated/studied/majored with/in X"
        if not matched:
            m = re.match(
                r'^I\s+(graduated|studied|majored)\s+(?:with |in |from )?(.+?)$',
                sentence, re.IGNORECASE
            )
            if m:
                obj, cond = _extract_condition(m.group(2).strip())
                if _valid_entity(obj):
                    triples.append(Triple("user", m.group(1) + "_with", obj, cond))
                    matched = True

        # "I watched/tried/visited/went to X"
        if not matched:
            m = re.match(
                r'^I\s+(watched|went|visited|tried|attended|bought|got|received|took|started|completed|finished)\s+(?:to |at |all )?(.+?)$',
                sentence, re.IGNORECASE
            )
            if m:
                obj, cond = _extract_condition(m.group(2).strip())
                if _valid_entity(obj):
                    triples.append(Triple("user", m.group(1), obj, cond))
                    matched = True

        # "I'm/I've been <verb>ing X"
        if not matched:
            m = re.match(r"^I(?:'m|'ve been)\s+(\w+ing)\s+(.+?)$", sentence, re.IGNORECASE)
            if m:
                obj, cond = _extract_condition(m.group(2).strip())
                if _valid_entity(obj):
                    triples.append(Triple("user", m.group(1), obj, cond))
                    matched = True

        if matched:
            continue

        # === "My X is/takes/costs Y" ===
        m = re.match(r'^[Mm]y\s+(.+?)\s+(?:is|are|was|were|takes?|costs?|has|lasts?)\s+(.+?)$', sentence)
        if m:
            attr = m.group(1).strip().replace(" ", "_")
            val, cond = _extract_condition(m.group(2).strip())
            if _valid_entity(val):
                triples.append(Triple("user", attr, val, cond))
                continue

        # === "The user X" (from rephrased text) ===
        m = re.match(r'^[Tt]he user\s+(\w+(?:s|ed|es)?)\s+(.+?)$', sentence)
        if m:
            verb = m.group(1).strip()
            rest = m.group(2).strip()
            # Check for preposition
            m2 = re.match(r'^(at|in|for|with|on|from|to)\s+(.+)$', rest)
            if m2:
                obj, cond = _extract_condition(m2.group(2).strip())
                if _valid_entity(obj):
                    triples.append(Triple("user", f"{verb}_{m2.group(1)}", obj, cond))
                    continue
            obj, cond = _extract_condition(rest)
            if _valid_entity(obj):
                triples.append(Triple("user", verb, obj, cond))
                continue

        # === "X <verb> Y [condition]" (general SVO) ===
        m = re.match(r'^(.+?)\s+(\w+(?:s|ed|es))\s+(.+?)$', sentence, re.IGNORECASE)
        if m:
            subj = m.group(1).strip()
            verb = m.group(2).strip()
            rest = m.group(3).strip()
            obj, cond = _extract_condition(rest)
            if _valid_entity(subj) and _valid_entity(obj) and len(subj.split()) < 6:
                triples.append(Triple(subj, verb, obj, cond))
                continue

        # === "X is/are Y" ===
        m = re.match(r'^(.+?)\s+(?:is|are|was|were)\s+(?:a |an |the )?(.+?)$', sentence, re.IGNORECASE)
        if m:
            subj = m.group(1).strip()
            rest = m.group(2).strip()
            obj, cond = _extract_condition(rest)
            if _valid_entity(subj) and _valid_entity(obj) and len(subj.split()) < 6:
                triples.append(Triple(subj, "is", obj, cond))

    return triples


def extract_triples_parallel_cli(
    texts: list[str],
    *,
    llm: Callable[[str], tuple[str, int, int]],
    max_chars_per_batch: int = 4000,
    max_workers: int = 4,
) -> list[Triple]:
    """Extract text batches through an application-provided local caller.

    Args:
        texts: List of text chunks to extract from.
        llm: Local LLM caller.
        max_chars_per_batch: Max chars per batch before splitting.
        max_workers: Number of concurrent calls.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Split texts into batches — keep more context per batch for better extraction
    batches = []
    current_batch = []
    current_len = 0
    for text in texts:
        truncated = text[:2000]  # keep more context per Q+A pair
        current_batch.append(truncated)
        current_len += len(truncated)
        if current_len > max_chars_per_batch:
            batches.append("\n---\n".join(current_batch))
            current_batch = []
            current_len = 0
    if current_batch:
        batches.append("\n---\n".join(current_batch))

    if not batches:
        return []

    # Run all batches in parallel
    def _extract_batch(batch_text):
        return _extract_triples_llm(batch_text, llm=llm)

    all_triples = []
    workers = min(max_workers, len(batches))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_extract_batch, b) for b in batches]
        for f in as_completed(futures):
            all_triples.extend(f.result())

    return all_triples


def triples_from_tuples(
    tuples: list[tuple[str, str, str]] | list[tuple[str, str, str, str]],
) -> list[Triple]:
    """Convenience: convert raw tuples to Triple objects."""
    result = []
    for t in tuples:
        if len(t) == 3:
            result.append(Triple(t[0], t[1], t[2]))
        elif len(t) == 4:
            result.append(Triple(t[0], t[1], t[2], t[3]))
        else:
            raise ValueError(f"Expected 3 or 4 element tuple, got {len(t)}")
    return result
