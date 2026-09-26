"""Triple extraction from raw text.

Provides both a rule-based extractor and an LLM-based extractor.
The rule-based version uses simple NLP heuristics; the LLM version
calls an API for higher-quality extraction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional


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
        use_llm: If True, use LLM-based extraction (requires Anthropic API key).
        use_cli: If True, use Claude CLI for extraction (no API key needed, but slow).
        fast: If True, use enhanced rule-based extraction (broader patterns,
              handles first-person, faster than CLI but less accurate).
        **llm_kwargs: Passed to the LLM extractor.

    Returns:
        List of extracted triples.
    """
    if use_cli:
        return _extract_triples_cli(text, **llm_kwargs)
    if use_llm:
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


def _extract_triples_llm(text: str, **kwargs) -> list[Triple]:
    """LLM-based triple extraction using the Anthropic API.

    Expects ANTHROPIC_API_KEY in the environment or passed via kwargs.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "LLM extraction requires the 'anthropic' package. "
            "Install with: pip install anthropic"
        )

    api_key = kwargs.get("api_key") or None
    model = kwargs.get("model", "claude-sonnet-4-20250514")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Extract knowledge graph triples from the following text.
Return a JSON array where each element has keys: "subject", "relation", "object", "condition".
- "subject" and "object" are entity names (canonicalize: lowercase, remove articles).
- "relation" is the relationship type (e.g., "works_at", "is_a", "located_in").
- "condition" is optional metadata (temporal scope, source, confidence). Use "" if none.

Be aggressive about entity resolution: "Alice", "alice", "Alice Smith" should all be "alice".

Text:
{text}

Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.content[0].text.strip()
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
    model: str = "haiku",
    max_chars_per_batch: int = 4000,
    max_workers: int = 4,
) -> list[Triple]:
    """Parallel batch extraction — split texts into batches, run CLI calls concurrently.

    Like sub-agents working together: each batch runs in a separate process,
    all in parallel. 4 batches × 10s each = 10s total instead of 40s.

    Args:
        texts: List of text chunks to extract from.
        model: CLI model to use.
        max_chars_per_batch: Max chars per batch before splitting.
        max_workers: Number of parallel CLI calls.
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
        return _extract_triples_cli(batch_text, model=model)

    all_triples = []
    workers = min(max_workers, len(batches))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_extract_batch, b) for b in batches]
        for f in as_completed(futures):
            all_triples.extend(f.result())

    return all_triples


def _extract_triples_cli(text: str, **kwargs) -> list[Triple]:
    """LLM-based triple extraction using the Claude CLI.

    No API key needed — uses the same Claude CLI as all other methods.
    """
    import subprocess
    import sys
    import os
    from pathlib import Path

    model = kwargs.get("model", "haiku")

    prompt = f"""Extract EVERY specific detail from this conversation as structured quadruples.

Return a JSON array. Each element: {{"subject", "relation", "object", "condition"}}.

CRITICAL RULES:
- subject = "user" for first-person (I/my/me)
- Preserve ALL specific names, numbers, colors, locations, times, prices — NEVER drop details
- If a fact is UPDATED later in the conversation, extract ONLY the latest value with condition="updated"
- For assistant-generated content (lists, descriptions, schedules), extract each specific item

Extract EVERYTHING:
1. SPECIFIC DETAILS: names of restaurants/places/people, exact numbers/times/prices, colors, descriptions
   - "The Plesiosaur had a blue scaly body" → {{"subject":"plesiosaur","relation":"appearance","object":"blue scaly body","condition":""}}
   - "I redeemed a $5 coupon at Target" → {{"subject":"user","relation":"redeemed_coupon","object":"$5 coffee creamer coupon","condition":"at target"}}
   - "Sugar Factory at Icon Park" → {{"subject":"sugar factory","relation":"located_at","object":"icon park","condition":""}}
2. KNOWLEDGE UPDATES: if old value X is corrected to Y, extract ONLY Y
   - "My time was 27:12" then later "I improved to 25:50" → extract 25:50 only, condition="personal best, updated"
3. COUNTS: "I tried 4 Korean restaurants" → {{"subject":"user","relation":"tried_count","object":"4","condition":"korean restaurants in city"}}
4. PREFERENCES (implicit): user's equipment/interests imply preferences
5. TEMPORAL: convert relative dates to absolute (today=2026-04-08)
6. ASSISTANT CONTENT: specific recommendations, schedules, descriptions the assistant provided
   - Each shift assignment, each restaurant name, each item in a list

Text:
{text[:5000]}

Return ONLY the JSON array."""

    env = os.environ.copy()
    if sys.platform == "win32" and "CLAUDE_CODE_GIT_BASH_PATH" not in env:
        for candidate in [
            r"D:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\bin\bash.exe",
        ]:
            if os.path.exists(candidate):
                env["CLAUDE_CODE_GIT_BASH_PATH"] = candidate
                break

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--max-turns", "1", "--model", model],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
        )
        content = result.stdout.strip()
    except Exception as e:
        print(f"CLI extraction failed: {e}", file=sys.stderr)
        return _extract_triples_rule_based(text)

    if not content:
        return _extract_triples_rule_based(text)

    # Extract JSON from response
    if "```" in content:
        content = re.sub(r'^.*?```(?:json)?\n?', '', content, flags=re.DOTALL)
        content = re.sub(r'\n?```.*$', '', content, flags=re.DOTALL)

    # Find the JSON array
    bracket_start = content.find("[")
    bracket_end = content.rfind("]")
    if bracket_start == -1 or bracket_end == -1:
        return _extract_triples_rule_based(text)

    content = content[bracket_start:bracket_end + 1]

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return _extract_triples_rule_based(text)

    return [
        Triple(
            subject=str(item.get("subject", "")).strip(),
            relation=str(item.get("relation", "")).strip(),
            object=str(item.get("object", "")).strip(),
            condition=str(item.get("condition", "")).strip(),
        )
        for item in data
        if item.get("subject") and item.get("relation") and item.get("object")
    ]


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
