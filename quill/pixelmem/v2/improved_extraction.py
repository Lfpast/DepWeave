"""Improved Extraction — targeted prompts for each content type.

The default extraction prompt misses:
  - Implicit preferences ("Sony camera" → user prefers Sony)
  - Assistant-generated details (shift schedules, story details)
  - Specific event dates ("met aunt on March 10" → user met_aunt 2026-03-10)
  - Numeric facts (costs, durations, counts)
  - Temporal relationships (bedtime before doctor appointment)

This module provides specialized extraction prompts per content lane,
plus a multi-pass extraction strategy that runs multiple targeted
prompts on the same text.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from pixelmem.triple_extractor import Triple


def ask_cli(prompt, model="haiku"):
    env = os.environ.copy()
    try:
        r = subprocess.run(
            ["claude", "-p", prompt[:5000], "--max-turns", "1", "--model", model],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).resolve().parent.parent.parent), env=env,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def parse_json_triples(raw: str) -> list[Triple]:
    """Parse JSON array of quadruples from LLM output."""
    try:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return [
                Triple(
                    str(d.get("s", "")),
                    str(d.get("r", "")),
                    str(d.get("o", "")),
                    str(d.get("c", "") or ""),
                )
                for d in data
                if d.get("s") and d.get("r") and d.get("o")
            ]
    except Exception:
        pass
    return []


# ── Specialized extraction prompts ──────────────────────────────

_CORE_RULES = (
    "Rules: lowercase entities, use underscores for spaces, "
    "map I/my/me/myself to 'user'. "
    "Convert ALL dates to absolute format YYYY-MM-DD (today=2026-04-08). "
    "Return ONLY a JSON array of objects with keys s, r, o, c."
)

PROMPT_GENERAL = (
    "Extract ALL knowledge as (subject, relation, object, condition) quadruples.\n"
    f"{_CORE_RULES}\n\n"
    "Text:\n{text}"
)

PROMPT_PREFERENCES = (
    "Extract the user's PREFERENCES, LIKES, DISLIKES, and EQUIPMENT/BRANDS mentioned.\n"
    "Look for:\n"
    "- Equipment owned: cameras, phones, cars, tools (brand + model)\n"
    "- Preferences: favorite foods, hotel styles, music, activities\n"
    "- Habits: routines, regular activities, frequencies\n"
    "- Dislikes: things avoided, disliked\n"
    "- Budget preferences: price ranges, spending habits\n\n"
    f"{_CORE_RULES}\n\n"
    "Text:\n{text}"
)

PROMPT_TEMPORAL = (
    "Extract ALL events, dates, times, and durations mentioned.\n"
    "CRITICAL: Convert every relative date to absolute YYYY-MM-DD.\n"
    "Look for:\n"
    "- Events with dates: 'attended sale on March 5' → date=2026-03-05\n"
    "- Durations: 'took 2 weeks', 'in 3 days', 'each way 45 minutes'\n"
    "- Times: 'went to bed at 2am', 'dinner at 7pm'\n"
    "- Frequencies: 'twice a week', 'every Sunday'\n"
    "- Start dates: 'started using Ibotta 3 weeks ago' → 2026-03-18\n"
    "- Sequences: events that happened before/after other events\n\n"
    f"{_CORE_RULES}\n\n"
    "Text:\n{text}"
)

PROMPT_ASSISTANT_CONTENT = (
    "Extract SPECIFIC DETAILS from the assistant's response.\n"
    "Look for:\n"
    "- Names: restaurant names, hotel names, app names, product names\n"
    "- Recommendations: specific items suggested with details\n"
    "- Generated content: story characters, descriptions, colors, attributes\n"
    "- Tables/schedules: shift assignments, rotation schedules, plans\n"
    "- Prices: costs, amounts, budgets mentioned\n"
    "- Locations: specific places, addresses, neighborhoods\n\n"
    f"{_CORE_RULES}\n\n"
    "Text:\n{text}"
)

PROMPT_NUMERIC = (
    "Extract ALL numeric facts, counts, amounts, and measurements.\n"
    "Look for:\n"
    "- Costs: '$300/night', '$30,000', 'pre-approved for $400K'\n"
    "- Counts: '4 Korean restaurants', '22 MCU movies', '3 social media breaks'\n"
    "- Durations: '2 weeks', '45 minutes each way', '4 hours'\n"
    "- Times: '2 AM', '7:30 PM', 'morning'\n"
    "- Distances/sizes: '5K run', '25 minutes'\n"
    "- Frequencies: 'three times a week', 'twice daily'\n\n"
    f"{_CORE_RULES}\n\n"
    "Text:\n{text}"
)


# ── Multi-pass extraction ───────────────────────────────────────

def extract_multipass(
    text: str,
    passes: list[str] = None,
    model: str = "haiku",
    max_workers: int = 4,
) -> list[Triple]:
    """Run multiple specialized extraction prompts in parallel.

    Each pass targets different types of knowledge. Results are
    merged and deduplicated.

    Args:
        text: Raw text to extract from.
        passes: List of pass names. Default: ["general", "preferences",
                "temporal", "numeric"]. Use "all" for all 5 passes.
        model: CLI model to use.
        max_workers: Parallel workers for CLI calls.
    """
    if passes is None:
        passes = ["general", "temporal", "numeric"]

    if "all" in passes:
        passes = ["general", "preferences", "temporal",
                  "assistant_content", "numeric"]

    prompt_map = {
        "general": PROMPT_GENERAL,
        "preferences": PROMPT_PREFERENCES,
        "temporal": PROMPT_TEMPORAL,
        "assistant_content": PROMPT_ASSISTANT_CONTENT,
        "numeric": PROMPT_NUMERIC,
    }

    def run_pass(pass_name):
        template = prompt_map.get(pass_name, PROMPT_GENERAL)
        prompt = template.format(text=text[:3000])
        raw = ask_cli(prompt, model)
        return parse_json_triples(raw)

    # Run passes in parallel
    all_triples = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_pass, p): p for p in passes}
        for f in as_completed(futures):
            try:
                all_triples.extend(f.result())
            except Exception:
                pass

    # Deduplicate by (s, r, o)
    seen = set()
    unique = []
    for t in all_triples:
        key = (t.subject.lower(), t.relation.lower(), t.object.lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique


def extract_conversation_multipass(
    qa_pairs: list[str],
    model: str = "haiku",
    max_chars_per_batch: int = 1500,
    max_workers: int = 6,
    passes: list[str] = None,
) -> list[Triple]:
    """Extract triples from conversation QA pairs using multi-pass.

    Batches QA pairs, then runs multi-pass extraction on each batch.
    All batches and passes run in parallel.

    Args:
        qa_pairs: List of "User: ...\nAssistant: ..." strings.
        model: CLI model.
        max_chars_per_batch: Max chars per extraction batch.
        max_workers: Parallel workers.
        passes: Extraction passes per batch.
    """
    if passes is None:
        passes = ["general", "temporal", "numeric"]

    # Build batches
    batches = []
    current, current_len = [], 0
    for pair in qa_pairs:
        text = pair[:max_chars_per_batch]
        if current_len + len(text) > max_chars_per_batch and current:
            batches.append("\n\n".join(current))
            current, current_len = [text], len(text)
        else:
            current.append(text)
            current_len += len(text)
    if current:
        batches.append("\n\n".join(current))

    # Run all batches × all passes in parallel
    all_triples = []

    def process_batch_pass(batch_text, pass_name):
        prompt_map = {
            "general": PROMPT_GENERAL,
            "preferences": PROMPT_PREFERENCES,
            "temporal": PROMPT_TEMPORAL,
            "assistant_content": PROMPT_ASSISTANT_CONTENT,
            "numeric": PROMPT_NUMERIC,
        }
        template = prompt_map.get(pass_name, PROMPT_GENERAL)
        prompt = template.format(text=batch_text[:3000])
        raw = ask_cli(prompt, model)
        return parse_json_triples(raw)

    tasks = [(b, p) for b in batches for p in passes]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_batch_pass, b, p) for b, p in tasks]
        for f in as_completed(futures):
            try:
                all_triples.extend(f.result())
            except Exception:
                pass

    # Deduplicate
    seen = set()
    unique = []
    for t in all_triples:
        key = (t.subject.lower(), t.relation.lower(), t.object.lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique


def select_passes_for_question(question: str) -> list[str]:
    """Auto-select extraction passes based on the question type.

    Ensures the right specialized extractor runs for each question.
    """
    q = question.lower()
    passes = ["general"]

    # Always add temporal for time-related questions
    if any(w in q for w in ["when", "ago", "weeks", "days", "months",
                             "how long", "how many week", "how many day",
                             "date", "time", "before", "after"]):
        passes.append("temporal")

    # Add numeric for count/amount questions
    if any(w in q for w in ["how many", "how much", "cost", "spent",
                             "price", "amount", "$", "total"]):
        passes.append("numeric")

    # Add preferences for recommendation/preference questions
    if any(w in q for w in ["prefer", "suggest", "recommend", "favorite",
                             "like", "complement", "accessories"]):
        passes.append("preferences")

    # Add assistant content for recall questions
    if any(w in q for w in ["remind", "previous chat", "conversation about",
                             "you told", "you said", "you recommended",
                             "shift rotation", "children's book", "story"]):
        passes.append("assistant_content")

    # Ensure at least temporal and numeric for robustness
    if "temporal" not in passes:
        passes.append("temporal")
    if "numeric" not in passes:
        passes.append("numeric")

    return list(dict.fromkeys(passes))  # deduplicate preserving order
