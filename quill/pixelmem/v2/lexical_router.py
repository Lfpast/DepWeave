"""Lightweight Lexical Router — BM25-style routing without embeddings.

Routes queries to relevant index records using:
  - Exact entity name matching
  - Normalized token BM25 scoring
  - Pronoun resolution (I/my/me → user)
  - Relation keyword matching
  - Date/temporal term detection

Returns ranked candidates with confidence scores.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from pixelmem.v2.compact_index import CompactIndex, IndexRecord


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "about", "after",
    "all", "also", "and", "any", "but", "by", "for", "from", "get",
    "how", "if", "in", "into", "it", "its", "just", "me", "my", "no",
    "not", "of", "on", "or", "our", "out", "so", "some", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this",
    "to", "up", "very", "was", "we", "what", "when", "where", "which",
    "who", "why", "with", "you", "your",
})

_PRONOUNS = frozenset({"i", "my", "me", "mine", "myself", "we", "our"})

_TEMPORAL_PATTERNS = re.compile(
    r'\b(when|ago|since|before|after|during|until|date|week|month|year|day'
    r'|yesterday|today|tomorrow|last|recent|how\s+long|how\s+many\s+(?:week|day|month|year))'
    r'\b', re.IGNORECASE
)

_QUERY_RELATION_HINTS = {
    "work": ["works_at", "employed_by", "ceo_of", "role"],
    "job": ["works_at", "employed_by", "role"],
    "live": ["lives_in", "located_in"],
    "born": ["born_in"],
    "manage": ["manages", "reports_to"],
    "skill": ["has_skill", "certified_in"],
    "study": ["studied_at", "degree"],
    "found": ["founded_by"],
    "partner": ["partner_of"],
    "know": ["knows", "collaborates_with"],
    "prefer": ["prefers", "favorite", "likes"],
    "commute": ["commute", "daily_commute"],
    # Workflow memory hints
    "import": ["imports", "depends_on"],
    "imports": ["imports", "depends_on"],
    "call": ["calls"],
    "calls": ["calls"],
    "contain": ["contains_function", "contains_class", "contains_method", "contains_file"],
    "contains": ["contains_function", "contains_class", "contains_method", "contains_file"],
    "function": ["contains_function", "calls", "parameters"],
    "class": ["contains_class", "extends"],
    "file": ["contains_file", "type"],
    "config": ["configures", "defines_constant"],
    "constant": ["defines_constant"],
    "test": ["tests"],
    "export": ["exports"],
    "depend": ["depends_on", "imports"],
    "decorator": ["decorated_by"],
    "extend": ["extends"],
    "handle": ["contains_function", "contains_class"],
    "define": ["defines_constant", "contains_function"],
}


@dataclass
class RouteResult:
    """Result from the lexical router."""
    entities: list[tuple[str, float]] = field(default_factory=list)  # (entity, score)
    records: list[tuple[IndexRecord, float]] = field(default_factory=list)  # (record, score)
    themes: list[tuple[str, float]] = field(default_factory=list)  # (theme, score)
    query_mode_hint: str = "general"  # hint for query classifier
    has_temporal: bool = False
    has_pronoun: bool = False
    confidence: float = 0.0

    def top_entities(self, k: int = 8) -> list[str]:
        return [e for e, _ in sorted(self.entities, key=lambda x: -x[1])[:k]]

    def top_records(self, k: int = 4) -> list[IndexRecord]:
        return [r for r, _ in sorted(self.records, key=lambda x: -x[1])[:k]]

    def top_themes(self, k: int = 3) -> list[str]:
        return [t for t, _ in sorted(self.themes, key=lambda x: -x[1])[:k]]


class LexicalRouter:
    """BM25-based query router over a CompactIndex."""

    def __init__(self, index: CompactIndex, k1: float = 1.2, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        # Build IDF table from entity/relation terms across records
        self._idf: dict[str, float] = {}
        self._avg_doc_len: float = 0
        self._doc_lengths: list[int] = []
        self._doc_terms: list[set[str]] = []
        self._build_idf()

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, split, strip stopwords."""
        tokens = re.findall(r'\w+', text.lower())
        return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    def _build_idf(self) -> None:
        """Build inverse document frequency table."""
        n_docs = len(self.index.records)
        if n_docs == 0:
            return

        doc_freq: dict[str, int] = defaultdict(int)

        for record in self.index.records:
            terms = set()
            for ent in record.entity_names:
                terms.update(self._tokenize(ent))
            for rel in record.relation_types:
                terms.update(self._tokenize(rel))
            terms.update(record.anchors)

            self._doc_terms.append(terms)
            self._doc_lengths.append(len(terms))

            for term in terms:
                doc_freq[term] += 1

        self._avg_doc_len = sum(self._doc_lengths) / max(1, n_docs)

        for term, df in doc_freq.items():
            self._idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        """BM25 score for a query against a document (record)."""
        doc_terms = self._doc_terms[doc_idx]
        doc_len = self._doc_lengths[doc_idx]

        score = 0.0
        for qt in query_tokens:
            if qt not in doc_terms:
                continue
            idf = self._idf.get(qt, 0)
            tf = 1  # binary TF (term is present or not)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / max(1, self._avg_doc_len)
            )
            score += idf * numerator / denominator
        return score

    def route(
        self,
        query: str,
        top_k_entities: int = 8,
        top_k_records: int = 4,
        top_k_themes: int = 3,
    ) -> RouteResult:
        """Route a query to relevant entities, records, and themes."""
        result = RouteResult()

        # Detect temporal
        result.has_temporal = bool(_TEMPORAL_PATTERNS.search(query))

        # Pronoun resolution
        query_lower = query.lower()
        query_tokens_raw = re.findall(r'\w+', query_lower)
        result.has_pronoun = bool(set(query_tokens_raw) & _PRONOUNS)

        # Expand pronouns
        if result.has_pronoun:
            query_tokens_raw = [
                "user" if t in _PRONOUNS else t for t in query_tokens_raw
            ]

        # Expand relation hints
        expanded = list(query_tokens_raw)
        for token in query_tokens_raw:
            if token in _QUERY_RELATION_HINTS:
                expanded.extend(_QUERY_RELATION_HINTS[token])

        query_tokens = [t for t in expanded if t not in _STOP_WORDS and len(t) > 1]

        # Extract entity-like patterns (e.g. person_14, org_6)
        q_entities = set(re.findall(r'\b\w+_\d+\b', query_lower))
        if result.has_pronoun:
            q_entities.add("user")

        # Extract file-path-like patterns (e.g. memory.py, decoder.py, auth.js)
        file_patterns = set(re.findall(r'\b[\w_-]+\.(?:py|js|ts|json|yaml|yml|md|toml|cfg)\b', query_lower))
        q_entities.update(file_patterns)

        # Also match bare filenames against full paths in the index
        # "memory.py" should match "pixelmem/memory.py"
        all_index_entities = self.index.all_entities()
        for fp in file_patterns:
            for ent in all_index_entities:
                if ent.endswith("/" + fp) or ent.endswith(os.sep + fp) or ent == fp:
                    q_entities.add(ent)
        # Also match keywords like "decoder", "encoder", "cli" against entity names
        for token in query_tokens:
            if len(token) > 3:
                for ent in all_index_entities:
                    # Match "decoder" to "pixelmem/decoder.py" or "decoder.py"
                    ent_base = ent.rsplit("/", 1)[-1].rsplit(".", 1)[0] if "/" in ent or "." in ent else ent
                    if token == ent_base:
                        q_entities.add(ent)

        # Phase 1: Exact entity match (highest priority)
        entity_scores: dict[str, float] = {}
        for ent in q_entities:
            records = self.index.lookup_entities([ent])
            if records:
                entity_scores[ent] = 100.0  # exact match

        # Phase 2: BM25 scoring over records
        record_scores: list[tuple[int, float]] = []
        for doc_idx in range(len(self.index.records)):
            score = self._bm25_score(query_tokens, doc_idx)
            # Boost if record contains exact entity match
            record = self.index.records[doc_idx]
            for ent in q_entities:
                if ent in record.entity_names:
                    score += 50.0
            if score > 0:
                record_scores.append((doc_idx, score))

        record_scores.sort(key=lambda x: -x[1])

        # Phase 3: Collect top entities from top records
        for doc_idx, score in record_scores[:top_k_records * 2]:
            record = self.index.records[doc_idx]
            for ent in record.entity_names:
                if ent not in entity_scores:
                    entity_scores[ent] = score * 0.1  # lower weight for indirect

        # Phase 4: Theme scoring
        theme_scores: dict[str, float] = defaultdict(float)
        for doc_idx, score in record_scores:
            record = self.index.records[doc_idx]
            theme = record.parent_id or record.id
            theme_scores[theme] += score

        # Build result
        result.entities = sorted(
            entity_scores.items(), key=lambda x: -x[1]
        )[:top_k_entities]

        result.records = [
            (self.index.records[doc_idx], score)
            for doc_idx, score in record_scores[:top_k_records]
        ]

        result.themes = sorted(
            theme_scores.items(), key=lambda x: -x[1]
        )[:top_k_themes]

        # Confidence
        max_score = max((s for _, s in result.records), default=0)
        result.confidence = min(1.0, max_score / 50.0) if max_score > 0 else 0.0

        # Query mode hint
        if result.has_temporal:
            result.query_mode_hint = "temporal"
        elif any("count" in t or "how many" in t for t in [query_lower]):
            result.query_mode_hint = "aggregation"
        elif q_entities and len(q_entities) >= 2:
            result.query_mode_hint = "relation_query"
        else:
            result.query_mode_hint = "fact_lookup"

        return result
