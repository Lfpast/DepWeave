"""Workflow Indexer — encode repo structure into PixelMem shards."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from pixelmem.shard_manager import ShardManager
from pixelmem.workflow.extractor import extract_repo, extract_file


class WorkflowIndexer:
    """Index a repository into PixelMem for workflow memory queries."""

    def __init__(
        self,
        store_dir: Optional[str] = None,
        shard_size: int = 64,
    ):
        self.store_dir = store_dir or tempfile.mkdtemp()
        self.mgr = ShardManager(self.store_dir, shard_size=shard_size)
        self.stats = {"files": 0, "triples": 0, "entities": 0}

    def index_repo(
        self,
        repo_path: str,
        max_files: int = 500,
        include_calls: bool = True,
    ) -> dict:
        """Index an entire repository."""
        triples = extract_repo(
            repo_path,
            max_files=max_files,
            include_calls=include_calls,
        )

        # Encode in chunks of 10
        for i in range(0, len(triples), 10):
            self.mgr.encode("", triples=triples[i:i+10])

        self.mgr.save()
        self.mgr._rebuild_entity_index()

        self.stats["triples"] = sum(s.triple_count() for s in self.mgr.shards)
        self.stats["entities"] = len(set(
            n for s in self.mgr.shards for n in s.entity_to_idx
        ))

        return {
            "triples_extracted": len(triples),
            "triples_stored": self.stats["triples"],
            "entities": self.stats["entities"],
            "shards": len(self.mgr.shards),
        }

    def query(self, question: str, max_hops: int = 2) -> str:
        """Query the workflow KG."""
        from pixelmem.decoder import triples_to_text
        results = self.mgr.query(question, max_hops=max_hops)
        return triples_to_text(results)
