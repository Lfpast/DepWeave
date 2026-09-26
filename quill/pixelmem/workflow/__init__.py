"""PixelMem Workflow Memory — deterministic codebase knowledge extraction.

Encodes repository structure as pixel KG:
  - Files → entities
  - Functions/classes → entities
  - imports, calls, contains, depends_on → relations
  - line ranges, versions → conditions

No LLM needed for extraction — uses AST parsing and regex.
"""

from pixelmem.workflow.extractor import extract_repo, extract_file
from pixelmem.workflow.indexer import WorkflowIndexer

__all__ = ["extract_repo", "extract_file", "WorkflowIndexer"]
