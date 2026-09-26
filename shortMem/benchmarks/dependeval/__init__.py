"""DependEval-specific pixel-encoded dependency ordering interface.

Public API:
    run_dependeval  -- end-to-end pipeline (drop-in for index_and_query)
    FileIDMapper    -- stable file-ID mapping
    PartialOrder    -- confidence-weighted ordering constraints
    PipelineResult  -- instrumented result with error attribution
"""

from .pipeline import run_dependeval
from .file_id_mapper import FileIDMapper
from .import_resolver import ImportCandidate
from .edge_confidence import DependencyEvidence
from .partial_order import PartialOrder
from .analyzer import PipelineResult

__all__ = [
    "run_dependeval",
    "FileIDMapper",
    "ImportCandidate",
    "DependencyEvidence",
    "PartialOrder",
    "PipelineResult",
]
