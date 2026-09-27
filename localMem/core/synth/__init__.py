"""V5 synthesis — LLM-driven generation of domain-specific plugins.

Stages (see docs/v5_plan.md §4):

1. SchemaDesigner      → relation / condition vocabulary
2. ExtractorSynth      → Extractor plugin (template-driven)
3. DerivationSynth     → chain rules
4. PromptSynth         → PromptTemplate
5. Refiner             → patch any single stage from a failure log

Current implementation is template-driven (P3 in the plan): the LLM fills
slots in pre-written templates, rather than writing free-form code. This
keeps the execution path safe without a sandbox.
"""

from core.synth.schema_designer import SchemaDesigner, SchemaProposal
from core.synth.extractor_synth import ExtractorSynth
from core.synth.derivation_synth import DerivationSynth
from core.synth.prompt_synth import PromptSynth
from core.synth.refine import Refiner, RefinementOutcome
from core.synth.orchestrator import synthesize_pipeline, SynthesisResult

__all__ = [
    "SchemaDesigner",
    "SchemaProposal",
    "ExtractorSynth",
    "DerivationSynth",
    "PromptSynth",
    "Refiner",
    "RefinementOutcome",
    "synthesize_pipeline",
    "SynthesisResult",
]
