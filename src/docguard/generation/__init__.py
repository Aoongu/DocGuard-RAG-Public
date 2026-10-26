"""Structured generation and auditable citation primitives."""

from docguard.generation.citations import (
    GenerationValidationError,
    parse_generation_output,
    verify_citations,
)
from docguard.generation.client import (
    FixedGenerationClient,
    GenerationAuthenticationError,
    GenerationClientClosedError,
    GenerationClientError,
    GenerationRateLimitError,
    GenerationResponseError,
    GenerationUpstreamError,
    OpenAICompatibleClient,
    StructuredGenerationClient,
)
from docguard.generation.config import GenerationConfig, load_generation_config
from docguard.generation.evidence import EvidenceMappingError, build_evidence
from docguard.generation.models import (
    Evidence,
    EvidenceProvenance,
    GeneratedAnswer,
    GenerationCompletion,
    GenerationRequest,
    GenerationTiming,
    GenerationUsage,
    ParsedGenerationOutput,
    VerifiedCitation,
    VerifiedGeneration,
)
from docguard.generation.pipeline import B2Pipeline
from docguard.generation.prompt import (
    build_messages,
    output_schema,
    output_schema_sha256,
    prompt_sha256,
)
from docguard.generation.service import GenerationService

__all__ = [
    "B2Pipeline",
    "Evidence",
    "EvidenceMappingError",
    "EvidenceProvenance",
    "FixedGenerationClient",
    "GeneratedAnswer",
    "GenerationAuthenticationError",
    "GenerationClientClosedError",
    "GenerationClientError",
    "GenerationCompletion",
    "GenerationConfig",
    "GenerationRateLimitError",
    "GenerationRequest",
    "GenerationResponseError",
    "GenerationService",
    "GenerationTiming",
    "GenerationUpstreamError",
    "GenerationUsage",
    "GenerationValidationError",
    "OpenAICompatibleClient",
    "ParsedGenerationOutput",
    "StructuredGenerationClient",
    "VerifiedCitation",
    "VerifiedGeneration",
    "build_evidence",
    "build_messages",
    "load_generation_config",
    "output_schema",
    "output_schema_sha256",
    "parse_generation_output",
    "prompt_sha256",
    "verify_citations",
]
