"""Shared data contracts, configuration, and infrastructure for the
Zero-Budget Autonomous Web Pentesting Framework.

Layering (import order is one-way; lower layers never import higher ones):

    schemas.py   <-- pure pydantic, no framework deps
       ^
       |
    state.py     <-- depends on schemas + langgraph (TypedDict only)
       |
    exceptions.py<-- pure python, no deps
       |
    config.py    <-- depends on exceptions (for ConfigurationError)
       |
    llm.py       <-- depends on config (for settings) + langchain SDKs

This top-level ``__init__`` re-exports the most commonly used symbols
from ``schemas`` and ``state`` so callers can write::

    from src.shared import AppState, Finding, Hypothesis

It deliberately does NOT import ``llm`` here, because that module pulls
in the heavyweight ``langchain-google-genai`` and ``langchain-openai``
SDKs. Keeping that import opt-in means:
- Tests can import ``AppState`` and ``Finding`` without installing the
  LLM SDKs.
- A misconfigured ``.env`` (missing API keys) does not prevent schema /
  state code from loading.
- Agent code that needs an LLM imports it explicitly:
  ``from src.shared.llm import gemini_flash``.

Configuration and exception symbols ARE re-exported here because they
are lightweight (no heavy SDK deps) and used pervasively across the
codebase.
"""

# ---------------------------------------------------------------------------
# Schemas + state (lightweight, no SDK deps)
# ---------------------------------------------------------------------------

from src.shared.schemas import (
    AttackComplexity,
    AttackVector,
    BusinessImpact,
    CIA,
    ConfidenceTrend,
    CrawlerResult,
    CVSSResult,
    CVSSScope,
    CVSSVector,
    ExecutionResult,
    Finding,
    FormInfo,
    HostInfo,
    HTTPMethod,
    HTTPRequestRecord,
    HTTPResponseRecord,
    Hypothesis,
    JSFile,
    Parameter,
    ParameterLocation,
    Payload,
    PayloadTransport,
    Phase,
    PrivilegesRequired,
    Protocol,
    RAGContext,
    RAGDocument,
    ReconResult,
    ScopeConfig,
    ServiceInfo,
    SeverityLevel,
    Target,
    UserInteraction,
    ValidationReport,
    ValidationVerdict,
    VulnerabilityCategory,
    WAFSignature,
)
from src.shared.state import AppState

# ---------------------------------------------------------------------------
# Exceptions (pure python)
# ---------------------------------------------------------------------------

from src.shared.exceptions import (
    ConfigurationError,
    CrawlerError,
    DependencyMissingError,
    ExecutionError,
    ExecutionTimeoutError,
    LLMError,
    LLMOutputParsingError,
    LLMRateLimitError,
    PentestFrameworkError,
    RAGUnavailableError,
    ScopeViolationError,
    ValidationInconclusiveError,
    WAFBlockError,
)

# ---------------------------------------------------------------------------
# Config (lightweight: pydantic-settings only)
# ---------------------------------------------------------------------------

from src.shared.config import (
    DEFAULT_ENV_PATH,
    KNOWLEDGE_BASE_DIR,
    PROJECT_ROOT,
    QDRANT_PERSIST_DIR,
    REPORTS_DIR,
    SCOPE_FILE_PATH,
    Settings,
    settings,
)

# ---------------------------------------------------------------------------
# Logging (lightweight: lazy structlog import, no SDK deps at import)
# ---------------------------------------------------------------------------

from src.shared.logging import (
    FRAMEWORK_ROOT_LOGGER_NAME,
    STANDARD_CONTEXT_KEYS,
    bind_context,
    get_logger,
    reset_context,
    setup_logging,
)

# NOTE: ``src.shared.llm`` is intentionally NOT imported here. Import it
# explicitly where needed: ``from src.shared.llm import gemini_flash``.

__all__ = [
    # State
    "AppState",
    # Enums
    "Phase",
    "SeverityLevel",
    "Protocol",
    "HTTPMethod",
    "ParameterLocation",
    "VulnerabilityCategory",
    "ConfidenceTrend",
    "PayloadTransport",
    "ValidationVerdict",
    "WAFSignature",
    "AttackVector",
    "AttackComplexity",
    "PrivilegesRequired",
    "UserInteraction",
    "CVSSScope",
    "CIA",
    # Schemas
    "ScopeConfig",
    "Target",
    "ServiceInfo",
    "HostInfo",
    "ReconResult",
    "Parameter",
    "FormInfo",
    "JSFile",
    "CrawlerResult",
    "Hypothesis",
    "Payload",
    "HTTPRequestRecord",
    "HTTPResponseRecord",
    "ExecutionResult",
    "ValidationReport",
    "RAGDocument",
    "RAGContext",
    "CVSSVector",
    "CVSSResult",
    "BusinessImpact",
    "Finding",
    # Exceptions
    "PentestFrameworkError",
    "ConfigurationError",
    "DependencyMissingError",
    "ScopeViolationError",
    "LLMError",
    "LLMOutputParsingError",
    "LLMRateLimitError",
    "ValidationInconclusiveError",
    "ExecutionError",
    "ExecutionTimeoutError",
    "WAFBlockError",
    "CrawlerError",
    "RAGUnavailableError",
    # Config
    "Settings",
    "settings",
    "PROJECT_ROOT",
    "DEFAULT_ENV_PATH",
    "SCOPE_FILE_PATH",
    "KNOWLEDGE_BASE_DIR",
    "QDRANT_PERSIST_DIR",
    "REPORTS_DIR",
    # Logging
    "setup_logging",
    "get_logger",
    "bind_context",
    "reset_context",
    "FRAMEWORK_ROOT_LOGGER_NAME",
    "STANDARD_CONTEXT_KEYS",
]
