"""
shared/exceptions.py
====================

Domain exception hierarchy for the Zero-Budget Autonomous Web Pentesting
Framework.

Design rules
------------
- Every framework-specific error inherits from ``PentestFrameworkError``,
  which inherits from ``Exception``. This gives callers a single catch
  point (``except PentestFrameworkError``) for any error originating from
  inside the framework, distinct from third-party exceptions
  (``httpx.TimeoutException``, ``ValidationError``, etc.).
- Each exception carries a ``details`` dict for machine-readable context
  that can be serialized into LangGraph's ``errors`` channel or surfaced
  via the FastAPI layer. The human-readable message goes to the standard
  ``args`` / ``str(exc)``.
- Exceptions are organized into three broad families:
    1. Configuration & dependency errors — raised at startup, fail fast.
    2. Agent / pipeline errors            — raised mid-execution,
                                            recoverable via retries.
    3. Execution & infrastructure errors  — raised by the sandbox or
                                            external tooling, often
                                            retryable but rate-limited.
- No exception swallows the original cause. Always use
  ``raise NewError(...) from original`` when re-raising.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class PentestFrameworkError(Exception):
    """Root of every framework-specific exception.

    Callers can do ``except PentestFrameworkError`` to catch any error
    originating from inside the framework, while still letting
    ``KeyboardInterrupt``, ``SystemExit``, and unrelated third-party
    exceptions propagate untouched.

    Parameters
    ----------
    message:
        Human-readable explanation. Becomes ``str(exc)``.
    details:
        Optional machine-readable context (session_id, node_name,
        target_url, etc.). Stored verbatim; never mutated by the
        framework. Callers can serialize it to JSON for logging or for
        the LangGraph ``errors`` channel.
    """

    def __init__(
        self,
        message: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details) if details else {}

    def __str__(self) -> str:
        if not self.message:
            return self.__class__.__name__
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs / state.

        Includes the exception class name, message, details dict, and the
        stringified cause (if any) so downstream consumers (LangGraph
        error channel, FastAPI error response, structured log writer) get
        a uniform shape.
        """
        return {
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "cause": str(self.__cause__) if self.__cause__ else None,
        }


# ---------------------------------------------------------------------------
# 1. Configuration & dependency errors (fail fast at startup)
# ---------------------------------------------------------------------------


class ConfigurationError(PentestFrameworkError):
    """Raised when required configuration is missing or invalid.

    Typical causes:
    - A required env var (``GEMINI_API_KEY``, ``DEEPSEEK_API_KEY``) is not
      set or is empty.
    - ``scope.json`` is malformed or references a non-existent path.
    - ``legal_acknowledged`` is False when the operator attempted to
      start a session.
    """


class DependencyMissingError(PentestFrameworkError):
    """Raised when an optional system dependency is required but absent.

    Examples:
    - ``playwright`` browsers not installed (``playwright install``).
    - ``nmap`` binary not on ``PATH``.
    - ``qdrant`` server unreachable at the configured URL.
    - BGE-M3 embedder weights not downloaded.

    The ``dependency`` attribute identifies which component is missing so
    the operator can resolve it without reading the traceback.
    """

    def __init__(
        self,
        message: str,
        *,
        dependency: str,
        install_hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details: dict[str, Any] = {
            "dependency": dependency,
            "install_hint": install_hint,
        }
        if details:
            merged_details.update(details)
        super().__init__(message, details=merged_details)
        self.dependency = dependency
        self.install_hint = install_hint


# ---------------------------------------------------------------------------
# 2. Scope & policy errors (fail closed — never attack out-of-scope)
# ---------------------------------------------------------------------------


class ScopeViolationError(PentestFrameworkError):
    """Raised when an action would touch a target outside the authorized
    scope, or hit a forbidden path/CIDR.

    This is the framework's most safety-critical error: the Scope Enforcer
    node, the Execution Sandbox, and any tool that issues a network request
    MUST re-check scope immediately before sending and raise this if the
    check fails. Treat any uncaught ``ScopeViolationError`` as a P0
    incident.

    Attributes
    ----------
    target:
        The URL, hostname, or CIDR that was about to be touched.
    reason:
        Short machine-readable code, e.g. ``"domain_not_in_scope"``,
        ``"cidr_out_of_scope"``, ``"path_forbidden"``,
        ``"port_not_allowed"``.
    """

    def __init__(
        self,
        message: str,
        *,
        target: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details: dict[str, Any] = {"target": target, "reason": reason}
        if details:
            merged_details.update(details)
        super().__init__(message, details=merged_details)
        self.target = target
        self.reason = reason


# ---------------------------------------------------------------------------
# 3. LLM / agent pipeline errors
# ---------------------------------------------------------------------------


class LLMError(PentestFrameworkError):
    """Base class for any error originating from an LLM call or its
    output processing. Subclassed for specific failure modes."""


class LLMOutputParsingError(LLMError):
    """Raised when an LLM response cannot be parsed into the expected
    Pydantic schema.

    The Hypothesis Analyzer, Payload Generator, Validator, etc. all
    expect the LLM to return structured JSON. When that JSON is missing,
    malformed, or fails schema validation, this error is raised so the
    Orchestrator can decide whether to retry with a stricter prompt or
    abandon the hypothesis.

    Attributes
    ----------
    raw_output:
        The verbatim LLM response that failed to parse, kept for
        debugging. Never redacted here — redaction is the reporter's job.
    schema_name:
        The Pydantic model class name the parser was targeting.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_output: str,
        schema_name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details: dict[str, Any] = {
            "schema_name": schema_name,
            "raw_output_length": len(raw_output),
            "raw_output_preview": raw_output[:500],
        }
        if details:
            merged_details.update(details)
        super().__init__(message, details=merged_details)
        self.raw_output = raw_output
        self.schema_name = schema_name


class LLMRateLimitError(LLMError):
    """Raised when an LLM provider returns a 429 / quota-exceeded
    response and the built-in retry budget is exhausted.

    Distinct from ``LLMError`` so the Orchestrator can apply different
    back-off policy (e.g. switch to a cheaper fallback model) without
    conflating it with parsing failures.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        retry_after_seconds: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "retry_after_seconds": retry_after_seconds,
        }
        if details:
            merged_details.update(details)
        super().__init__(message, details=merged_details)
        self.provider = provider
        self.model = model
        self.retry_after_seconds = retry_after_seconds


class ValidationInconclusiveError(PentestFrameworkError):
    """Raised by the Validator node when it cannot reach a True/False
    Positive verdict with sufficient confidence.

    This is NOT a parsing failure — the Validator successfully executed
    its logic but the evidence was ambiguous (e.g. WAF returned a generic
    403 for both the malicious and the benign control payload). The
    Orchestrator typically responds by routing back to Payload Optimizer.

    Attributes
    ----------
    payload_id:
        The payload that was being validated.
    confidence:
        The Validator's confidence in its best-guess verdict, in [0, 1].
        Will be below the configured threshold (default 0.6).
    """

    def __init__(
        self,
        message: str,
        *,
        payload_id: str,
        confidence: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details: dict[str, Any] = {
            "payload_id": payload_id,
            "confidence": confidence,
        }
        if details:
            merged_details.update(details)
        super().__init__(message, details=merged_details)
        self.payload_id = payload_id
        self.confidence = confidence


# ---------------------------------------------------------------------------
# 4. Execution & infrastructure errors
# ---------------------------------------------------------------------------


class ExecutionError(PentestFrameworkError):
    """Base class for any error originating from the Execution Sandbox
    or its underlying HTTP / browser transport."""


class ExecutionTimeoutError(ExecutionError):
    """Raised when a payload execution exceeds the configured timeout.

    Distinct from a generic ``httpx.TimeoutException`` so the framework
    can apply its own retry / escalation policy (e.g. one timeout is
    retryable; three consecutive timeouts against the same target abort
    the hypothesis).

    Attributes
    ----------
    payload_id:
        The payload that timed out.
    timeout_seconds:
        The configured timeout that was exceeded.
    transport:
        ``"httpx"`` or ``"playwright"`` — identifies which transport
        raised the timeout, since they have very different performance
        profiles.
    """

    def __init__(
        self,
        message: str,
        *,
        payload_id: str,
        timeout_seconds: float,
        transport: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details: dict[str, Any] = {
            "payload_id": payload_id,
            "timeout_seconds": timeout_seconds,
            "transport": transport,
        }
        if details:
            merged_details.update(details)
        super().__init__(message, details=merged_details)
        self.payload_id = payload_id
        self.timeout_seconds = timeout_seconds
        self.transport = transport


class WAFBlockError(ExecutionError):
    """Raised when the Execution Sandbox detects that a WAF blocked the
    payload (HTTP 403 with a known WAF signature page, or a connection
    reset mid-request after repeated suspicious requests).

    This is the signal the Orchestrator uses to route to the Payload
    Optimizer node — it is NOT necessarily a failure of the hypothesis,
    just an indication that the current payload shape was filtered.

    Attributes
    ----------
    payload_id:
        The payload that was blocked.
    waf_signature:
        The detected WAF fingerprint (``cloudflare``, ``aws_waf``, ...).
        ``unknown`` if we could not classify.
    status_code:
        The HTTP status returned (typically 403, occasionally 502/520).
    """

    def __init__(
        self,
        message: str,
        *,
        payload_id: str,
        waf_signature: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details: dict[str, Any] = {
            "payload_id": payload_id,
            "waf_signature": waf_signature,
            "status_code": status_code,
        }
        if details:
            merged_details.update(details)
        super().__init__(message, details=merged_details)
        self.payload_id = payload_id
        self.waf_signature = waf_signature
        self.status_code = status_code


class CrawlerError(ExecutionError):
    """Raised by the Playwright-based crawler when it cannot reach a
    page, when navigation is blocked by a JS challenge it cannot solve,
    or when the DOM does not settle within the configured wait budget."""


class RAGUnavailableError(PentestFrameworkError):
    """Raised by the Knowledge RAG node when Qdrant is unreachable or
    the BGE-M3 embedder is not initialized.

    The Knowledge RAG node is *advisory* — its absence should not block
    the pipeline — so the Orchestrator catches this and continues with an
    empty ``RAGContext``. The error is still raised so the operator is
    aware the methodology lookup was skipped.
    """


# ---------------------------------------------------------------------------
# Convenience grouping
# ---------------------------------------------------------------------------


# These names are exported via ``src.shared.__init__``. Keep the list in
# sync with the ``__all__`` block there.
__all__ = [
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
]
