"""
shared/schemas.py
=================

Pydantic v2 data contracts for the Zero-Budget Autonomous Web Pentesting
Framework.

Every agent node in the LangGraph workflow exchanges data through these
models. They are intentionally strict (`extra="forbid"`) so that malformed
LLM JSON output, malformed tool output, or upstream schema drift fails
loudly at the boundary rather than propagating silent corruption downstream.

Design rules
------------
- All models are immutable by default (`frozen=True`); mutation creates a
  new instance via `model_copy(update={...})`. This keeps LangGraph state
  updates predictable and auditable.
- All datetime values are timezone-aware UTC.
- All enums inherit from `str` so they serialize to JSON as their value
  (and survive a round-trip through LangGraph's checkpointer).
- Optional fields default to `None`; collections default to empty via
  `Field(default_factory=...)` (never use a bare mutable default).
- `HttpUrl` is used for user-facing URLs; raw `str` is kept for internal
  identifiers (session_id, hypothesis_id, ...) so Pydantic's URL
  normalization does not mangle them.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class _BaseModel(BaseModel):
    """Common configuration for every schema in the framework.

    - ``frozen=True``      : instances are immutable. Updates go through
                             ``model_copy(update={...})`` which forces an
                             explicit, auditable copy.
    - ``extra="forbid"``   : unknown keys raise ``ValidationError`` instead
                             of being silently dropped. Catches schema drift
                             between LLM JSON output and our contracts.
    - ``populate_by_name=True`` : allows construction by either the python
                             field name or its serialized alias (useful when
                             ingesting JSON from third-party tools such as
                             Nmap XML / Subfinder JSON).
    - ``arbitrary_types_allowed=True`` : permits types Pydantic does not
                             natively serialize (e.g. Playwright objects
                             passed through briefly before normalization).
    - ``str_strip_whitespace=True`` : trims stray whitespace from string
                             inputs — a common cause of false negatives in
                             scope matching and URL comparison.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
        arbitrary_types_allowed=True,
        use_enum_values=False,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def _utc_now() -> datetime:
    """Timezone-aware UTC ``now`` factory. Used as a Field default_factory."""
    return datetime.now(UTC)


def _uuid4_str() -> str:
    """Stable UUID4 string factory. Used for deterministic ID fields."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Phase(str, Enum):
    """High-level workflow phases the Orchestrator can route between.

    The order in this enum is NOT the execution order — the Orchestrator
    node decides the next phase dynamically based on ``AppState``.
    """

    INITIALIZATION = "initialization"
    SCOPE_ENFORCEMENT = "scope_enforcement"
    PLANNING = "planning"
    RECON = "recon"
    CRAWLING = "crawling"
    HYPOTHESIS = "hypothesis"
    PAYLOAD_GENERATION = "payload_generation"
    PAYLOAD_OPTIMIZATION = "payload_optimization"
    EXECUTION = "execution"
    VALIDATION = "validation"
    RAG_LOOKUP = "rag_lookup"
    SCORING = "scoring"
    IMPACT_ANALYSIS = "impact_analysis"
    REPORTING = "reporting"
    COMPLETE = "complete"
    ERROR = "error"


class SeverityLevel(str, Enum):
    """Severity buckets used both for CVSS base severity and report ranking."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Protocol(str, Enum):
    """Application-layer protocols in scope of the framework."""

    HTTP = "http"
    HTTPS = "https"
    WS = "ws"
    WSS = "wss"


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    CONNECT = "CONNECT"
    TRACE = "TRACE"


class ParameterLocation(str, Enum):
    """Where a parameter lives in an HTTP request."""

    QUERY = "query"
    PATH = "path"
    HEADER = "header"
    COOKIE = "cookie"
    BODY_FORM = "body_form"
    BODY_JSON = "body_json"
    BODY_XML = "body_xml"
    BODY_MULTIPART = "body_multipart"


class VulnerabilityCategory(str, Enum):
    """Taxonomy of web vulnerabilities the framework can hypothesize about.

    Values are short slugs (e.g. ``sqli``) so they double as RAG filter
    tags and as Markdown report anchors.
    """

    SQL_INJECTION = "sqli"
    SQL_INJECTION_BLIND = "sqli_blind"
    SQL_INJECTION_TIME = "sqli_time"
    XSS_REFLECTED = "xss_reflected"
    XSS_STORED = "xss_stored"
    XSS_DOM = "xss_dom"
    IDOR = "idor"
    BOLA = "bola"  # Broken Object Level Authorization
    BFLA = "bfla"  # Broken Function Level Authorization
    SSRF = "ssrf"
    XXE = "xxe"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    OPEN_REDIRECT = "open_redirect"
    CSRF = "csrf"
    SSTI = "ssti"  # Server-Side Template Injection
    DESERIALIZATION = "deserialization"
    GRAPHQL_INTROSPECTION = "graphql_introspection"
    GRAPHQL_BATCHING = "graphql_batching"
    JWT_NONE_ALG = "jwt_none_alg"
    JWT_WEAK_SECRET = "jwt_weak_secret"
    AUTH_BYPASS = "auth_bypass"
    SESSION_FIXATION = "session_fixation"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    SECURITY_MISCONFIG = "security_misconfiguration"
    BROKEN_ACCESS_CONTROL = "broken_access_control"
    VULNERABLE_COMPONENTS = "vulnerable_components"
    BUSINESS_LOGIC = "business_logic"
    RACE_CONDITION = "race_condition"
    UNKNOWN = "unknown"


class ConfidenceTrend(str, Enum):
    """Direction of confidence change between iterations of a hypothesis."""

    RISING = "rising"
    STABLE = "stable"
    FALLING = "falling"


class PayloadTransport(str, Enum):
    """How a payload is delivered to the target."""

    HTTP_REQUEST = "http_request"
    WEBSOCKET = "websocket"
    PLAYWRIGHT_DOM = "playwright_dom"
    GRAPHQL_QUERY = "graphql_query"


class ValidationVerdict(str, Enum):
    """Outcome of the Validator node's True/False positive analysis."""

    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    INCONCLUSIVE = "inconclusive"


class WAFSignature(str, Enum):
    """WAF fingerprints the Recon Parser / Execution Sandbox can detect."""

    CLOUDFLARE = "cloudflare"
    AWS_WAF = "aws_waf"
    AKAMAI = "akamai"
    F5_BIGIP = "f5_bigip"
    IMPERVA = "imperva"
    SUCURI = "sucuri"
    MODSECURITY = "modsecurity"
    UNKNOWN = "unknown"
    NONE = "none"


# ---------------------------------------------------------------------------
# CVSS 3.1 enums
# ---------------------------------------------------------------------------


class AttackVector(str, Enum):
    NETWORK = "network"
    ADJACENT = "adjacent"
    LOCAL = "local"
    PHYSICAL = "physical"


class AttackComplexity(str, Enum):
    LOW = "low"
    HIGH = "high"


class PrivilegesRequired(str, Enum):
    NONE = "none"
    LOW = "low"
    HIGH = "high"


class UserInteraction(str, Enum):
    NONE = "none"
    REQUIRED = "required"


class CVSSScope(str, Enum):
    """Renamed to ``CVSSScope`` to avoid clashing with Pydantic internals."""

    UNCHANGED = "unchanged"
    CHANGED = "changed"


class CIA(str, Enum):
    """Confidentiality / Integrity / Availability impact metric."""

    NONE = "none"
    LOW = "low"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class ScopeConfig(_BaseModel):
    """Loaded verbatim from ``scope.json`` at session start.

    The Scope Enforcer node reads this and refuses to operate against any
    target that falls outside ``in_scope_domains`` / ``in_scope_cidrs`` or
    inside ``out_of_scope_domains`` / ``out_of_scope_cidrs``. Out-of-scope
    entries always take precedence — this is a fail-closed design.
    """

    in_scope_domains: list[str] = Field(
        default_factory=list,
        description="Allowed target domains, e.g. ['example.com', 'api.example.com'].",
    )
    in_scope_cidrs: list[str] = Field(
        default_factory=list,
        description="Allowed CIDR ranges, e.g. ['10.0.0.0/24'].",
    )
    out_of_scope_domains: list[str] = Field(
        default_factory=list,
        description="Explicit exclusions; take precedence over in-scope.",
    )
    out_of_scope_cidrs: list[str] = Field(
        default_factory=list,
        description="Explicit CIDR exclusions; take precedence over in-scope.",
    )
    allowed_ports: list[int] = Field(
        default_factory=lambda: [80, 443, 8000, 8080, 8443],
        description="Web-relevant ports only. Non-web ports are dropped by the Web Filter node.",
    )
    forbidden_paths: list[str] = Field(
        default_factory=list,
        description="Paths the operator does NOT want touched, e.g. ['/admin', '/logout'].",
    )
    requires_auth: bool = Field(
        default=False,
        description="If True, the framework must acquire credentials/tokens before attacking.",
    )
    auth_config: dict[str, Any] | None = Field(
        default=None,
        description="Auth recipe: login URL, form fields, token extraction rules, etc.",
    )
    max_requests_per_second: int = Field(
        default=10,
        ge=1,
        description="Global rate limit enforced by the Execution Sandbox.",
    )
    max_concurrent_requests: int = Field(
        default=5,
        ge=1,
        description="Concurrency cap enforced by the Execution Sandbox.",
    )
    legal_acknowledged: bool = Field(
        default=False,
        description="Operator attests written authorization exists. Refuses to run if False.",
    )


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


class Target(_BaseModel):
    """The single target the framework is currently pointed at.

    A target is a specific URL + HTTP method pair, not just a hostname.
    This lets the Planner reason about endpoints directly rather than
    re-deriving them on every hop.
    """

    url: HttpUrl
    method: HTTPMethod = HTTPMethod.GET
    is_in_scope: bool = Field(
        default=False,
        description="Set by the Scope Enforcer node. Defaults to False so a missed "
        "scope check is a fail-closed refusal.",
    )
    scope_verified_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Recon
# ---------------------------------------------------------------------------


class ServiceInfo(_BaseModel):
    """One listening service on a host, as parsed from Nmap output."""

    port: int = Field(ge=1, le=65535)
    protocol: Protocol
    service_name: str
    service_version: str | None = None
    product: str | None = None
    banner: str | None = None
    is_web: bool = Field(
        default=False,
        description="True if service_name in {http, https, http-alt, http-proxy, https-alt}.",
    )


class HostInfo(_BaseModel):
    """One discovered host with its services."""

    hostname: str | None = None
    ip_address: str
    is_alive: bool = True
    os_guess: str | None = None
    services: list[ServiceInfo] = Field(default_factory=list)
    discovered_at: datetime = Field(default_factory=_utc_now)


class ReconResult(_BaseModel):
    """Output of the Recon Parser node.

    The Web Filter has already stripped non-web services by the time this
    is populated — any ServiceInfo here is guaranteed to have
    ``is_web == True``.
    """

    target: Target
    hosts: list[HostInfo] = Field(default_factory=list)
    web_endpoints: list[HttpUrl] = Field(default_factory=list)
    subdomains: list[str] = Field(default_factory=list)
    technologies_detected: list[str] = Field(default_factory=list)
    waf_signature: WAFSignature = WAFSignature.NONE
    parsed_at: datetime = Field(default_factory=_utc_now)
    source_tool: str = Field(
        default="unknown",
        description="Which tool produced the raw output: 'nmap', 'subfinder', 'httpx', etc.",
    )


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------


class Parameter(_BaseModel):
    """A single injectable parameter discovered during crawling.

    The Crawler Parser fills in ``is_reflected`` and ``is_injectable``
    using simple heuristics so the Hypothesis Analyzer can prioritize
    high-signal parameters without re-running the crawl.
    """

    name: str
    location: ParameterLocation
    value: str | None = None
    param_type: str | None = Field(
        default=None,
        description="Inferred type: 'string', 'int', 'bool', 'json', 'file', etc.",
    )
    is_reflected: bool = Field(
        default=False,
        description="True if the value appears unescaped in the HTTP response body.",
    )
    is_injectable: bool = Field(
        default=False,
        description="Heuristic: True if reflection + dynamic context (e.g. query + HTML body).",
    )


class FormInfo(_BaseModel):
    """An HTML form discovered by the crawler."""

    action: HttpUrl
    method: HTTPMethod
    fields: list[Parameter] = Field(default_factory=list)
    has_csrf_token: bool = False
    enctype: str | None = None


class JSFile(_BaseModel):
    """A JavaScript file referenced by the target application."""

    url: HttpUrl
    content_sha256: str | None = None
    size_bytes: int | None = None
    endpoints_discovered: list[str] = Field(default_factory=list)
    secrets_discovered: list[str] = Field(
        default_factory=list,
        description="Redacted secret fingerprints, e.g. 'aws_access_key_id pattern match'.",
    )
    interesting_patterns: list[str] = Field(default_factory=list)


class CrawlerResult(_BaseModel):
    """Output of the Crawler Parser node."""

    target: Target
    urls: list[HttpUrl] = Field(
        default_factory=list,
        description="All unique URLs discovered by Playwright crawl.",
    )
    parameters: list[Parameter] = Field(default_factory=list)
    forms: list[FormInfo] = Field(default_factory=list)
    js_files: list[JSFile] = Field(default_factory=list)
    cookies: list[dict[str, Any]] = Field(default_factory=list)
    storage_entries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="localStorage / sessionStorage / indexedDB entries Playwright captured.",
    )
    crawled_at: datetime = Field(default_factory=_utc_now)
    crawl_depth: int = Field(default=0, ge=0)
    crawl_duration_seconds: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


class Hypothesis(_BaseModel):
    """A candidate vulnerability produced by the Hypothesis Analyzer.

    Every hypothesis has a deterministic UUID so downstream nodes can refer
    to it without ambiguity, even after LangGraph state mutations.
    """

    id: str = Field(default_factory=_uuid4_str)
    category: VulnerabilityCategory
    target_url: HttpUrl
    target_parameter: Parameter | None = None
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Analyst-style confidence score in [0.0, 1.0].",
    )
    confidence_trend: ConfidenceTrend = ConfidenceTrend.STABLE
    reasoning: str
    evidence: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    needs_optimization: bool = Field(
        default=False,
        description="True if WAF detected OR confidence < 0.6. Triggers Payload Optimizer.",
    )
    created_at: datetime = Field(default_factory=_utc_now)
    related_hypothesis_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


class Payload(_BaseModel):
    """A crafted proof-of-concept produced by the Payload Generator and
    optionally refined by the Payload Optimizer.

    The ``detection_signature`` field is critical: it is the deterministic
    marker the Validator looks for in the response to confirm exploitation.
    Without it, the Validator cannot distinguish a true positive from a
    false positive.
    """

    id: str = Field(default_factory=_uuid4_str)
    hypothesis_id: str
    raw: str = Field(description="The actual payload string to inject.")
    encoded_variants: list[str] = Field(
        default_factory=list,
        description="Alternative encodings: base64, url, double-url, unicode, etc.",
    )
    transport: PayloadTransport
    http_method: HTTPMethod | None = None
    target_url: HttpUrl
    injection_point: Parameter | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    expected_behavior: str = Field(
        description="What a successful exploitation should look like.",
    )
    detection_signature: str = Field(
        description="Deterministic marker the Validator will look for in the response.",
    )
    is_optimized: bool = False
    created_at: datetime = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class HTTPRequestRecord(_BaseModel):
    """A complete HTTP request as sent by the Execution Sandbox."""

    method: HTTPMethod
    url: HttpUrl
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = None
    body_bytes_b64: str | None = Field(
        default=None,
        description="Base64-encoded binary body (for non-text payloads).",
    )


class HTTPResponseRecord(_BaseModel):
    """A complete HTTP response as received by the Execution Sandbox."""

    status_code: int = Field(ge=100, le=599)
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = None
    body_bytes_b64: str | None = None
    elapsed_ms: int = Field(ge=0)
    received_at: datetime = Field(default_factory=_utc_now)


class ExecutionResult(_BaseModel):
    """Output of the Execution Sandbox.

    Contains a complete transcript of what was sent and what was received.
    No interpretation here — that is the Validator's job.
    """

    payload_id: str
    request: HTTPRequestRecord
    response: HTTPResponseRecord | None = None
    error: str | None = None
    retry_attempts: int = Field(default=0, ge=0)
    rate_limited: bool = False
    executed_at: datetime = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationReport(_BaseModel):
    """Output of the Validator node.

    Decides TRUE_POSITIVE / FALSE_POSITIVE / INCONCLUSIVE by comparing the
    ExecutionResult against the Payload's ``detection_signature`` and
    ``expected_behavior``. The ``id`` is the canonical handle referenced
    by ``Finding.validation_id`` for end-to-end traceability.
    """

    id: str = Field(default_factory=_uuid4_str)
    payload_id: str
    hypothesis_id: str
    verdict: ValidationVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    matched_signatures: list[str] = Field(default_factory=list)
    reasoning: str
    supporting_evidence: list[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


class RAGDocument(_BaseModel):
    """One retrieved chunk from the Qdrant knowledge base."""

    source: str
    section: str | None = None
    content: str
    methodology_tags: list[VulnerabilityCategory] = Field(default_factory=list)
    source_url: HttpUrl | None = None


class RAGContext(_BaseModel):
    """Result of a Knowledge RAG lookup.

    Stored on AppState so downstream nodes (Reporter, Business Impact
    Writer) can cite the same sources without re-querying Qdrant.
    """

    query: str
    retrieved_documents: list[RAGDocument] = Field(default_factory=list)
    similarity_scores: list[float] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=_utc_now)
    embedder_model: str = "bge-m3"


# ---------------------------------------------------------------------------
# CVSS 3.1
# ---------------------------------------------------------------------------


class CVSSVector(_BaseModel):
    """CVSS 3.1 base metrics.

    The CVSS Engine (pure Python) consumes this and produces a
    ``CVSSResult`` deterministically — no LLM in the loop. This guarantees
    reproducible scoring across runs.
    """

    attack_vector: AttackVector
    attack_complexity: AttackComplexity
    privileges_required: PrivilegesRequired
    user_interaction: UserInteraction
    scope: CVSSScope
    confidentiality: CIA
    integrity: CIA
    availability: CIA


class CVSSResult(_BaseModel):
    """Scored CVSS vector with computed base score and severity."""

    finding_id: str
    vector: CVSSVector
    base_score: float = Field(ge=0.0, le=10.0)
    base_severity: SeverityLevel
    vector_string: str
    calculated_at: datetime = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Business Impact
# ---------------------------------------------------------------------------


class BusinessImpact(_BaseModel):
    """Output of the Business Impact Writer node.

    The ``narrative`` field is what the Reporter embeds verbatim into the
    final Markdown report — it is plain prose, not structured data.
    """

    finding_id: str
    narrative: str = Field(
        description="1-3 paragraph plain-language narrative the reporter embeds verbatim.",
    )
    affected_assets: list[str] = Field(default_factory=list)
    financial_impact: str | None = None
    operational_impact: str | None = None
    legal_impact: str | None = None
    reputational_impact: str | None = None
    recommended_mitigation: list[str] = Field(default_factory=list)
    written_at: datetime = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Finding (canonical confirmed vuln)
# ---------------------------------------------------------------------------


class Finding(_BaseModel):
    """A confirmed vulnerability.

    Assembled by the Orchestrator after the Validator returns
    ``TRUE_POSITIVE`` and the CVSS Engine + Business Impact Writer have
    enriched it. This is the canonical object that gets persisted to disk
    and rendered in the final report.
    """

    id: str = Field(default_factory=_uuid4_str)
    hypothesis_id: str
    payload_id: str
    validation_id: str | None = None
    category: VulnerabilityCategory
    title: str
    target_url: HttpUrl
    target_parameter: Parameter | None = None
    severity: SeverityLevel
    cvss: CVSSResult | None = None
    business_impact: BusinessImpact | None = None
    proof_of_concept: str = Field(
        description="Sanitized PoC the reporter can include verbatim.",
    )
    raw_request: HTTPRequestRecord | None = None
    raw_response: HTTPResponseRecord | None = None
    rag_references: list[RAGDocument] = Field(default_factory=list)
    confirmed_at: datetime = Field(default_factory=_utc_now)
    tags: list[str] = Field(default_factory=list)
