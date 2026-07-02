"""
src/agents/payload_generator.py
===============================

Node 9 of the 16-node LangGraph framework: the **Payload Generator** —
the "Weaponsmith" that crafts benign Proof-of-Concept payloads.

This node consumes the vulnerability hypotheses produced by the
Hypothesis Analyzer (Node 8) and generates a :class:`Payload` for each
hypothesis that does not already have one. The payloads are benign PoCs
designed to *prove* a vulnerability exists without causing harm —
``SLEEP(10)`` for time-based SQLi, ``alert(1)`` for XSS,
``cat /etc/passwd`` for command injection, etc.

Why DeepSeek V3?
----------------
The Architect specified DeepSeek V3 (``deepseek-chat``) for this node
because payload crafting requires precise, schema-compliant output
rather than deep reasoning. V3 is faster than R1, cheaper, and —
crucially — supports ``with_structured_output()`` natively, which
guarantees the LLM's response conforms to the :class:`Payload` schema
at the API level. No ``<think>`` tag preprocessing needed.

Deduplication logic
-------------------
The node iterates through ``state["hypotheses"]`` and skips any
hypothesis that already has a payload in ``state["payloads"]`` (matched
by ``payload.hypothesis_id == hypothesis.id``). This makes the node
idempotent: re-running it after a partial failure only generates
payloads for the hypotheses that are still missing one.

Per-hypothesis error isolation
------------------------------
If a single payload generation fails (LLM parse error, rate limit,
unexpected exception), the node logs the error with the hypothesis_id
context and **continues to the next hypothesis**. It does NOT crash
the entire node. This is critical for long-running sessions where one
bad LLM response should not waste the work done for other hypotheses.

Defense-in-depth post-processing
--------------------------------
The LLM is instructed to follow several constraints, but we do NOT
trust it. The :func:`_enforce_payload_fields` helper corrects:

1. **``hypothesis_id``** — forcibly set to the actual hypothesis ID.
   The LLM is told the ID in the prompt, but may hallucinate a
   different one.

2. **``is_optimized``** — forcibly set to ``False``. The Payload
   Optimizer (Node 10) is the only node allowed to set this to
   ``True``.

3. **``target_url``** — forcibly set to the hypothesis's target_url.
   Prevents the LLM from targeting a different URL than the one the
   hypothesis was about.

4. **``injection_point``** — forcibly set to the hypothesis's
   ``target_parameter``. Ensures the payload's injection point matches
   what the Hypothesis Analyzer identified.

5. **``transport``** — validated to be ``HTTP_REQUEST`` or
   ``GRAPHQL_QUERY``. If the LLM returns ``WEBSOCKET`` or
   ``PLAYWRIGHT_DOM``, the transport is reset to ``HTTP_REQUEST``
   (the most common case) and a warning is logged.

LangGraph contract
------------------
::

    async def generate_payloads(state: AppState) -> dict:

- Reads: ``state["hypotheses"]`` (list of :class:`Hypothesis`),
         ``state["payloads"]`` (list of :class:`Payload` — may be
         empty or missing).
- Writes: returns ``{"payloads": [<Payload>, ...]}`` — a list of
  zero or more newly-generated payloads, ready to be merged into
  the ``payloads`` channel via the ``operator.add`` reducer.

Raises
------
This node does NOT raise :class:`LLMOutputParsingError` or
:class:`LLMRateLimitError` to the caller. Per the Architect's spec,
single-hypothesis failures are logged and the node continues. The
node only raises :class:`PentestFrameworkError` if ``hypotheses`` is
missing entirely.
"""

from __future__ import annotations

from typing import Any

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from src.shared.exceptions import (
    LLMOutputParsingError,
    LLMRateLimitError,
    PentestFrameworkError,
)
from src.shared.llm import deepseek_v3
from src.shared.logging import get_logger
from src.shared.schemas import (
    Hypothesis,
    HTTPMethod,
    Payload,
    PayloadTransport,
)
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT: str = """\
You are an Expert Exploit Developer embedded in an automated web \
pentesting framework. Your job is to craft a safe, benign \
Proof-of-Concept (PoC) payload for a given vulnerability hypothesis.

BENIGN PAYLOAD POLICY — VIOLATING THIS IS A P0 BUG:
1. The payload MUST be benign. It must prove the vulnerability exists \
WITHOUT causing any damage to the target system.
2. USE these benign indicators:
   - SQLi (time-based): SLEEP(10), BENCHMARK(10000000, MD5('x')), \
PG_SLEEP(10)
   - SQLi (error-based): a single quote ' to trigger a database error, \
or a UNION SELECT with a harmless marker string
   - SQLi (boolean): ' OR '1'='1 -- (returns all rows, no data modification)
   - XSS: <script>alert(1)</script>, <img src=x onerror=alert(1)>, \
javascript:alert(1)
   - SSRF: a URL pointing to a safe internal address (e.g. \
http://169.254.169.254/latest/meta-data/ for AWS, \
http://localhost:80/) or a Burp Collaborator-style canary domain
   - Command Injection: ; cat /etc/passwd, | id, $(whoami), `id`
   - Path Traversal: ../../../../etc/passwd, ..\\..\\..\\windows\\win.ini
   - Open Redirect: a URL pointing to https://example.com/ (a safe \
redirect target)
   - SSTI: {{7*7}} (Jinja2), ${7*7} (FreeMarker/Thymeleaf), \
#{7*7} (Ruby ERB)
   - XXE: a benign entity definition that reads /etc/passwd
   - GraphQL: __schema query for introspection, or a batch query array
   - JWT: alg:none header to test for the none-algorithm vulnerability
3. NEVER use destructive payloads:
   - NO rm, del, DROP TABLE, DELETE FROM, TRUNCATE, format, mkfs, \
shutdown, or any command that modifies or deletes data.
   - NO reverse shells — this is a PoC, not a real attack.
   - NO privilege escalation exploits — prove the vulnerability, \
don't exploit it fully.

DETECTION SIGNATURE:
- Set detection_signature to an EXACT string or distinctive marker \
that the Validator node can search for in the HTTP response.
- For time-based SQLi: the signature can be empty (the Validator \
checks elapsed_ms, not body content). Set it to "" or to the SLEEP \
command string.
- For error-based SQLi: the signature should be a database error \
keyword (e.g. "SQL syntax", "ORA-", "SQLSTATE").
- For XSS: the signature should be the exact payload string (e.g. \
"<script>alert(1)</script>").
- For SSRF: the signature should be a string expected in the internal \
service's response (e.g. "ami-id" for AWS metadata).
- For command injection: the signature should be a string from the \
command output (e.g. "root:" for /etc/passwd, "uid=" for id).
- For path traversal: the signature should be "root:" (/etc/passwd) \
or "[fonts]" (win.ini).
- For open redirect: the signature should be the redirect target URL.
- For SSTI: the signature should be "49" (7*7=49).

TRANSPORT SELECTION:
- Use HTTP_REQUEST for most vulnerabilities (GET/POST with the payload \
in the query string, body, or header).
- Use GRAPHQL_QUERY only for GraphQL-specific vulnerabilities \
(introspection, batching, injection in GraphQL variables).

OUTPUT REQUIREMENTS:
- Return a single Payload object matching the schema.
- Set is_optimized to False (the Payload Optimizer node will refine \
it later if needed).
- Set http_method to the most appropriate method for the injection \
point (GET for query params, POST for body params).
- Include 1-3 encoded_variants if the payload might need URL-encoding, \
base64, or double-encoding to bypass naive filters.

{format_instructions}

Output ONLY the JSON object. Do not wrap it in markdown code fences. \
Do not add commentary before or after the JSON.
"""


_USER_PROMPT_TEMPLATE: str = """\
Craft a benign PoC payload for the following vulnerability hypothesis.

=== HYPOTHESIS ===
ID: {hypothesis_id}
Category: {category}
Target URL: {target_url}
Target Parameter: {target_parameter}
Analyst Reasoning: {reasoning}
Analyst Evidence: {evidence}
Prerequisites: {prerequisites}
Analyst Confidence: {analyst_confidence}

=== INJECTION CONTEXT ===
HTTP Method (suggested): {suggested_method}
Injection Point (suggested): {suggested_injection_point}

=== INSTRUCTIONS ===
1. Craft a SINGLE benign PoC payload that would prove this hypothesis \
if the vulnerability exists.
2. Set the target_url to exactly: {target_url}
3. Set the hypothesis_id to exactly: {hypothesis_id}
4. Set is_optimized to false.
5. Set transport to "http_request" or "graphql_query".
6. Set detection_signature to the exact string the Validator should \
look for in the response.
7. Set expected_behavior to describe what a successful exploitation \
would look like (e.g. "Response delayed by 10 seconds", "Response \
body contains <script>alert(1)</script>").

Return the Payload JSON object now.
"""


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


async def generate_payloads(state: AppState) -> dict[str, Any]:
    """LangGraph Node 9: generate benign PoC payloads for hypotheses
    that don't already have one.

    Parameters
    ----------
    state:
        The current :class:`~src.shared.state.AppState`. Must contain
        ``hypotheses`` (list of :class:`Hypothesis`). ``payloads`` is
        optional (defaults to empty list if missing).

    Returns
    -------
    dict
        ``{"payloads": [<Payload>, ...]}`` — a list of zero or more
        newly-generated payloads. Hypotheses that already have a
        payload are skipped; hypotheses whose payload generation
        failed are also skipped (with a logged warning).

    Raises
    ------
    PentestFrameworkError
        If ``hypotheses`` is missing or None.
    """
    log = get_logger("payload_generator")

    # ---------------------------------------------------------------
    # 1. Read + validate inputs.
    # ---------------------------------------------------------------
    hypotheses: list[Hypothesis] | None = state.get("hypotheses")
    if hypotheses is None:
        raise PentestFrameworkError(
            "Payload Generator cannot run: state['hypotheses'] is missing. "
            "The Hypothesis Analyzer must run before the Payload Generator.",
            details={
                "available_keys": list(state.keys()),
                "has_hypotheses": "hypotheses" in state,
            },
        )

    existing_payloads: list[Payload] = state.get("payloads") or []

    # ---------------------------------------------------------------
    # 2. Find hypotheses that don't have a payload yet.
    # ---------------------------------------------------------------
    existing_hypothesis_ids: set[str] = {
        p.hypothesis_id for p in existing_payloads
    }
    new_hypotheses: list[Hypothesis] = [
        h for h in hypotheses if h.id not in existing_hypothesis_ids
    ]

    log.info(
        "payload_generation_started",
        total_hypotheses=len(hypotheses),
        hypotheses_with_payload=len(existing_hypothesis_ids),
        hypotheses_to_process=len(new_hypotheses),
    )

    if not new_hypotheses:
        log.info("payload_generation_skipped", reason="no_new_hypotheses")
        return {"payloads": []}

    # ---------------------------------------------------------------
    # 3. Generate one payload per new hypothesis.
    # ---------------------------------------------------------------
    generated_payloads: list[Payload] = []
    success_count: int = 0
    failure_count: int = 0

    for hypothesis in new_hypotheses:
        # Bind hypothesis context for this iteration's log lines.
        iter_log = log.bind(hypothesis_id=hypothesis.id)

        try:
            payload: Payload = await _generate_single_payload(hypothesis, iter_log)
            generated_payloads.append(payload)
            success_count += 1
            iter_log.info(
                "payload_generated",
                category=hypothesis.category.value,
                transport=payload.transport.value,
                method=payload.http_method.value if payload.http_method else "N/A",
                has_detection_signature=bool(payload.detection_signature),
            )
        except (LLMOutputParsingError, LLMRateLimitError) as exc:
            # Classified LLM errors — log and continue to the next
            # hypothesis. Per the Architect's spec, a single failure
            # must not crash the entire node.
            failure_count += 1
            iter_log.warning(
                "payload_generation_failed",
                error_type=type(exc).__name__,
                error_message=str(exc)[:200],
                category=hypothesis.category.value,
            )
        except Exception as exc:
            # Unexpected error — log with traceback and continue.
            failure_count += 1
            iter_log.exception(
                "payload_generation_unexpected_error",
                error_type=type(exc).__name__,
                category=hypothesis.category.value,
            )

    # ---------------------------------------------------------------
    # 4. Log summary and return.
    # ---------------------------------------------------------------
    log.info(
        "payload_generation_complete",
        payloads_generated=success_count,
        payloads_failed=failure_count,
        total_hypotheses_processed=len(new_hypotheses),
    )

    result: dict[str, Any] = {"payloads": generated_payloads}

    # ---------------------------------------------------------------
    # 5. CRITICAL FIX: set active_payload_id so the Orchestrator's
    #    Rule 4 (_check_execution_needed) can detect freshly generated
    #    payloads. Without this, payloads were generated but never
    #    routed to the Execution Sandbox.
    # ---------------------------------------------------------------
    if generated_payloads:
        result["active_payload_id"] = generated_payloads[0].id

    return result


# ---------------------------------------------------------------------------
# Internal: single-payload generation
# ---------------------------------------------------------------------------


async def _generate_single_payload(
    hypothesis: Hypothesis, log: Any
) -> Payload:
    """Generate a single :class:`Payload` for the given hypothesis.

    This function wraps the LLM call, output parsing, and
    defense-in-depth post-processing for ONE hypothesis. Errors are
    raised (not caught) so the caller can log them and continue to
    the next hypothesis.

    Parameters
    ----------
    hypothesis:
        The hypothesis to generate a payload for.
    log:
        The bound structlog logger (with ``hypothesis_id`` already
        bound by the caller).

    Returns
    -------
    Payload
        A fully-populated, defense-in-depth-corrected Payload.

    Raises
    ------
    LLMOutputParsingError
        If the LLM output cannot be parsed into :class:`Payload`.
    LLMRateLimitError
        If the DeepSeek API returns 429 / quota-exceeded.
    """
    # --- Build the structured-output LLM ---
    # V3 supports with_structured_output() natively — no PydanticOutputParser
    # or <think> tag preprocessing needed (unlike R1).
    structured_llm = deepseek_v3.with_structured_output(Payload)

    user_prompt: str = _build_user_prompt(hypothesis)

    # --- Invoke the LLM ---
    try:
        payload: Payload = await structured_llm.ainvoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as exc:
        exc_repr = repr(exc)
        exc_str = str(exc)
        haystack = f"{exc_repr} {exc_str}"

        if _is_rate_limit_error(haystack):
            raise LLMRateLimitError(
                f"Payload Generator LLM rate limited for hypothesis "
                f"'{hypothesis.id}': {exc}",
                provider="deepseek",
                model="deepseek-chat",
                retry_after_seconds=_extract_retry_after(exc),
            ) from exc

        # Non-rate-limit error during the LLM call itself (network,
        # auth, etc.). Wrap in LLMOutputParsingError.
        raise LLMOutputParsingError(
            f"Payload Generator LLM call failed for hypothesis "
            f"'{hypothesis.id}': {exc}",
            raw_output=f"(exception: {exc_repr})",
            schema_name="Payload",
            details={
                "hypothesis_id": hypothesis.id,
                "exception_type": type(exc).__name__,
                "exception_repr": exc_repr[:500],
            },
        ) from exc

    # --- Defense-in-depth: enforce payload fields ---
    payload = _enforce_payload_fields(payload, hypothesis, log)

    return payload


# ---------------------------------------------------------------------------
# Internal: user prompt construction
# ---------------------------------------------------------------------------


def _build_user_prompt(hypothesis: Hypothesis) -> str:
    """Build the user prompt for a single hypothesis.

    The prompt provides the hypothesis details and instructs the LLM
    to craft a benign PoC payload. The ``hypothesis_id`` and
    ``target_url`` are explicitly stated so the LLM can include them
    in the output — but we enforce them post-parse anyway
    (defense-in-depth).
    """
    # Format target_parameter for display.
    if hypothesis.target_parameter is not None:
        tp = hypothesis.target_parameter
        target_param_str: str = (
            f"name={tp.name}, location={tp.location.value}"
            + (f", type={tp.param_type}" if tp.param_type else "")
        )
        suggested_injection: str = target_param_str
    else:
        target_param_str = "(none — URL-level hypothesis)"
        suggested_injection = "(URL-level — inject in path or query string)"

    # Suggest an HTTP method based on the parameter location.
    if hypothesis.target_parameter is not None:
        loc = hypothesis.target_parameter.location
        if loc.value in ("body_form", "body_json", "body_xml", "body_multipart"):
            suggested_method: str = "POST"
        else:
            suggested_method = "GET"
    else:
        suggested_method = "GET"

    # Format evidence and prerequisites lists.
    evidence_str: str = (
        "; ".join(hypothesis.evidence) if hypothesis.evidence else "(none)"
    )
    prerequisites_str: str = (
        "; ".join(hypothesis.prerequisites) if hypothesis.prerequisites else "(none)"
    )

    return _USER_PROMPT_TEMPLATE.format(
        hypothesis_id=hypothesis.id,
        category=hypothesis.category.value,
        target_url=str(hypothesis.target_url),
        target_parameter=target_param_str,
        reasoning=hypothesis.reasoning,
        evidence=evidence_str,
        prerequisites=prerequisites_str,
        analyst_confidence=hypothesis.confidence,
        suggested_method=suggested_method,
        suggested_injection_point=suggested_injection,
    )


# ---------------------------------------------------------------------------
# Internal: defense-in-depth — payload field enforcement
# ---------------------------------------------------------------------------


def _enforce_payload_fields(
    payload: Payload, hypothesis: Hypothesis, log: Any
) -> Payload:
    """Forcibly correct payload fields to match the hypothesis.

    The LLM is told the ``hypothesis_id``, ``target_url``, and
    ``injection_point`` in the prompt, but we do NOT trust it to echo
    them correctly. This helper:

    1. Sets ``hypothesis_id`` to ``hypothesis.id``.
    2. Sets ``is_optimized`` to ``False`` (only the Payload Optimizer
       may set this to ``True``).
    3. Sets ``target_url`` to ``hypothesis.target_url``.
    4. Sets ``injection_point`` to ``hypothesis.target_parameter``.
    5. Validates ``transport`` — if it's WEBSOCKET or PLAYWRIGHT_DOM,
       resets to HTTP_REQUEST with a warning.

    If any field needed correction, a new frozen ``Payload`` is
    returned via ``model_copy(update={...})``. If no corrections were
    needed, the original instance is returned unchanged.
    """
    corrections: dict[str, Any] = {}

    # 1. hypothesis_id
    if payload.hypothesis_id != hypothesis.id:
        log.warning(
            "payload_field_corrected",
            field="hypothesis_id",
            original_value=payload.hypothesis_id,
            corrected_value=hypothesis.id,
        )
        corrections["hypothesis_id"] = hypothesis.id

    # 2. is_optimized — must always be False from this node.
    if payload.is_optimized is not False:
        log.warning(
            "payload_field_corrected",
            field="is_optimized",
            original_value=payload.is_optimized,
            corrected_value=False,
        )
        corrections["is_optimized"] = False

    # 3. target_url — must match the hypothesis.
    if str(payload.target_url) != str(hypothesis.target_url):
        log.warning(
            "payload_field_corrected",
            field="target_url",
            original_value=str(payload.target_url),
            corrected_value=str(hypothesis.target_url),
        )
        corrections["target_url"] = hypothesis.target_url

    # 4. injection_point — must match the hypothesis's target_parameter.
    if payload.injection_point is not hypothesis.target_parameter:
        # We compare by reference first (fast path). If they differ,
        # we check if they're structurally equivalent. If not, we
        # correct to the hypothesis's value.
        if not _params_equal(payload.injection_point, hypothesis.target_parameter):
            log.warning(
                "payload_field_corrected",
                field="injection_point",
                original_value=(
                    payload.injection_point.name
                    if payload.injection_point
                    else "None"
                ),
                corrected_value=(
                    hypothesis.target_parameter.name
                    if hypothesis.target_parameter
                    else "None"
                ),
            )
            corrections["injection_point"] = hypothesis.target_parameter

    # 5. transport — must be HTTP_REQUEST or GRAPHQL_QUERY.
    if payload.transport not in (PayloadTransport.HTTP_REQUEST, PayloadTransport.GRAPHQL_QUERY):
        log.warning(
            "payload_field_corrected",
            field="transport",
            original_value=payload.transport.value,
            corrected_value=PayloadTransport.HTTP_REQUEST.value,
        )
        corrections["transport"] = PayloadTransport.HTTP_REQUEST

    if corrections:
        log.info(
            "payload_corrections_applied",
            corrected_fields=list(corrections.keys()),
            correction_count=len(corrections),
        )
        return payload.model_copy(update=corrections)

    return payload


def _params_equal(a: Any, b: Any) -> bool:
    """Return True if two Parameter objects are structurally equal.

    Handles the case where both are None, or where both have the same
    name and location (the fields that matter for injection-point
    matching).
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.name == b.name and a.location == b.location


# ---------------------------------------------------------------------------
# Internal: error classification (duplicated from validator.py)
# ---------------------------------------------------------------------------


def _is_rate_limit_error(haystack: str) -> bool:
    """Return True if the lowercased exception text indicates a rate
    limit or quota-exceeded error from the LLM provider.

    Duplicated from :func:`src.agents.validator._is_rate_limit_error`
    to keep agent modules independent.
    """
    if not haystack:
        return False
    normalized = haystack.lower()
    rate_limit_indicators = (
        "429",
        "rate limit",
        "rate_limit",
        "ratelimit",
        "quota",
        "quota exceeded",
        "resourceexhausted",
        "resource_exhausted",
        "resource exhausted",
        "too many requests",
        "throttled",
        "throttling",
        "retry-after",
        "retry_after",
        "service unavailable",
    )
    return any(indicator in normalized for indicator in rate_limit_indicators)


def _extract_retry_after(exc: Exception) -> float | None:
    """Best-effort extraction of a ``Retry-After`` value (in seconds).

    Duplicated from :func:`src.agents.validator._extract_retry_after`.
    """
    import re

    for attr_name in ("retry_after", "retry_after_seconds", "retry_after_secs"):
        candidate = getattr(exc, attr_name, None)
        if isinstance(candidate, (int, float)) and candidate > 0:
            return float(candidate)

    msg = str(exc)
    patterns = (
        r"retry[- ]after[:\s]+(\d+(?:\.\d+)?)",
        r"retry[:\s]+in[:\s]+(\d+(?:\.\d+)?)\s*seconds?",
        r"try[:\s]+again[:\s]+in[:\s]+(\d+(?:\.\d+)?)\s*seconds?",
    )
    for pattern in patterns:
        match = re.search(pattern, msg, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue

    return None


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = ["generate_payloads"]