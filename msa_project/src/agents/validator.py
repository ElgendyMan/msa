"""
src/agents/validator.py
=======================

Node 12 of the 16-node LangGraph framework: the **Validator** — the
"Judge" and Zero-False-Positive Engine.

This is the most critical LLM node in the framework. Every confirmed
finding in the final report passes through here. A false positive
slipping through undermines the entire framework's value; a true
positive being rejected wastes the Payload Generator's work.

Graceful Degradation
--------------------
When ALL LLM providers are rate-limited or unavailable, this node
falls back to a pure-Python rule-based validator that checks
detection signatures, HTTP status codes, and elapsed times — no LLM
needed. The pipeline continues instead of crashing.

Parallel Validation (Optimization D)
------------------------------------
When there are multiple pending execution results (those without a
validation report), this node validates them ALL concurrently via
asyncio.gather.

Finding Creation
----------------
When the verdict is TRUE_POSITIVE (with sufficient confidence), this
node creates a Finding object and returns it in the state update.
This is the missing link that was causing 0 findings in the report.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from src.shared.config import settings
from src.shared.exceptions import (
    LLMOutputParsingError,
    LLMRateLimitError,
    PentestFrameworkError,
    ValidationInconclusiveError,
)
from src.shared.llm import deepseek_r1
from src.shared.logging import get_logger
from src.shared.schemas import (
    ExecutionResult,
    Finding,
    Hypothesis,
    Payload,
    SeverityLevel,
    ValidationReport,
    ValidationVerdict,
)
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_TEMPLATE: str = """\
You are a Principal Security Auditor. Your job is to definitively prove \
or disprove a vulnerability hypothesis based on the hard evidence of an \
HTTP request and its response.

You must think step-by-step:
1. Examine the hypothesis: what vulnerability category is claimed, and \
what was the analyst's reasoning?
2. Examine the payload: what was sent, what was the expected behavior, \
and what detection signature should we look for?
3. Examine the response: what HTTP status, headers, body, and elapsed \
time came back?
4. Compare expected vs actual behavior. Does the response match what \
a successful exploitation would look like?
5. Check for the detection signature in the response body or headers.
6. Arrive at a definitive verdict.

VALIDATION CRITERIA BY VULNERABILITY TYPE:

XSS (Reflected):
- TRUE_POSITIVE if the payload appears in the response body EXACTLY as \
sent, WITHOUT HTML encoding or sanitization.
- FALSE_POSITIVE if the payload is reflected but encoded/escaped.

SQL Injection (Time-based):
- TRUE_POSITIVE if elapsed_ms >= expected_delay * 900.
- FALSE_POSITIVE if elapsed_ms < expected_delay * 200.

SQL Injection (Error-based):
- TRUE_POSITIVE if the response body contains a database-specific error \
message (e.g. "SQL syntax", "ORA-", "SQLSTATE").

Command Injection:
- TRUE_POSITIVE if the response body contains output from command \
execution (e.g. "uid=0(root)").

Path Traversal:
- TRUE_POSITIVE if the response body contains the contents of the \
targeted file (e.g. /etc/passwd entries).

Open Redirect:
- TRUE_POSITIVE if the response status is 301/302/303/307/308 AND the \
Location header contains the attacker-controlled URL.

GENERAL RULES:
1. NEVER mark TRUE_POSITIVE based solely on a 200 OK status.
2. NEVER mark TRUE_POSITIVE if the execution failed (error field is not null).
3. If confidence < 0.6, return INCONCLUSIVE.
4. Confidence must be in [0.0, 1.0].

OUTPUT FORMAT:
{format_instructions}

Output ONLY the JSON object. Do not wrap it in markdown code fences. \
If you need to reason, do so in a <think>...</think> block BEFORE the JSON.
"""


_USER_PROMPT_TEMPLATE: str = """\
Validate the following vulnerability hypothesis against the execution \
evidence. Think step-by-step, then output the JSON ValidationReport.

=== HYPOTHESIS ===
ID: {hypothesis_id}
Category: {category}
Target URL: {target_url}
Target Parameter: {target_parameter}
Analyst Reasoning: {reasoning}
Analyst Confidence: {analyst_confidence}

=== PAYLOAD ===
ID: {payload_id}
Raw Payload: {raw_payload}
Expected Behavior: {expected_behavior}
Detection Signature: {detection_signature}
HTTP Method: {http_method}
Injection Point: {injection_point}

=== EXECUTION RESULT ===
Payload ID: {payload_id}
Error: {execution_error}

Request:
  Method: {request_method}
  URL: {request_url}
  Headers: {request_headers}
  Body: {request_body}

Response:
  Status Code: {response_status_code}
  Elapsed (ms): {response_elapsed_ms}
  Headers: {response_headers}
  Body: {response_body}

Return the JSON object now.
"""


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


async def validate_execution(state: AppState) -> dict[str, Any]:
    """LangGraph Node 12: validate execution results.

    Supports PARALLEL validation and GRACEFUL DEGRADATION when LLM
    providers are unavailable.

    Returns
    -------
    dict
        ``{"validation_reports": [...], "confirmed_findings": [...]}``
    """
    log = get_logger("validator")

    # ---------------------------------------------------------------
    # 1. Resolve execution_results → payloads → hypotheses.
    # ---------------------------------------------------------------
    execution_results: list[ExecutionResult] | None = state.get("execution_results")
    if not execution_results:
        raise PentestFrameworkError(
            "Validator cannot run: state['execution_results'] is missing or empty.",
            details={"available_keys": list(state.keys())},
        )

    payloads: list[Payload] | None = state.get("payloads")
    if not payloads:
        raise PentestFrameworkError(
            "Validator cannot run: state['payloads'] is missing or empty.",
            details={"available_keys": list(state.keys())},
        )

    hypotheses: list[Hypothesis] | None = state.get("hypotheses")
    if not hypotheses:
        raise PentestFrameworkError(
            "Validator cannot run: state['hypotheses'] is missing or empty.",
            details={"available_keys": list(state.keys())},
        )

    # Find execution results that don't have a validation report yet.
    existing_reports: list[ValidationReport] | None = state.get("validation_reports")
    validated_payload_ids: set[str] = (
        {r.payload_id for r in existing_reports} if existing_reports else set()
    )
    pending_results: list[ExecutionResult] = [
        er for er in execution_results
        if er.payload_id not in validated_payload_ids
    ]

    if not pending_results:
        log.info("validation_skipped_all_validated")
        return {"validation_reports": []}

    # Build (execution_result, payload, hypothesis) triples.
    triples: list[tuple[ExecutionResult, Payload, Hypothesis]] = []
    for er in pending_results:
        payload: Payload | None = next(
            (p for p in payloads if p.id == er.payload_id), None
        )
        if payload is None:
            log.warning("payload_not_found_for_execution", payload_id=er.payload_id)
            continue
        hypothesis: Hypothesis | None = next(
            (h for h in hypotheses if h.id == payload.hypothesis_id), None
        )
        if hypothesis is None:
            log.warning("hypothesis_not_found_for_payload",
                        hypothesis_id=payload.hypothesis_id)
            continue
        triples.append((er, payload, hypothesis))

    if not triples:
        log.warning("no_validatable_triples")
        return {"validation_reports": []}

    log.info("validation_started", pending_count=len(triples),
             parallel=len(triples) > 1)

    # ---------------------------------------------------------------
    # 2. Validate each triple — PARALLEL if >1, SINGLE if 1.
    # ---------------------------------------------------------------
    if len(triples) == 1:
        er, p, h = triples[0]
        report, finding = await _validate_single(er, p, h, log)
        result: dict[str, Any] = {"validation_reports": [report]}
        if finding is not None:
            result["confirmed_findings"] = [finding]
        return result

    # Multiple validations — run in PARALLEL.
    log.info("parallel_validation_started", count=len(triples))

    async def _validate_one(
        er: ExecutionResult, p: Payload, h: Hypothesis
    ) -> tuple[ValidationReport, Finding | None]:
        v_log = log.bind(payload_id=p.id, hypothesis_id=h.id)
        return await _validate_single(er, p, h, v_log)

    validation_outputs: list[tuple[ValidationReport, Finding | None]] = (
        await asyncio.gather(*[_validate_one(er, p, h) for er, p, h in triples])
    )

    all_reports: list[ValidationReport] = []
    all_findings: list[Finding] = []
    inconclusive_payload_id: str | None = None
    for report, finding in validation_outputs:
        all_reports.append(report)
        if finding is not None:
            all_findings.append(finding)
        # Track the FIRST INCONCLUSIVE payload so the optimizer knows
        # which one to optimize. The orchestrator's Rule 6 checks for
        # ANY INCONCLUSIVE report; the validator sets active_payload_id
        # to the inconclusive payload so the optimizer can find it.
        if report.verdict == ValidationVerdict.INCONCLUSIVE and inconclusive_payload_id is None:
            inconclusive_payload_id = report.payload_id

    log.info("parallel_validation_complete",
             count=len(all_reports), findings_created=len(all_findings),
             inconclusive_payload_id=inconclusive_payload_id)

    result: dict[str, Any] = {"validation_reports": all_reports}
    if all_findings:
        result["confirmed_findings"] = all_findings
    # CRITICAL: Set active_payload_id to the INCONCLUSIVE payload so
    # the Payload Optimizer knows which payload to optimize.
    if inconclusive_payload_id is not None:
        result["active_payload_id"] = inconclusive_payload_id
        log.info("active_payload_id_set_to_inconclusive",
                 active_payload_id=inconclusive_payload_id,
                 message="Routing to payload_optimizer for INCONCLUSIVE payload.")
    return result


# ---------------------------------------------------------------------------
# Internal: single-triple validation
# ---------------------------------------------------------------------------


async def _validate_single(
    execution_result: ExecutionResult,
    payload: Payload,
    hypothesis: Hypothesis,
    log: Any,
) -> tuple[ValidationReport, Finding | None]:
    """Validate a single triple. Returns (report, finding_or_None)."""

    log.info(
        "validation_started",
        category=hypothesis.category.value,
        execution_has_response=execution_result.response is not None,
        execution_error=execution_result.error,
    )

    # Build the parser and prompts.
    parser: PydanticOutputParser = PydanticOutputParser(
        pydantic_object=ValidationReport
    )
    system_prompt: str = _SYSTEM_PROMPT_TEMPLATE.format(
        format_instructions=parser.get_format_instructions()
    )
    user_prompt: str = _build_user_prompt(hypothesis, payload, execution_result)

    # Invoke the LLM.
    try:
        response_message = await deepseek_r1.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except Exception as exc:
        exc_repr = repr(exc)
        exc_str = str(exc)
        haystack = f"{exc_repr} {exc_str}"

        if _is_rate_limit_error(haystack):
            # GRACEFUL DEGRADATION: rule-based fallback.
            log.warning(
                "llm_rate_limited_using_rule_based_fallback",
                error_type=type(exc).__name__,
                error_message=exc_str[:200],
            )
            report = _rule_based_fallback_validation(
                hypothesis, payload, execution_result, log
            )
            finding = None
            if report.verdict == ValidationVerdict.TRUE_POSITIVE:
                finding = _create_finding_from_validation(
                    hypothesis, payload, execution_result, report
                )
                log.info("finding_created", finding_id=finding.id,
                         message="TRUE_POSITIVE (rule-based fallback)")
            return report, finding

        # Unexpected error — also use fallback.
        log.exception("llm_call_unexpected_error", error_type=type(exc).__name__)
        log.warning("llm_unexpected_error_using_rule_based_fallback",
                    error_type=type(exc).__name__)
        report = _rule_based_fallback_validation(
            hypothesis, payload, execution_result, log
        )
        finding = None
        if report.verdict == ValidationVerdict.TRUE_POSITIVE:
            finding = _create_finding_from_validation(
                hypothesis, payload, execution_result, report
            )
        return report, finding

    raw_llm_output: str = response_message.content

    # Parse the LLM output.
    report: ValidationReport | None = None
    try:
        report = await parser.aparse(raw_llm_output)
    except (OutputParserException, ValidationError):
        log.debug("parser_direct_failed_trying_preprocessed")

    if report is None:
        cleaned: str = _preprocess_llm_output(raw_llm_output)
        try:
            report = ValidationReport.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            log.warning("llm_output_parse_failed", error_type=type(exc).__name__)
            raise LLMOutputParsingError(
                f"Validator LLM output could not be parsed: {exc}",
                raw_output=raw_llm_output,
                schema_name="ValidationReport",
                details={"cleaned_preview": cleaned[:500]},
            ) from exc

    # Enforce identity fields.
    if report.payload_id != payload.id or report.hypothesis_id != hypothesis.id:
        log.warning("identity_fields_corrected",
                     llm_payload_id=report.payload_id,
                     expected_payload_id=payload.id)
        report = report.model_copy(update={
            "payload_id": payload.id,
            "hypothesis_id": hypothesis.id,
        })

    # Check verdict + confidence threshold.
    # CRITICAL: NEVER raise — return the report. The Orchestrator's
    # rule 6 routes INCONCLUSIVE to payload_optimizer.
    threshold: float = settings.VALIDATION_CONFIDENCE_THRESHOLD

    if report.verdict == ValidationVerdict.INCONCLUSIVE:
        log.info("validation_inconclusive", verdict=report.verdict.value,
                 confidence=report.confidence, payload_id=payload.id)
    elif report.confidence < threshold:
        original_verdict: ValidationVerdict = report.verdict
        log.info("validation_demoted_to_inconclusive",
                 original_verdict=original_verdict.value,
                 confidence=report.confidence, threshold=threshold)
        report = report.model_copy(update={
            "verdict": ValidationVerdict.INCONCLUSIVE,
            "reasoning": f"[DEMOTED from {original_verdict.value} due to low "
                        f"confidence {report.confidence} < {threshold}] "
                        f"{report.reasoning}",
        })
    else:
        log.info("validation_complete", verdict=report.verdict.value,
                 confidence=report.confidence,
                 matched_signatures=report.matched_signatures)

    # Create a Finding if TRUE_POSITIVE.
    finding: Finding | None = None
    if report.verdict == ValidationVerdict.TRUE_POSITIVE:
        finding = _create_finding_from_validation(
            hypothesis, payload, execution_result, report
        )
        log.info("finding_created", finding_id=finding.id,
                 title=finding.title, severity=finding.severity.value,
                 message="TRUE_POSITIVE confirmed; Finding created.")

    return report, finding


# ---------------------------------------------------------------------------
# Internal: Finding creation
# ---------------------------------------------------------------------------


def _create_finding_from_validation(
    hypothesis: Hypothesis,
    payload: Payload,
    execution_result: ExecutionResult,
    report: ValidationReport,
) -> Finding:
    """Construct a Finding from a confirmed TRUE_POSITIVE."""
    category_label: str = hypothesis.category.value.upper().replace("_", " ")
    target_url_str: str = str(hypothesis.target_url)
    if len(target_url_str) > 80:
        title: str = f"{category_label} in {target_url_str[:77]}..."
    else:
        title = f"{category_label} in {target_url_str}"

    matched_sigs: str = (
        ", ".join(report.matched_signatures) if report.matched_signatures else "(none)"
    )
    method_str: str = payload.http_method.value if payload.http_method else "GET"
    poc_lines: list[str] = [
        f"Vulnerability: {category_label}",
        f"Target: {target_url_str}",
        f"Method: {method_str}",
        f"Payload: {payload.raw}",
        f"Detection Signatures Matched: {matched_sigs}",
        f"Validator Confidence: {report.confidence}",
        f"Validator Reasoning: {report.reasoning}",
    ]
    if payload.injection_point is not None:
        poc_lines.insert(3, f"Injection Point: {payload.injection_point.name} "
                            f"({payload.injection_point.location.value})")

    return Finding(
        hypothesis_id=hypothesis.id,
        payload_id=payload.id,
        validation_id=report.id,
        category=hypothesis.category,
        title=title,
        target_url=hypothesis.target_url,
        target_parameter=hypothesis.target_parameter,
        severity=SeverityLevel.INFO,
        cvss=None,
        business_impact=None,
        proof_of_concept="\n".join(poc_lines),
        raw_request=execution_result.request,
        raw_response=execution_result.response,
        rag_references=[],
        tags=[hypothesis.category.value],
    )


# ---------------------------------------------------------------------------
# Internal: user prompt construction
# ---------------------------------------------------------------------------


def _build_user_prompt(
    hypothesis: Hypothesis,
    payload: Payload,
    execution_result: ExecutionResult,
) -> str:
    """Build the user prompt with all evidence formatted for the LLM."""
    target_param_str: str = "None"
    if hypothesis.target_parameter is not None:
        tp = hypothesis.target_parameter
        target_param_str = f"name={tp.name}, location={tp.location.value}"

    injection_point_str: str = "None"
    if payload.injection_point is not None:
        ip = payload.injection_point
        injection_point_str = f"name={ip.name}, location={ip.location.value}"

    req = execution_result.request
    resp = execution_result.response

    request_headers_str: str = _format_headers(req.headers)
    request_body_str: str = _format_body(req.body, req.body_bytes_b64)

    if resp is not None:
        response_status_code: str = str(resp.status_code)
        response_elapsed_ms: str = str(resp.elapsed_ms)
        response_headers_str: str = _format_headers(resp.headers)
        response_body_str: str = _format_body(resp.body, resp.body_bytes_b64)
    else:
        response_status_code = "(no response)"
        response_elapsed_ms = "(no response)"
        response_headers_str = "(no response)"
        response_body_str = "(no response)"

    return _USER_PROMPT_TEMPLATE.format(
        hypothesis_id=hypothesis.id,
        category=hypothesis.category.value,
        target_url=str(hypothesis.target_url),
        target_parameter=target_param_str,
        reasoning=hypothesis.reasoning,
        analyst_confidence=hypothesis.confidence,
        payload_id=payload.id,
        raw_payload=payload.raw,
        expected_behavior=payload.expected_behavior,
        detection_signature=payload.detection_signature,
        http_method=payload.http_method.value if payload.http_method else "N/A",
        injection_point=injection_point_str,
        execution_error=execution_result.error or "None",
        request_method=req.method.value,
        request_url=str(req.url),
        request_headers=request_headers_str,
        request_body=request_body_str,
        response_status_code=response_status_code,
        response_elapsed_ms=response_elapsed_ms,
        response_headers=response_headers_str,
        response_body=response_body_str,
    )


def _format_headers(headers: dict[str, str]) -> str:
    if not headers:
        return "(none)"
    lines: list[str] = []
    for k, v in headers.items():
        val_display = v[:500] if len(v) > 500 else v
        if len(v) > 500:
            val_display += "... (truncated)"
        lines.append(f"    {k}: {val_display}")
    return "\n".join(lines)


def _format_body(body: str | None, body_b64: str | None) -> str:
    max_chars: int = 4096
    if body is not None:
        if len(body) > max_chars:
            return body[:max_chars] + f"\n... (truncated, total {len(body)} chars)"
        return body
    if body_b64 is not None:
        try:
            decoded_bytes: bytes = base64.b64decode(body_b64)
            try:
                decoded_str = decoded_bytes.decode("utf-8")
                if len(decoded_str) > max_chars:
                    return decoded_str[:max_chars] + f"\n... (truncated)"
                return decoded_str
            except UnicodeDecodeError:
                hex_preview = decoded_bytes[:256].hex(" ")
                return f"(binary body, {len(decoded_bytes)} bytes, hex: {hex_preview})"
        except Exception:
            return f"(base64 body, decode failed)"
    return "(empty)"


# ---------------------------------------------------------------------------
# Internal: R1 output preprocessing
# ---------------------------------------------------------------------------


def _preprocess_llm_output(text: str) -> str:
    """Preprocess R1's output to extract JSON."""
    # 1. Remove <think>...</think> blocks (closed + unclosed).
    cleaned: str = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    cleaned = re.sub(
        r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE
    )
    # 2. Remove markdown code fences.
    cleaned = re.sub(
        r"```(?:json)?\s*(.*?)\s*```", r"\1", cleaned,
        flags=re.DOTALL | re.IGNORECASE
    )
    cleaned = cleaned.strip("`").strip()
    # 3. Extract JSON substring.
    if cleaned and not cleaned.startswith("{"):
        first_brace: int = cleaned.find("{")
        last_brace: int = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace : last_brace + 1]
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Internal: error classification
# ---------------------------------------------------------------------------


def _is_rate_limit_error(haystack: str) -> bool:
    if not haystack:
        return False
    normalized = haystack.lower()
    indicators = (
        "429", "rate limit", "rate_limit", "ratelimit", "quota",
        "quota exceeded", "resourceexhausted", "resource_exhausted",
        "too many requests", "throttled", "throttling",
        "retry-after", "retry_after", "service unavailable",
        "tokens per day", "tpd",
    )
    return any(ind in normalized for ind in indicators)


def _extract_retry_after(exc: Exception) -> float | None:
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
# Internal: Pure-Python rule-based fallback validator (no LLM)
# ---------------------------------------------------------------------------


def _rule_based_fallback_validation(
    hypothesis: Hypothesis,
    payload: Payload,
    execution_result: ExecutionResult,
    log: Any,
) -> ValidationReport:
    """Validate using pure-Python rules (no LLM).

    Makes a best-effort verdict based on:
    - Detection signature presence in the response body.
    - HTTP status code.
    - Elapsed time (for time-based SQLi).
    - Execution errors.
    """
    import uuid

    log.info("rule_based_validation_started",
             category=hypothesis.category.value, payload_id=payload.id)

    verdict: ValidationVerdict = ValidationVerdict.INCONCLUSIVE
    confidence: float = 0.4
    matched_signatures: list[str] = []
    reasoning: str = "Rule-based fallback (LLM unavailable). "
    supporting_evidence: list[str] = []

    resp = execution_result.response

    if execution_result.error is not None:
        verdict = ValidationVerdict.INCONCLUSIVE
        confidence = 0.3
        reasoning += f"Execution error: {execution_result.error[:200]}"
    elif resp is None:
        verdict = ValidationVerdict.INCONCLUSIVE
        confidence = 0.3
        reasoning += "No response received."
    else:
        body: str = ""
        if resp.body is not None:
            body = resp.body
        elif resp.body_bytes_b64 is not None:
            try:
                body = base64.b64decode(resp.body_bytes_b64).decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                body = ""

        sig: str = payload.detection_signature
        if sig and sig in body:
            matched_signatures.append(sig)
            supporting_evidence.append(f"signature '{sig}' found in body")

            if "xss" in hypothesis.category.value:
                verdict = ValidationVerdict.TRUE_POSITIVE
                confidence = 0.7
                reasoning += "XSS payload reflected verbatim."
            elif "sqli" in hypothesis.category.value and any(
                kw in body.lower() for kw in (
                    "sql syntax", "ora-", "sqlstate", "mysql_", "pg::syntaxerror"
                )
            ):
                verdict = ValidationVerdict.TRUE_POSITIVE
                confidence = 0.7
                reasoning += "Database error detected."
            elif "path_traversal" in hypothesis.category.value:
                verdict = ValidationVerdict.TRUE_POSITIVE
                confidence = 0.7
                reasoning += "File contents detected."
            elif "command_injection" in hypothesis.category.value:
                verdict = ValidationVerdict.TRUE_POSITIVE
                confidence = 0.7
                reasoning += "Command output detected."
            elif "ssti" in hypothesis.category.value:
                verdict = ValidationVerdict.TRUE_POSITIVE
                confidence = 0.7
                reasoning += "Template evaluation result detected."
            else:
                verdict = ValidationVerdict.TRUE_POSITIVE
                confidence = 0.6
                reasoning += "Detection signature found."
        else:
            # Check for time-based SQLi.
            if "sqli_time" in hypothesis.category.value:
                expected_delay_ms: int = 5000
                if "sleep(5)" in payload.raw.lower():
                    expected_delay_ms = 5000
                elif "sleep(10)" in payload.raw.lower():
                    expected_delay_ms = 10000
                if resp.elapsed_ms >= expected_delay_ms * 0.8:
                    verdict = ValidationVerdict.TRUE_POSITIVE
                    confidence = 0.7
                    reasoning += (
                        f"Response delayed {resp.elapsed_ms}ms "
                        f"(expected ~{expected_delay_ms}ms)."
                    )
                else:
                    verdict = ValidationVerdict.FALSE_POSITIVE
                    confidence = 0.6
                    reasoning += f"Fast response ({resp.elapsed_ms}ms)."
            elif "open_redirect" in hypothesis.category.value:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location: str = resp.headers.get("location", "")
                    if payload.raw in location:
                        verdict = ValidationVerdict.TRUE_POSITIVE
                        confidence = 0.7
                        reasoning += f"Redirect to attacker URL (Location={location[:100]})."
                        matched_signatures.append(payload.raw)
                    else:
                        verdict = ValidationVerdict.FALSE_POSITIVE
                        confidence = 0.6
                        reasoning += "Redirect did not go to attacker URL."
                else:
                    verdict = ValidationVerdict.FALSE_POSITIVE
                    confidence = 0.5
                    reasoning += f"No redirect (status={resp.status_code})."
            else:
                verdict = ValidationVerdict.INCONCLUSIVE
                confidence = 0.4
                reasoning += (
                    f"Signature not found (status={resp.status_code}, "
                    f"body_length={len(body)})."
                )

    log.info("rule_based_validation_complete", verdict=verdict.value,
             confidence=confidence, matched_signatures=matched_signatures)

    return ValidationReport(
        id=str(uuid.uuid4()),
        payload_id=payload.id,
        hypothesis_id=hypothesis.id,
        verdict=verdict,
        confidence=confidence,
        matched_signatures=matched_signatures,
        reasoning=reasoning,
        supporting_evidence=supporting_evidence,
    )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = ["validate_execution"]
