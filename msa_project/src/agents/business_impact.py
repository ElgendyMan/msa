"""
src/agents/business_impact.py
=============================

Node 15 of the 16-node LangGraph framework: the **Business Impact Writer**.

Graceful Degradation: When ALL LLM providers are rate-limited, falls
back to a template-based impact assessment — no LLM needed.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from src.shared.exceptions import (
    LLMOutputParsingError,
    LLMRateLimitError,
    PentestFrameworkError,
)
from src.shared.llm import gemini_flash
from src.shared.logging import get_logger
from src.shared.schemas import BusinessImpact, Finding
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT: str = """\
You are a Business Risk Analyst. Write a clear, accurate business impact \
assessment for a confirmed vulnerability finding.

GUIDELINES:
1. The narrative should be 1-3 paragraphs of plain-language prose.
2. Cover financial, operational, legal, and reputational impacts.
3. Be factual — do not exaggerate.
4. Provide specific, actionable remediation steps.

Output ONLY valid JSON matching the BusinessImpact schema.
"""


_USER_PROMPT_TEMPLATE: str = """\
Generate a business impact assessment for the following finding.

=== FINDING ===
Title: {title}
Category: {category}
Severity: {severity}
CVSS Score: {cvss_score}
Target URL: {target_url}

=== PROOF OF CONCEPT ===
{proof_of_concept}

Return the BusinessImpact JSON object now.
"""


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


async def evaluate_impact(state: AppState) -> dict[str, Any]:
    """LangGraph Node 15: evaluate business impact of confirmed findings.

    Only processes findings that are missing a business impact narrative.
    Supports GRACEFUL DEGRADATION when LLM providers are unavailable.
    """
    log = get_logger("business_impact")

    # 1. Read confirmed findings.
    confirmed_findings: list[Finding] | None = state.get("confirmed_findings")
    if not confirmed_findings:
        log.info("business_impact_skipped_no_findings")
        return {"business_impacts": []}

    # Only process findings that are missing a business impact narrative.
    # This prevents re-processing already-enriched findings on each routing cycle.
    pending: list[Finding] = [
        f for f in confirmed_findings if not getattr(f, "business_impact", None)
    ]
    if not pending:
        log.info("business_impact_skipped_all_findings_already_processed")
        return {"business_impacts": []}

    log = log.bind(finding_count=len(pending))
    log.info("business_impact_started")

    # 2. Generate impact for each pending finding.
    impacts: list[BusinessImpact] = []

    for finding in pending:
        try:
            impact: BusinessImpact = await _generate_impact(finding, log)
        except Exception as exc:
            # GRACEFUL DEGRADATION: use template if LLM fails.
            log.warning("impact_generation_failed_using_template",
                        finding_id=finding.id, error_type=type(exc).__name__)
            impact = _template_fallback_impact(finding, log)

        impacts.append(impact)

    log.info("business_impact_complete", impacts_count=len(impacts))

    return {"business_impacts": impacts}


# ---------------------------------------------------------------------------
# Internal: impact generation
# ---------------------------------------------------------------------------


async def _generate_impact(finding: Finding, log: Any) -> BusinessImpact:
    """Generate a business impact assessment for a single finding via LLM."""
    structured_llm = gemini_flash.with_structured_output(BusinessImpact)

    user_prompt: str = _build_finding_prompt(finding)

    try:
        impact: BusinessImpact = await structured_llm.ainvoke(
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
            log.warning(
                "llm_rate_limited_using_template_fallback",
                error_type=type(exc).__name__,
                error_message=exc_str[:200],
                finding_id=finding.id,
            )
            return _template_fallback_impact(finding, log)

        log.exception("llm_call_unexpected_error",
                      error_type=type(exc).__name__, finding_id=finding.id)
        log.warning("llm_unexpected_error_using_template_fallback",
                    error_type=type(exc).__name__, finding_id=finding.id)
        return _template_fallback_impact(finding, log)

    # Enforce finding_id.
    if impact.finding_id != finding.id:
        impact = impact.model_copy(update={"finding_id": finding.id})

    return impact


def _build_finding_prompt(finding: Finding) -> str:
    cat_label: str = finding.category.value.upper().replace("_", " ")
    severity_str: str = finding.severity.value.upper()
    cvss_score: str = f"{finding.cvss.base_score}" if finding.cvss else "N/A"

    return _USER_PROMPT_TEMPLATE.format(
        title=finding.title,
        category=cat_label,
        severity=severity_str,
        cvss_score=cvss_score,
        target_url=str(finding.target_url),
        proof_of_concept=finding.proof_of_concept,
    )


# ---------------------------------------------------------------------------
# Internal: defense-in-depth
# ---------------------------------------------------------------------------


def _enforce_finding_id(
    impact: BusinessImpact, finding: Finding, log: Any
) -> BusinessImpact:
    if impact.finding_id != finding.id:
        log.warning("finding_id_corrected",
                     original=impact.finding_id, expected=finding.id)
        return impact.model_copy(update={"finding_id": finding.id})
    return impact


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


# ---------------------------------------------------------------------------
# Internal: Pure-Python template-based fallback impact (no LLM)
# ---------------------------------------------------------------------------


def _template_fallback_impact(finding: Finding, log: Any) -> BusinessImpact:
    """Generate a business impact assessment using a template (no LLM)."""
    cat_label: str = finding.category.value.upper().replace("_", " ")
    severity_str: str = finding.severity.value.upper()
    cvss_score: float = finding.cvss.base_score if finding.cvss else 0.0

    narrative: str = (
        f"A {cat_label} vulnerability was confirmed at {finding.target_url}. "
        f"The vulnerability has a CVSS base score of {cvss_score} "
        f"(severity: {severity_str}). "
    )

    if "sqli" in finding.category.value:
        narrative += (
            "This SQL Injection vulnerability could allow an attacker to "
            "read, modify, or delete database contents, potentially "
            "exposing sensitive user data and compromising application "
            "integrity."
        )
    elif "xss" in finding.category.value:
        narrative += (
            "This Cross-Site Scripting vulnerability could allow an "
            "attacker to execute malicious scripts in victims' browsers, "
            "potentially stealing session tokens, redirecting users to "
            "malicious sites, or defacing the application."
        )
    elif "command_injection" in finding.category.value:
        narrative += (
            "This Command Injection vulnerability could allow an attacker "
            "to execute arbitrary operating system commands on the server, "
            "potentially leading to full system compromise."
        )
    elif "ssrf" in finding.category.value:
        narrative += (
            "This SSRF vulnerability could allow an attacker to access "
            "internal services, cloud metadata endpoints, or other "
            "network resources not intended for external access."
        )
    elif "path_traversal" in finding.category.value:
        narrative += (
            "This Path Traversal vulnerability could allow an attacker to "
            "read arbitrary files on the server, potentially exposing "
            "configuration files, credentials, or source code."
        )
    else:
        narrative += (
            "This vulnerability could allow an attacker to compromise "
            "the application's security properties."
        )

    financial_impact: str = (
        f"Potential financial impact: {severity_str} severity "
        f"(CVSS {cvss_score})."
    )
    operational_impact: str = (
        f"Potential operational impact: {severity_str} severity — "
        f"may affect application availability or integrity."
    )
    legal_impact: str = "Potential legal/compliance impact if user data is exposed."
    reputational_impact: str = "Potential reputational damage if exploited publicly."

    recommended_mitigation: list[str] = []
    if "sqli" in finding.category.value:
        recommended_mitigation = [
            "Use parameterized queries / prepared statements.",
            "Validate and sanitize all user inputs.",
            "Apply least-privilege database permissions.",
        ]
    elif "xss" in finding.category.value:
        recommended_mitigation = [
            "Apply context-aware output encoding.",
            "Use Content Security Policy (CSP) headers.",
            "Validate and sanitize all user inputs.",
        ]
    elif "command_injection" in finding.category.value:
        recommended_mitigation = [
            "Avoid shell commands; use language-native APIs.",
            "If shell commands are unavoidable, use strict allowlists.",
            "Run the application with least privileges.",
        ]
    else:
        recommended_mitigation = [
            "Implement proper input validation.",
            "Follow OWASP best practices.",
            "Apply defense-in-depth principles.",
        ]

    log.info("template_fallback_impact_generated",
             finding_id=finding.id, narrative_length=len(narrative),
             message="Template-based impact generated (LLM was unavailable).")

    return BusinessImpact(
        finding_id=finding.id,
        narrative=narrative,
        affected_assets=[str(finding.target_url)],
        financial_impact=financial_impact,
        operational_impact=operational_impact,
        legal_impact=legal_impact,
        reputational_impact=reputational_impact,
        recommended_mitigation=recommended_mitigation,
    )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = ["evaluate_impact"]
