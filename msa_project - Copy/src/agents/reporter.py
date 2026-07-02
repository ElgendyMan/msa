"""
src/agents/reporter.py
======================

Node 16 of the 16-node LangGraph framework: the **Reporter**.

Graceful Degradation: When ALL LLM providers are rate-limited, falls
back to a template-based report generator — no LLM needed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.shared.exceptions import (
    LLMOutputParsingError,
    LLMRateLimitError,
    PentestFrameworkError,
)
from src.shared.llm import gemini_pro
from src.shared.logging import get_logger
from src.shared.schemas import Finding
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Private wrapper model for structured output
# ---------------------------------------------------------------------------


class _ReportContent(BaseModel):
    """Private wrapper for the LLM's structured output."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(description="Concise, descriptive finding title.")
    executive_summary: str = Field(description="1-2 paragraph summary for executives.")
    steps_to_reproduce: list[str] = Field(
        default_factory=list,
        description="Numbered steps to reproduce the vulnerability.",
    )
    remediation: list[str] = Field(
        default_factory=list,
        description="Specific, actionable remediation steps.",
    )


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT: str = """\
You are a Lead Security Consultant writing a professional penetration \
testing report. Write clear, accurate, and actionable narrative content \
for a confirmed vulnerability finding.

WRITING GUIDELINES:
1. Write for a mixed audience.
2. Be factual. Do not exaggerate.
3. The title should be concise and descriptive.
4. The executive_summary should explain WHAT, WHERE, and WHY it matters.
5. The steps_to_reproduce should be numbered and reproducible.
6. The remediation should be specific and actionable.

Output ONLY valid JSON with keys: title, executive_summary, steps_to_reproduce, remediation.
"""


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


async def generate_report(state: AppState) -> dict[str, Any]:
    """LangGraph Node 16: generate the final Markdown report.

    Supports GRACEFUL DEGRADATION when LLM providers are unavailable.
    """
    log = get_logger("reporter")

    # 1. Read confirmed findings.
    confirmed_findings: list[Finding] | None = state.get("confirmed_findings")
    if not confirmed_findings:
        log.info("report_skipped_no_findings")
        return {"final_report": "No vulnerabilities confirmed."}

    log = log.bind(finding_count=len(confirmed_findings))
    log.info("report_generation_started")

    # 2. Generate content for each finding.
    all_sections: list[str] = []

    # Add report header.
    all_sections.append("# Penetration Test Report\n")
    all_sections.append(f"**Findings:** {len(confirmed_findings)}\n")
    all_sections.append("---\n")

    for finding in confirmed_findings:
        try:
            content: _ReportContent = await _generate_finding_content(finding, log)
        except Exception as exc:
            # GRACEFUL DEGRADATION: use template if LLM fails.
            log.warning("finding_content_generation_failed_using_template",
                        finding_id=finding.id, error_type=type(exc).__name__)
            content = _template_fallback_report(finding, log)

        # Build the Markdown section for this finding.
        section: list[str] = []
        section.append(f"## {content.title}\n")
        section.append(f"**Severity:** {finding.severity.value.upper()}\n")
        if finding.cvss is not None:
            section.append(f"**CVSS:** {finding.cvss.base_score} "
                           f"({finding.cvss.vector_string})\n")
        section.append(f"\n### Executive Summary\n\n{content.executive_summary}\n")
        if content.steps_to_reproduce:
            section.append("\n### Steps to Reproduce\n")
            for step in content.steps_to_reproduce:
                section.append(f"{step}")
            section.append("")
        section.append(f"\n### Proof of Concept\n\n```\n{finding.proof_of_concept}\n```\n")
        if content.remediation:
            section.append("\n### Remediation\n")
            for rem in content.remediation:
                section.append(f"- {rem}")
            section.append("")
        all_sections.append("\n".join(section))
        all_sections.append("---\n")

    final_report: str = "\n".join(all_sections)

    log.info("report_generation_complete",
             report_length=len(final_report),
             findings_count=len(confirmed_findings))

    return {"final_report": final_report}


# ---------------------------------------------------------------------------
# Internal: finding content generation
# ---------------------------------------------------------------------------


async def _generate_finding_content(finding: Finding, log: Any) -> _ReportContent:
    """Generate report content for a single finding via LLM."""
    structured_llm = gemini_pro.with_structured_output(_ReportContent)

    user_prompt: str = _build_finding_prompt(finding)

    try:
        content: _ReportContent = await structured_llm.ainvoke(
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
            return _template_fallback_report(finding, log)

        log.exception("llm_call_unexpected_error",
                      error_type=type(exc).__name__, finding_id=finding.id)
        log.warning("llm_unexpected_error_using_template_fallback",
                    error_type=type(exc).__name__, finding_id=finding.id)
        return _template_fallback_report(finding, log)

    return content


def _build_finding_prompt(finding: Finding) -> str:
    """Build the user prompt for a single finding."""
    cat_label: str = finding.category.value.upper().replace("_", " ")
    severity_str: str = finding.severity.value.upper()
    cvss_score: str = f"{finding.cvss.base_score}" if finding.cvss else "N/A"

    return f"""Generate report content for the following confirmed vulnerability.

=== FINDING ===
Title: {finding.title}
Category: {cat_label}
Severity: {severity_str}
CVSS Score: {cvss_score}
Target URL: {finding.target_url}
Target Parameter: {finding.target_parameter.name if finding.target_parameter else 'N/A'}

=== PROOF OF CONCEPT ===
{finding.proof_of_concept}

=== INSTRUCTIONS ===
Generate a JSON object with:
- title: concise, descriptive title
- executive_summary: 1-2 paragraph summary
- steps_to_reproduce: numbered steps
- remediation: actionable fix steps

Return the JSON object now.
"""


def _template_fallback_report(finding: Finding, log: Any) -> _ReportContent:
    """Generate a report using a template (no LLM)."""
    cat_label: str = finding.category.value.upper().replace("_", " ")
    severity_str: str = finding.severity.value.upper()
    cvss_score: str = f"{finding.cvss.base_score}" if finding.cvss else "N/A"

    title: str = finding.title if finding.title else f"{cat_label} in {finding.target_url}"

    exec_summary: str = (
        f"A {cat_label} vulnerability was identified at {finding.target_url}. "
        f"The vulnerability was confirmed through automated testing. "
        f"The severity is rated as {severity_str} with a CVSS base score of {cvss_score}."
    )

    steps: list[str] = [
        f"1. Navigate to {finding.target_url}",
        f"2. Identify the vulnerable parameter: "
        f"{finding.target_parameter.name if finding.target_parameter else '(URL-level)'}",
        f"3. Inject the payload as shown in the Proof of Concept below.",
        f"4. Observe the response indicating successful exploitation.",
    ]

    remediation: list[str] = [
        f"Implement proper input validation for the {cat_label} vulnerability.",
        "Use parameterized queries / prepared statements (for SQLi).",
        "Apply output encoding / sanitization (for XSS).",
        "Follow OWASP best practices for the affected component.",
    ]

    log.info("template_fallback_report_generated",
             finding_id=finding.id, title=title,
             message="Template-based report generated (LLM was unavailable).")

    return _ReportContent(
        title=title,
        executive_summary=exec_summary,
        steps_to_reproduce=steps,
        remediation=remediation,
    )


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
# Re-exports
# ---------------------------------------------------------------------------


__all__ = ["generate_report"]
