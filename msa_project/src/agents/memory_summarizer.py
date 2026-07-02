"""
src/agents/memory_summarizer.py
===============================

Node 7 of the 16-node LangGraph framework: the **Memory Summarizer** —
the "Context Compressor".

This node prevents the LangGraph state from growing unboundedly during
long pentest sessions. Raw recon output, raw crawler output, and
accumulated error logs can easily exceed 100KB, which bloats every
LLM prompt that includes state context. The Memory Summarizer detects
when the state is getting large, compresses the raw logs into a
concise summary, and clears the raw data from the state.

Why Gemini Flash?
-----------------
The Architect specified Gemini Flash (``gemini-2.5-flash``) for this
node because summarization is a high-volume, low-complexity task.
Flash is fast and cheap, and the ``temperature=0.1`` setting (provided
in :mod:`src.shared.llm`) ensures deterministic, factual summaries.
Flash supports ``with_structured_output()`` natively, so the
:class:`_MemorySummary` schema is enforced at the API level.

Context window pressure heuristic
---------------------------------
The "pressure" is a rough heuristic for how full the state is:

    pressure = (len(raw_recon_output) + len(raw_crawler_output)
                + len(str(errors))) / 100_000

A pressure of 1.0 means the raw logs total ~100KB. The threshold for
triggering compression is 0.8 (80KB). This is intentionally
conservative — we'd rather compress too early than too late, because
running out of context window mid-pipeline causes LLM call failures
that are hard to recover from.

State mutation (critical)
-------------------------
When compression is triggered, the node returns:

- ``memory_summary``: the new compressed summary (replaces any
  existing summary).
- ``context_window_pressure``: reset to ``0.1`` (the raw logs are
  gone, so the pressure is near zero).
- ``raw_recon_output``: set to ``""`` (cleared to free memory).
- ``raw_crawler_output``: set to ``""`` (cleared to free memory).

This is one of the few nodes that explicitly *deletes* data from the
state. The raw logs are no longer needed once they've been parsed
(the structured ``recon_data`` and ``crawler_data`` objects persist)
and summarized (the ``memory_summary`` retains the critical facts).

LangGraph contract
------------------
::

    async def summarize_memory(state: AppState) -> dict:

- Reads: the entire ``state`` (for pressure calculation and the
  existing ``memory_summary``).
- Writes: returns a partial state dict. When pressure is low, only
  ``context_window_pressure`` is updated. When pressure is high, the
  raw logs are cleared and ``memory_summary`` is replaced.

Raises
------
- :class:`LLMOutputParsingError` — LLM output could not be parsed
  into :class:`_MemorySummary`. (Only raised when compression is
  triggered; low-pressure path never calls the LLM.)
- :class:`LLMRateLimitError` — Gemini API returned 429 / quota.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.shared.exceptions import (
    LLMOutputParsingError,
    LLMRateLimitError,
)
from src.shared.llm import gemini_flash
from src.shared.logging import get_logger
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: The pressure threshold above which compression is triggered.
#: 0.8 means ~80KB of raw logs. This is intentionally conservative.
_COMPRESSION_THRESHOLD: float = 0.8

#: The divisor for the pressure calculation. 100,000 means the
#: pressure is 1.0 when raw logs total ~100KB.
_PRESSURE_DIVISOR: int = 100_000

#: The pressure value set after compression. The raw logs are gone,
#: so the pressure is near zero. We use 0.1 (not 0.0) to indicate
#: that some state (the summary itself, structured data) still exists.
_POST_COMPRESSION_PRESSURE: float = 0.1


# ---------------------------------------------------------------------------
# Private wrapper model for Gemini Flash structured output
# ---------------------------------------------------------------------------


class _MemorySummary(BaseModel):
    """Private wrapper model for the LLM's compressed memory summary.

    Gemini Flash's ``with_structured_output()`` enforces this schema
    at the API level. The LLM returns a JSON object with a single
    ``summary`` field containing the compressed narrative.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    summary: str = Field(
        description="A highly compressed summary of everything "
        "discovered so far in the pentest engagement. Retain critical "
        "facts: target IPs, main technologies, key endpoints, "
        "confirmed vulnerabilities, WAF signatures. Discard verbose "
        "logs, raw HTTP responses, and redundant details.",
    )


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT: str = """\
You are a Context Compressor embedded in an automated web pentesting \
framework. Your job is to read the existing memory summary (if any) \
and the current raw logs, and produce a NEW, highly compressed summary \
of what has been discovered so far.

COMPRESSION RULES:
1. Retain CRITICAL FACTS:
   - Target IP addresses and hostnames
   - Main technologies detected (e.g. "nginx/1.21", "PHP/8.1")
   - Key endpoints and URLs discovered
   - Confirmed vulnerabilities (category, target, severity)
   - WAF signatures detected (e.g. "Cloudflare")
   - Authentication requirements
   - Any blockers or issues encountered

2. DISCARD verbose data:
   - Raw HTTP request/response bodies
   - Full Nmap scan output
   - Full Playwright crawl logs
   - Detailed error stack traces
   - Redundant or duplicate information

3. FORMAT the summary as a concise bulleted list (using "- " prefix) \
or a short paragraph. Maximum 500 words. Every word must earn its place.

4. If an existing memory_summary is provided, INTEGRATE it with the \
new raw logs. Do not lose facts from the old summary — only add new \
information and remove verbosity.

5. Do NOT include payload strings, PoC commands, or exploit code. \
Those are stored separately in the structured state.

Output ONLY the JSON object matching the _MemorySummary schema. The \
"summary" field should contain the compressed narrative.
"""


_USER_PROMPT_TEMPLATE: str = """\
Compress the following pentest engagement context into a concise summary.

=== EXISTING MEMORY SUMMARY ===
{existing_summary}

=== RAW RECON OUTPUT ({recon_length} chars) ===
{raw_recon_output}

=== RAW CRAWLER OUTPUT ({crawler_length} chars) ===
{raw_crawler_output}

=== ACCUMULATED ERRORS ({error_count} errors, {error_length} chars) ===
{errors_text}

=== INSTRUCTIONS ===
Produce a highly compressed summary of the engagement state. Retain \
critical facts (IPs, technologies, endpoints, vulnerabilities, WAF \
signatures). Discard verbose logs. Maximum 500 words.

Return the _MemorySummary JSON object now.
"""


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


async def summarize_memory(state: AppState) -> dict[str, Any]:
    """LangGraph Node 7: compress the state's raw logs when context
    window pressure is high.

    Parameters
    ----------
    state:
        The current :class:`~src.shared.state.AppState`. The node reads
        ``raw_recon_output``, ``raw_crawler_output``, ``errors``, and
        ``memory_summary`` to calculate pressure and perform
        compression.

    Returns
    -------
    dict
        When pressure is low (``< 0.8``):
            ``{"context_window_pressure": <float>}``
        When pressure is high (``>= 0.8``) and compression succeeds:
            ``{"memory_summary": <str>,
               "context_window_pressure": 0.1,
               "raw_recon_output": "",
               "raw_crawler_output": ""}``

    Raises
    ------
    LLMOutputParsingError
        If the LLM output cannot be parsed into :class:`_MemorySummary`
        (only when compression is triggered).
    LLMRateLimitError
        If the Gemini API returns 429 / quota-exceeded (only when
        compression is triggered).
    """
    log = get_logger("memory_summarizer")

    # ---------------------------------------------------------------
    # 1. Calculate context window pressure.
    # ---------------------------------------------------------------
    pressure: float = _calculate_pressure(state)

    log.info(
        "memory_pressure_checked",
        pressure=round(pressure, 4),
        threshold=_COMPRESSION_THRESHOLD,
        compression_triggered=pressure >= _COMPRESSION_THRESHOLD,
    )

    # ---------------------------------------------------------------
    # 2. If pressure is low, do nothing (just update the pressure value).
    # ---------------------------------------------------------------
    if pressure < _COMPRESSION_THRESHOLD:
        log.info(
            "memory_compression_skipped",
            reason="pressure_below_threshold",
            pressure=round(pressure, 4),
        )
        return {"context_window_pressure": pressure}

    # ---------------------------------------------------------------
    # 3. Pressure is high — compress via Gemini Flash.
    # ---------------------------------------------------------------
    log.info(
        "memory_compression_started",
        pressure=round(pressure, 4),
    )

    # Build the LLM prompt.
    existing_summary: str = state.get("memory_summary") or "(none)"
    raw_recon: str = state.get("raw_recon_output") or ""
    raw_crawler: str = state.get("raw_crawler_output") or ""
    errors: list[dict[str, Any]] | None = state.get("errors")
    errors_text: str = _format_errors_for_prompt(errors)

    user_prompt: str = _USER_PROMPT_TEMPLATE.format(
        existing_summary=existing_summary,
        raw_recon_output=_truncate_for_prompt(raw_recon),
        recon_length=len(raw_recon),
        raw_crawler_output=_truncate_for_prompt(raw_crawler),
        crawler_length=len(raw_crawler),
        errors_text=_truncate_for_prompt(errors_text),
        error_count=len(errors) if errors else 0,
        error_length=len(errors_text),
    )

    # Invoke Gemini Flash with structured output.
    structured_llm = gemini_flash.with_structured_output(_MemorySummary)

    try:
        result: _MemorySummary = await structured_llm.ainvoke(
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
                "memory_compression_rate_limited",
                error_type=type(exc).__name__,
                error_message=exc_str[:200],
            )
            raise LLMRateLimitError(
                f"Memory Summarizer LLM rate limited: {exc}",
                provider="google",
                model="gemini-2.5-flash",
                retry_after_seconds=_extract_retry_after(exc),
            ) from exc

        log.warning(
            "memory_compression_llm_failed",
            error_type=type(exc).__name__,
            error_message=exc_str[:200],
        )
        raise LLMOutputParsingError(
            f"Memory Summarizer LLM call failed: {exc}",
            raw_output=f"(exception: {exc_repr})",
            schema_name="_MemorySummary",
            details={
                "exception_type": type(exc).__name__,
                "exception_repr": exc_repr[:500],
                "pressure_before": pressure,
            },
        ) from exc

    new_summary: str = result.summary

    log.info(
        "memory_compression_complete",
        pressure_before=round(pressure, 4),
        pressure_after=_POST_COMPRESSION_PRESSURE,
        old_summary_length=len(existing_summary),
        new_summary_length=len(new_summary),
        raw_recon_cleared=len(raw_recon) > 0,
        raw_crawler_cleared=len(raw_crawler) > 0,
    )

    # ---------------------------------------------------------------
    # 4. Return the compressed state.
    # ---------------------------------------------------------------
    # The raw logs are explicitly cleared to free memory. The
    # structured data (recon_data, crawler_data) is NOT cleared —
    # it's the parsed, structured form and is still needed by
    # downstream nodes.
    return {
        "memory_summary": new_summary,
        "context_window_pressure": _POST_COMPRESSION_PRESSURE,
        "raw_recon_output": "",
        "raw_crawler_output": "",
    }


# ---------------------------------------------------------------------------
# Internal: pressure calculation
# ---------------------------------------------------------------------------


def _calculate_pressure(state: AppState) -> float:
    """Calculate the context window pressure heuristic.

    pressure = (len(raw_recon_output) + len(raw_crawler_output)
                + len(str(errors))) / 100_000

    A pressure of 1.0 means ~100KB of raw logs. The compression
    threshold is 0.8 (80KB).

    Parameters
    ----------
    state:
        The current AppState.

    Returns
    -------
    float
        The pressure value (typically in [0.0, 2.0+]).
    """
    raw_recon: str = state.get("raw_recon_output") or ""
    raw_crawler: str = state.get("raw_crawler_output") or ""

    errors: list[dict[str, Any]] | None = state.get("errors")
    if errors:
        # Stringify the errors list. This is a rough heuristic —
        # we don't need the exact byte count, just a reasonable
        # approximation of how much space the errors take up.
        errors_str: str = str(errors)
    else:
        errors_str = ""

    total_chars: int = len(raw_recon) + len(raw_crawler) + len(errors_str)
    return total_chars / _PRESSURE_DIVISOR


# ---------------------------------------------------------------------------
# Internal: prompt helpers
# ---------------------------------------------------------------------------


#: Maximum length of each section in the LLM prompt. If the raw logs
#: are very large (which they will be, since that's why we're
#: compressing), we truncate them to fit within Gemini Flash's context
#: window. The LLM doesn't need to see every byte — it needs enough
#: to extract the critical facts.
_MAX_SECTION_LENGTH: int = 30_000


def _truncate_for_prompt(text: str) -> str:
    """Truncate a string to fit within the LLM prompt.

    If the text exceeds :data:`_MAX_SECTION_LENGTH`, it's truncated
    with a notice. The LLM doesn't need to see every byte of a 200KB
    Nmap log — it needs enough to extract the critical facts.
    """
    if len(text) <= _MAX_SECTION_LENGTH:
        return text

    # Keep the first half and the last half of the allowed length.
    # The beginning usually has the most important data (host info,
    # open ports); the end may have the most recent results.
    half: int = _MAX_SECTION_LENGTH // 2
    truncated: str = text[:half]
    truncated += f"\n\n... [{len(text) - _MAX_SECTION_LENGTH} chars truncated] ...\n\n"
    truncated += text[-half:]
    return truncated


def _format_errors_for_prompt(errors: list[dict[str, Any]] | None) -> str:
    """Format the errors list into a readable string for the LLM prompt.

    Each error is formatted as:
        [error_type] message

    Very long error messages are truncated to 500 chars each.
    """
    if not errors:
        return "(no errors)"

    lines: list[str] = []
    for err in errors:
        exc_type: str = err.get("exception_type", "unknown")
        message: str = err.get("message", "")
        if len(message) > 500:
            message = message[:500] + "... (truncated)"
        lines.append(f"[{exc_type}] {message}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal: error classification (duplicated from other agents)
# ---------------------------------------------------------------------------


def _is_rate_limit_error(haystack: str) -> bool:
    """Case-insensitive rate-limit indicator matching.

    Duplicated from :func:`src.agents.validator._is_rate_limit_error`
    to keep agent modules independent.
    """
    if not haystack:
        return False
    normalized = haystack.lower()
    rate_limit_indicators = (
        "429", "rate limit", "rate_limit", "ratelimit",
        "quota", "quota exceeded",
        "resourceexhausted", "resource_exhausted", "resource exhausted",
        "too many requests", "throttled", "throttling",
        "retry-after", "retry_after", "service unavailable",
    )
    return any(indicator in normalized for indicator in rate_limit_indicators)


def _extract_retry_after(exc: Exception) -> float | None:
    """Best-effort Retry-After extraction from exception.

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


__all__ = ["summarize_memory"]
