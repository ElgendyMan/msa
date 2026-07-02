"""
src/agents/planner.py
=====================

Node 3 of the 16-node LangGraph framework: the **Planner** — the
"State Reasoner" and the final DeepSeek R1 node.

The Planner looks at the big picture and decides the next high-level
*macro-transition* in the pentest pipeline. It does NOT handle
low-level execution loops (those are the Orchestrator's job via its
deterministic rules). The Planner only fires when the Orchestrator's
exception rules don't match — i.e., when there's no pending execution,
validation, optimization, or CVSS scoring to do.

Why DeepSeek R1?
----------------
The Architect specified DeepSeek R1 (``deepseek-reasoner``) for this
node because strategic planning requires *reasoning*: the LLM must
look at the current state (what recon has been done, what hypotheses
have been generated, what findings have been confirmed) and decide
the next logical phase. R1's chain-of-thought capability lets it work
through the state holistically before committing to a decision.

R1 does NOT support ``with_structured_output()``, so we use a private
wrapper model :class:`_PlannerOutput` with
:class:`PydanticOutputParser` and inject
``parser.get_format_instructions()`` into the system prompt. R1's
output is preprocessed to strip ``<think>`` tags before parsing (same
pattern as the Validator, Hypothesis Analyzer, and Payload Optimizer).

Macro-transitions vs. micro-transitions
---------------------------------------
The pipeline has two layers of routing:

- **Macro-transitions** (Planner's job): high-level phase changes like
  "recon → crawl", "crawl → hypothesis", "hypothesis → payload",
  "findings → report". These happen infrequently and require
  strategic judgment.

- **Micro-transitions** (Orchestrator's job): low-level loops like
  "payload generated → execute it", "execution done → validate it",
  "validation inconclusive → optimize payload". These happen
  frequently and are 100% deterministic.

The Orchestrator's exception rules (rules 4-7) handle micro-transitions
automatically. The Planner only fires when the Orchestrator falls
through to rule 8 (fallback) — meaning there's no pending micro-work.

Defense-in-depth: deterministic fallback
-----------------------------------------
Even with the LLM, the Planner includes a pure-Python deterministic
fallback (:func:`_deterministic_plan`). If the LLM call fails for ANY
reason (rate limit, parse error, network error), the fallback is used
instead of crashing the session. This ensures the pipeline can always
make progress even when the LLM is unavailable.

The fallback implements the same logic guidelines the LLM is instructed
to follow, so the LLM's contribution is *better judgment* rather than
*correctness* — the fallback is always correct, the LLM is optionally
smarter.

LangGraph contract
------------------
::

    async def plan_next_step(state: AppState) -> dict:

- Reads: ``recon_data``, ``crawler_data``, ``hypotheses``, ``payloads``,
         ``confirmed_findings``, ``phase_history``, ``raw_recon_output``,
         ``raw_crawler_output``, ``active_payload_id``,
         ``execution_results``, ``validation_reports``.
- Writes: returns ``{"next_phase": <str>, "phase_history": [<str>]}`` —
  updates the ``next_phase`` pointer (overwrite reducer) and appends
  to ``phase_history`` (``operator.add`` reducer).

Raises
------
- :class:`LLMOutputParsingError` — LLM output could not be parsed into
  :class:`_PlannerOutput`. (The deterministic fallback is used
  instead, so this is only raised if the fallback also fails — which
  should be impossible.)
- :class:`LLMRateLimitError` — LLM provider returned 429 / quota.
  (Same: fallback is used instead.)
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.shared.exceptions import (
    LLMOutputParsingError,
    LLMRateLimitError,
)
from src.shared.llm import deepseek_r1
from src.shared.logging import get_logger
from src.shared.schemas import (
    CrawlerResult,
    Finding,
    Hypothesis,
    Payload,
    ReconResult,
    ValidationReport,
)
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Private wrapper model for R1 structured output
# ---------------------------------------------------------------------------


class _PlannerOutput(BaseModel):
    """Private wrapper model for the Planner's R1 structured output.

    R1 does not support ``with_structured_output()``, so we use
    :class:`PydanticOutputParser` with this simple wrapper. The LLM
    returns a JSON object with two fields:

    - ``next_phase``: the name of the next macro-phase to route to.
    - ``reasoning``: a brief explanation of why this phase was chosen.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    next_phase: str = Field(
        description="The next macro-phase to route to. Must be one of: "
        "'recon_parser', 'crawler_parser', 'hypothesis_analyzer', "
        "'payload_generator', 'reporter', or 'complete'.",
    )

    reasoning: str = Field(
        description="A brief (1-2 sentence) explanation of why this "
        "phase was chosen based on the current state.",
    )


# ---------------------------------------------------------------------------
# Valid next_phase values
# ---------------------------------------------------------------------------


#: The set of valid next_phase values the Planner can return. The
#: Orchestrator reads ``state["next_phase"]`` and routes to the
#: corresponding node. If the Planner returns a value not in this set,
#: the Orchestrator's fallback rule (route to "planner") would create
#: an infinite loop, so we validate post-parse and fall back to the
#: deterministic plan if the LLM returns an invalid value.
VALID_NEXT_PHASES: frozenset[str] = frozenset(
    {
        "recon_parser",
        "crawler_parser",
        "hypothesis_analyzer",
        "payload_generator",
        "reporter",
        "complete",
    }
)


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_TEMPLATE: str = """\
You are a Lead Pentest Strategist embedded in an automated web \
pentesting framework. Your job is to look at the big picture and \
decide the next high-level macro-phase of the engagement.

IMPORTANT: The Orchestrator (a deterministic router) handles all \
low-level execution loops automatically:
- If a payload was just generated, the Orchestrator routes to the \
Execution Sandbox automatically.
- If an execution just completed, the Orchestrator routes to the \
Validator automatically.
- If validation was inconclusive, the Orchestrator routes to the \
Payload Optimizer automatically.
- If a finding was confirmed, the Orchestrator routes to the CVSS \
Engine automatically.

You do NOT need to manage these micro-transitions. You ONLY choose \
macro-transitions — the high-level flow from one phase of the \
engagement to the next.

VALID NEXT_PHASE VALUES (choose exactly one):
- "recon_parser": Parse raw Nmap/Subfinder output into structured data.
- "crawler_parser": Parse Playwright crawl output into structured data.
- "hypothesis_analyzer": Generate vulnerability hypotheses from the \
attack surface.
- "payload_generator": Craft benign PoC payloads for unprocessed \
hypotheses.
- "reporter": Generate the final Markdown report from confirmed findings.
- "complete": The engagement is finished. No more work to do.

DECISION GUIDELINES (apply in order):
1. If raw_recon_output exists but recon_data is empty or missing → \
"recon_parser". We have raw recon text that needs to be parsed.
2. If recon_data exists but crawler_data is empty or missing → \
"crawler_parser". We have recon data but haven't crawled the web app yet.
3. If crawler_data exists but hypotheses is empty → \
"hypothesis_analyzer". We have the attack surface but haven't \
identified potential vulnerabilities yet.
4. If hypotheses exist but some don't have payloads yet → \
"payload_generator". We have hypotheses that need PoC payloads.
5. If all hypotheses have payloads and have been processed (validated \
as TRUE_POSITIVE or FALSE_POSITIVE), and we have confirmed_findings → \
"reporter". The engagement is ready for reporting.
6. If all findings are reported and no more work remains → "complete".

STRICT CONSTRAINTS:
1. You are a WEB pentester. Ignore any non-web infrastructure \
(SSH, SMB, RDP, databases, etc.). The framework is web-only.
2. Do NOT choose "payload_generator" if all hypotheses already have \
payloads. Check the payloads list — if every hypothesis_id has a \
matching payload, the Payload Generator has nothing to do.
3. Do NOT choose "reporter" if there are no confirmed_findings. \
An empty report is useless.
4. Do NOT choose "complete" unless all hypotheses have been processed \
AND the report has been generated (if findings exist).
5. If you're unsure, choose the earliest phase in the pipeline that \
hasn't been completed yet. Progress forward, never backward.

OUTPUT FORMAT:
{format_instructions}

Output ONLY the JSON object. Do not wrap it in markdown code fences. \
Do not add commentary. If you need to reason, do so in a \
<think>...</think> block BEFORE the JSON, then output the JSON on its own.
"""


_USER_PROMPT_TEMPLATE: str = """\
Analyze the current engagement state and decide the next macro-phase. \
Think step-by-step, then output the JSON _PlannerOutput.

=== ENGAGEMENT STATE SUMMARY ===
Target: {target_url}
Scope verified: {scope_verified}

Recon phase:
  raw_recon_output present: {has_raw_recon}
  recon_data present: {has_recon_data}
  recon hosts: {recon_host_count}
  recon web endpoints: {recon_endpoint_count}

Crawl phase:
  raw_crawler_output present: {has_raw_crawler}
  crawler_data present: {has_crawler_data}
  crawler URLs: {crawler_url_count}
  crawler parameters: {crawler_param_count}
  crawler forms: {crawler_form_count}

Hypothesis phase:
  hypotheses count: {hypothesis_count}
  hypothesis categories: {hypothesis_categories}

Payload phase:
  payloads count: {payload_count}
  hypotheses with payloads: {hypotheses_with_payloads}
  hypotheses without payloads: {hypotheses_without_payloads}
  active_payload_id: {active_payload_id}

Execution phase:
  execution_results count: {execution_count}

Validation phase:
  validation_reports count: {validation_count}
  true_positives: {true_positive_count}
  false_positives: {false_positive_count}
  inconclusive: {inconclusive_count}

Findings phase:
  confirmed_findings count: {confirmed_finding_count}
  confirmed finding categories: {confirmed_finding_categories}

Report phase:
  final_report present: {has_final_report}

Phase history (most recent last):
  {phase_history}

=== INSTRUCTIONS ===
Based on the state summary above, choose the next macro-phase. Apply \
the decision guidelines from the system prompt in order. Return the \
_PlannerOutput JSON object with next_phase and reasoning.
"""


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


async def plan_next_step(state: AppState) -> dict[str, Any]:
    """LangGraph Node 3: decide the next macro-phase of the engagement.

    Parameters
    ----------
    state:
        The current :class:`~src.shared.state.AppState`. The Planner
        reads the overall state to understand where the engagement is.

    Returns
    -------
    dict
        ``{"next_phase": <str>, "phase_history": [<str>]}`` — updates
        the ``next_phase`` pointer (the Orchestrator reads this on the
        next routing cycle) and appends the phase to ``phase_history``
        (via the ``operator.add`` reducer).

    Raises
    ------
    LLMOutputParsingError
        If the LLM output cannot be parsed AND the deterministic
        fallback also fails (should be impossible).
    LLMRateLimitError
        If the DeepSeek API returns 429 / quota AND the deterministic
        fallback also fails (should be impossible).
    """
    log = get_logger("planner")

    # Bind target context for logging.
    target: Any = state.get("target")
    target_url: str = str(target.url) if target is not None else "(unknown)"
    log = log.bind(target_url=target_url)

    log.info("planner_started")

    # ---------------------------------------------------------------
    # 1. Try the LLM-based plan.
    # ---------------------------------------------------------------
    planner_output: _PlannerOutput | None = None
    llm_error: Exception | None = None

    try:
        planner_output = await _llm_plan(state, log)
    except (LLMOutputParsingError, LLMRateLimitError) as exc:
        # LLM failed — we'll fall back to the deterministic plan.
        # Log the error but do NOT re-raise; the fallback ensures the
        # pipeline can always make progress.
        llm_error = exc
        log.warning(
            "planner_llm_failed_using_fallback",
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
        )
    except Exception as exc:
        # Unexpected error — same treatment: fall back.
        llm_error = exc
        log.warning(
            "planner_llm_unexpected_error_using_fallback",
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
        )

    # ---------------------------------------------------------------
    # 2. If the LLM succeeded, validate its output.
    # ---------------------------------------------------------------
    if planner_output is not None:
        # Validate next_phase is a recognized value.
        if planner_output.next_phase not in VALID_NEXT_PHASES:
            log.warning(
                "planner_llm_invalid_next_phase",
                llm_next_phase=planner_output.next_phase,
                valid_phases=sorted(VALID_NEXT_PHASES),
            )
            planner_output = None  # Fall back to deterministic.
        else:
            log.info(
                "planner_llm_decision",
                next_phase=planner_output.next_phase,
                reasoning=planner_output.reasoning[:200],
            )

    # ---------------------------------------------------------------
    # 3. If the LLM failed or returned an invalid value, use the
    #    deterministic fallback.
    # ---------------------------------------------------------------
    if planner_output is None:
        next_phase: str = _deterministic_plan(state)
        reasoning: str = (
            "Deterministic fallback (LLM unavailable or returned invalid output). "
            f"LLM error: {type(llm_error).__name__ if llm_error else 'invalid output'}"
        )
        planner_output = _PlannerOutput(
            next_phase=next_phase,
            reasoning=reasoning,
        )
        log.info(
            "planner_fallback_decision",
            next_phase=next_phase,
            reasoning=reasoning,
        )

    # ---------------------------------------------------------------
    # 4. Log the final decision and return.
    # ---------------------------------------------------------------
    log.info(
        "planner_complete",
        next_phase=planner_output.next_phase,
        reasoning=planner_output.reasoning,
        used_fallback=llm_error is not None,
    )

    return {
        "next_phase": planner_output.next_phase,
        "phase_history": [planner_output.next_phase],
    }


# ---------------------------------------------------------------------------
# Internal: LLM-based planning
# ---------------------------------------------------------------------------


async def _llm_plan(state: AppState, log: Any) -> _PlannerOutput:
    """Invoke DeepSeek R1 to decide the next macro-phase.

    Raises :class:`LLMOutputParsingError` or :class:`LLMRateLimitError`
    on failure. The caller catches these and falls back to the
    deterministic plan.
    """
    # Build parser + prompts.
    parser: PydanticOutputParser = PydanticOutputParser(
        pydantic_object=_PlannerOutput
    )

    system_prompt: str = _SYSTEM_PROMPT_TEMPLATE.format(
        format_instructions=parser.get_format_instructions()
    )

    user_prompt: str = _build_user_prompt(state)

    # Invoke R1.
    try:
        response_message = await deepseek_r1.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as exc:
        exc_repr = repr(exc)
        exc_str = str(exc)
        haystack = f"{exc_repr} {exc_str}"

        if _is_rate_limit_error(haystack):
            raise LLMRateLimitError(
                f"Planner LLM rate limited: {exc}",
                provider="deepseek",
                model="deepseek-reasoner",
                retry_after_seconds=_extract_retry_after(exc),
            ) from exc

        raise LLMOutputParsingError(
            f"Planner LLM call failed: {exc}",
            raw_output=f"(exception: {exc_repr})",
            schema_name="_PlannerOutput",
            details={
                "exception_type": type(exc).__name__,
                "exception_repr": exc_repr[:500],
            },
        ) from exc

    raw_llm_output: str = response_message.content

    # Parse (with R1 <think> preprocessing fallback).
    parsed: _PlannerOutput | None = None
    try:
        parsed = await parser.aparse(raw_llm_output)
    except (OutputParserException, ValidationError):
        pass

    if parsed is None:
        cleaned: str = _preprocess_llm_output(raw_llm_output)
        try:
            parsed = _PlannerOutput.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            raise LLMOutputParsingError(
                f"Planner LLM output unparseable: {exc}",
                raw_output=raw_llm_output,
                schema_name="_PlannerOutput",
                details={
                    "cleaned_output_preview": cleaned[:500],
                    "parse_error": str(exc)[:500],
                },
            ) from exc

    return parsed


# ---------------------------------------------------------------------------
# Internal: deterministic fallback plan
# ---------------------------------------------------------------------------


def _deterministic_plan(state: AppState) -> str:
    """Pure-Python deterministic fallback for when the LLM is unavailable.

    Implements the same decision guidelines the LLM is instructed to
    follow. This ensures the pipeline can always make progress even
    when DeepSeek R1 is rate-limited, down, or returning garbage.

    The logic is intentionally simple and forward-progressing: it
    checks each phase in order and routes to the first incomplete one.
    """
    # 1. Raw recon exists but no structured recon_data → recon_parser.
    raw_recon: str | None = state.get("raw_recon_output")
    recon_data: ReconResult | None = state.get("recon_data")
    if raw_recon and raw_recon.strip() and recon_data is None:
        return "recon_parser"

    # 2. Recon data exists but no crawler_data → crawler_parser.
    #    (Also require that we have raw_crawler_output to parse.)
    raw_crawler: str | None = state.get("raw_crawler_output")
    crawler_data: CrawlerResult | None = state.get("crawler_data")
    if recon_data is not None and crawler_data is None:
        if raw_crawler and raw_crawler.strip():
            return "crawler_parser"
        # If no raw_crawler_output, we can't crawl. Fall through to
        # hypothesis analysis (which works with just recon_data).

    # 3. Crawler data exists but no hypotheses → hypothesis_analyzer.
    hypotheses: list[Hypothesis] | None = state.get("hypotheses")
    if (crawler_data is not None or recon_data is not None) and not hypotheses:
        return "hypothesis_analyzer"

    # 4. Hypotheses exist but some don't have payloads → payload_generator.
    payloads: list[Payload] | None = state.get("payloads")
    if hypotheses and payloads is not None:
        hypothesis_ids: set[str] = {h.id for h in hypotheses}
        payload_hypothesis_ids: set[str] = {p.hypothesis_id for p in payloads}
        unprocessed: set[str] = hypothesis_ids - payload_hypothesis_ids
        if unprocessed:
            return "payload_generator"

    # Also check if hypotheses exist but payloads is empty/missing.
    if hypotheses and not payloads:
        return "payload_generator"

    # 5. All hypotheses processed, confirmed findings exist, no report → reporter.
    confirmed_findings: list[Finding] | None = state.get("confirmed_findings")
    final_report: str | None = state.get("final_report")
    if confirmed_findings and not final_report:
        return "reporter"

    # 6. Everything is done → complete.
    return "complete"


# ---------------------------------------------------------------------------
# Internal: user prompt construction
# ---------------------------------------------------------------------------


def _build_user_prompt(state: AppState) -> str:
    """Build the user prompt with a summary of the engagement state.

    The prompt provides a structured summary of every phase's status
    so the LLM can make an informed decision. Counts and boolean flags
    are used rather than full data dumps to stay within token limits.
    """
    # --- Target ---
    target: Any = state.get("target")
    target_url: str = str(target.url) if target is not None else "(unknown)"
    scope_verified: bool = state.get("scope_verified", False)

    # --- Recon phase ---
    raw_recon: str | None = state.get("raw_recon_output")
    recon_data: ReconResult | None = state.get("recon_data")
    has_raw_recon: bool = bool(raw_recon and raw_recon.strip())
    has_recon_data: bool = recon_data is not None
    recon_host_count: int = len(recon_data.hosts) if recon_data else 0
    recon_endpoint_count: int = len(recon_data.web_endpoints) if recon_data else 0

    # --- Crawl phase ---
    raw_crawler: str | None = state.get("raw_crawler_output")
    crawler_data: CrawlerResult | None = state.get("crawler_data")
    has_raw_crawler: bool = bool(raw_crawler and raw_crawler.strip())
    has_crawler_data: bool = crawler_data is not None
    crawler_url_count: int = len(crawler_data.urls) if crawler_data else 0
    crawler_param_count: int = len(crawler_data.parameters) if crawler_data else 0
    crawler_form_count: int = len(crawler_data.forms) if crawler_data else 0

    # --- Hypothesis phase ---
    hypotheses: list[Hypothesis] | None = state.get("hypotheses")
    hypothesis_count: int = len(hypotheses) if hypotheses else 0
    hypothesis_categories: str = (
        ", ".join(sorted({h.category.value for h in hypotheses}))
        if hypotheses else "(none)"
    )

    # --- Payload phase ---
    payloads: list[Payload] | None = state.get("payloads")
    payload_count: int = len(payloads) if payloads else 0
    hypothesis_ids: set[str] = {h.id for h in hypotheses} if hypotheses else set()
    payload_hyp_ids: set[str] = (
        {p.hypothesis_id for p in payloads} if payloads else set()
    )
    hyps_with_payloads: int = len(hypothesis_ids & payload_hyp_ids)
    hyps_without_payloads: int = len(hypothesis_ids - payload_hyp_ids)
    active_payload_id: str | None = state.get("active_payload_id")

    # --- Execution phase ---
    execution_results: list[Any] | None = state.get("execution_results")
    execution_count: int = len(execution_results) if execution_results else 0

    # --- Validation phase ---
    validation_reports: list[ValidationReport] | None = state.get("validation_reports")
    validation_count: int = len(validation_reports) if validation_reports else 0
    if validation_reports:
        true_positives: int = sum(
            1 for v in validation_reports if v.verdict.value == "true_positive"
        )
        false_positives: int = sum(
            1 for v in validation_reports if v.verdict.value == "false_positive"
        )
        inconclusive: int = sum(
            1 for v in validation_reports if v.verdict.value == "inconclusive"
        )
    else:
        true_positives = false_positives = inconclusive = 0

    # --- Findings phase ---
    confirmed_findings: list[Finding] | None = state.get("confirmed_findings")
    confirmed_finding_count: int = len(confirmed_findings) if confirmed_findings else 0
    confirmed_finding_categories: str = (
        ", ".join(sorted({f.category.value for f in confirmed_findings}))
        if confirmed_findings else "(none)"
    )

    # --- Report phase ---
    final_report: str | None = state.get("final_report")
    has_final_report: bool = bool(final_report and final_report.strip())

    # --- Phase history ---
    phase_history: list[Any] | None = state.get("phase_history")
    phase_history_str: str = (
        ", ".join(str(p) for p in phase_history)
        if phase_history else "(empty)"
    )

    return _USER_PROMPT_TEMPLATE.format(
        target_url=target_url,
        scope_verified=scope_verified,
        has_raw_recon=has_raw_recon,
        has_recon_data=has_recon_data,
        recon_host_count=recon_host_count,
        recon_endpoint_count=recon_endpoint_count,
        has_raw_crawler=has_raw_crawler,
        has_crawler_data=has_crawler_data,
        crawler_url_count=crawler_url_count,
        crawler_param_count=crawler_param_count,
        crawler_form_count=crawler_form_count,
        hypothesis_count=hypothesis_count,
        hypothesis_categories=hypothesis_categories,
        payload_count=payload_count,
        hypotheses_with_payloads=hyps_with_payloads,
        hypotheses_without_payloads=hyps_without_payloads,
        active_payload_id=active_payload_id or "(none)",
        execution_count=execution_count,
        validation_count=validation_count,
        true_positive_count=true_positives,
        false_positive_count=false_positives,
        inconclusive_count=inconclusive,
        confirmed_finding_count=confirmed_finding_count,
        confirmed_finding_categories=confirmed_finding_categories,
        has_final_report=has_final_report,
        phase_history=phase_history_str,
    )


# ---------------------------------------------------------------------------
# Internal: R1 output preprocessing (duplicated from validator.py)
# ---------------------------------------------------------------------------


def _preprocess_llm_output(text: str) -> str:
    """Preprocess DeepSeek R1's output to extract the JSON payload.

    Duplicated from :func:`src.agents.validator._preprocess_llm_output`
    to keep agent modules independent.

    1. Removes ``<think>...</think>`` blocks.
    2. Removes markdown code fences.
    3. Extracts substring between first ``{`` and last ``}``.
    """
    cleaned: str = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    cleaned = re.sub(
        r"```(?:json)?\s*(.*?)\s*```", r"\1", cleaned, flags=re.DOTALL | re.IGNORECASE
    )
    cleaned = cleaned.strip("`").strip()
    if cleaned and not cleaned.startswith("{"):
        first_brace: int = cleaned.find("{")
        last_brace: int = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace : last_brace + 1]
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Internal: error classification (duplicated from validator.py)
# ---------------------------------------------------------------------------


def _is_rate_limit_error(haystack: str) -> bool:
    """Case-insensitive rate-limit indicator matching."""
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
    """Best-effort Retry-After extraction from exception."""
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


__all__ = ["plan_next_step", "VALID_NEXT_PHASES"]
