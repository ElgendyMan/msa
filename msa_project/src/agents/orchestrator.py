"""
src/agents/orchestrator.py
==========================

Node 2 of the 16-node LangGraph framework: the **Orchestrator** —
the "Stateless Router".
"""

from __future__ import annotations

from typing import Any

from src.shared.logging import get_logger
from src.shared.schemas import (
    ExecutionResult,
    Finding,
    Payload,
    ValidationReport,
    ValidationVerdict,
)
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Routing target constants
# ---------------------------------------------------------------------------

TARGET_ERROR_HANDLER: str = "error_handler"
TARGET_SCOPE_ENFORCER: str = "scope_enforcer"
TARGET_PLANNER: str = "planner"
TARGET_RECON_PARSER: str = "recon_parser"
TARGET_CRAWLER_PARSER: str = "crawler_parser"
TARGET_WEB_FILTER: str = "web_filter"
TARGET_HYPOTHESIS_ANALYZER: str = "hypothesis_analyzer"
TARGET_PAYLOAD_GENERATOR: str = "payload_generator"
TARGET_PAYLOAD_OPTIMIZER: str = "payload_optimizer"
TARGET_EXECUTION_SANDBOX: str = "execution_sandbox"
TARGET_VALIDATOR: str = "validator"
TARGET_KNOWLEDGE_RAG: str = "knowledge_rag"
TARGET_CVSS_ENGINE: str = "cvss_engine"
TARGET_BUSINESS_IMPACT: str = "business_impact"
TARGET_REPORTER: str = "reporter"
TARGET_MEMORY_SUMMARIZER: str = "memory_summarizer"

ALL_TARGETS: frozenset[str] = frozenset(
    {
        TARGET_ERROR_HANDLER, TARGET_SCOPE_ENFORCER, TARGET_PLANNER,
        TARGET_RECON_PARSER, TARGET_CRAWLER_PARSER, TARGET_WEB_FILTER,
        TARGET_HYPOTHESIS_ANALYZER, TARGET_PAYLOAD_GENERATOR,
        TARGET_PAYLOAD_OPTIMIZER, TARGET_EXECUTION_SANDBOX, TARGET_VALIDATOR,
        TARGET_KNOWLEDGE_RAG, TARGET_CVSS_ENGINE, TARGET_BUSINESS_IMPACT,
        TARGET_REPORTER, TARGET_MEMORY_SUMMARIZER,
    }
)


# ---------------------------------------------------------------------------
# CRITICAL FIX: one-shot consumption tracker for Planner's next_phase.
# ---------------------------------------------------------------------------
# state["next_phase"] is a plain overwrite channel, never cleared by any
# node after the Orchestrator routes to it. Since route_next_phase is a
# read-only conditional edge function (cannot mutate AppState), we track
# per-session how many phase_history entries have already been "spent" by
# Rule 3. The Planner appends exactly one phase_history entry every time
# it sets next_phase, so comparing len(phase_history) to the last-consumed
# count detects a NEW directive vs. a stale one from last cycle.
#
# Without this, Rule 3 wins forever after the Planner's first call,
# permanently starving Rules 4-7 and Rule 8 — an infinite loop.
_consumed_phase_history_len: dict[str, int] = {}


def _reset_consumption_tracker(session_id: str) -> None:
    """Test/utility helper: clear the one-shot tracker for a session."""
    _consumed_phase_history_len.pop(session_id, None)


# ---------------------------------------------------------------------------
# Public conditional edge function
# ---------------------------------------------------------------------------


def route_next_phase(state: AppState) -> str:
    """LangGraph conditional edge: route to the next node based on state.

    Priority order: 1) error_handler, 2) scope_enforcer, 3) planner
    override (one-shot), 4) execution needed, 5) validation needed,
    6) optimization needed, 7) CVSS needed, 8) fallback -> planner.
    """
    log = get_logger("orchestrator")

    target: str = _check_error_rule(state)
    if target is not None:
        log.info("routing_decision", target=target, rule="error_handler",
                  error_count=len(state.get("errors") or []))
        return target

    target = _check_scope_rule(state)
    if target is not None:
        log.info("routing_decision", target=target, rule="scope_enforcement",
                  scope_verified=state.get("scope_verified"))
        return target

    target = _check_planner_override(state)
    if target is not None:
        log.info("routing_decision", target=target, rule="planner_override",
                  next_phase=state.get("next_phase"))
        return target

    target = _check_execution_needed(state)
    if target is not None:
        log.info("routing_decision", target=target, rule="execution_needed",
                  active_payload_id=state.get("active_payload_id"))
        return target

    target = _check_validation_needed(state)
    if target is not None:
        log.info("routing_decision", target=target, rule="validation_needed")
        return target

    target = _check_optimization_needed(state)
    if target is not None:
        log.info("routing_decision", target=target, rule="optimization_needed")
        return target

    target = _check_cvss_needed(state)
    if target is not None:
        log.info("routing_decision", target=target, rule="cvss_needed")
        return target

    log.info("routing_decision", target=TARGET_PLANNER, rule="fallback")
    return TARGET_PLANNER


# ---------------------------------------------------------------------------
# Internal: routing rule implementations
# ---------------------------------------------------------------------------


def _check_error_rule(state: AppState) -> str | None:
    """Rule 1: route to error_handler if state["errors"] is non-empty."""
    errors: list[dict[str, Any]] | None = state.get("errors")
    if errors and len(errors) > 0:
        return TARGET_ERROR_HANDLER
    return None


def _check_scope_rule(state: AppState) -> str | None:
    """Rule 2: route to scope_enforcer if scope_verified is not True."""
    scope_verified: bool | None = state.get("scope_verified")
    if scope_verified is not True:
        return TARGET_SCOPE_ENFORCER
    return None


def _check_planner_override(state: AppState) -> str | None:
    """Rule 3: one-shot Planner directive consumption.

    Fires only on the cycle immediately following a NEW Planner call
    (detected via phase_history growth). On every subsequent cycle with
    the same stale next_phase value, returns None so lower-priority
    rules / the Rule 8 fallback get a turn instead of looping forever.
    """
    next_phase: str | None = state.get("next_phase")
    if not (next_phase is not None and isinstance(next_phase, str) and next_phase.strip()):
        return None

    session_id: str = state.get("session_id") or "_default"
    phase_history: list[Any] = state.get("phase_history") or []
    consumed: int = _consumed_phase_history_len.get(session_id, 0)

    if len(phase_history) <= consumed:
        return None

    _consumed_phase_history_len[session_id] = len(phase_history)
    return next_phase.strip()


def _check_execution_needed(state: AppState) -> str | None:
    """Rule 4: active_payload_id set, no matching execution result yet."""
    active_payload_id: str | None = state.get("active_payload_id")
    if not active_payload_id:
        return None

    execution_results: list[ExecutionResult] | None = state.get("execution_results")
    if not execution_results:
        return TARGET_EXECUTION_SANDBOX

    has_result_for_active: bool = any(
        er.payload_id == active_payload_id for er in execution_results
    )
    if not has_result_for_active:
        return TARGET_EXECUTION_SANDBOX

    return None


def _check_validation_needed(state: AppState) -> str | None:
    """Rule 5: last execution result has no matching validation report."""
    execution_results: list[ExecutionResult] | None = state.get("execution_results")
    if not execution_results:
        return None

    last_execution: ExecutionResult = execution_results[-1]
    last_payload_id: str = last_execution.payload_id

    validation_reports: list[ValidationReport] | None = state.get("validation_reports")
    if not validation_reports:
        return TARGET_VALIDATOR

    has_validation: bool = any(
        vr.payload_id == last_payload_id for vr in validation_reports
    )
    if not has_validation:
        return TARGET_VALIDATOR

    return None


def _check_optimization_needed(state: AppState) -> str | None:
    """Rule 6: ANY validation report is INCONCLUSIVE -> payload_optimizer.

    Scans ALL reports (not just the last) since parallel validation can
    return reports out of order.
    """
    validation_reports: list[ValidationReport] | None = state.get("validation_reports")
    if not validation_reports:
        return None

    for report in validation_reports:
        if report.verdict == ValidationVerdict.INCONCLUSIVE:
            return TARGET_PAYLOAD_OPTIMIZER

    return None


def _check_cvss_needed(state: AppState) -> str | None:
    """Rule 7: last validation is TRUE_POSITIVE and its Finding has no CVSS.

    Checks the LATEST matching finding (not the first) to avoid an
    infinite loop when cvss_engine appends an updated Finding rather
    than replacing the old one.
    """
    validation_reports: list[ValidationReport] | None = state.get("validation_reports")
    if not validation_reports:
        return None

    last_report: ValidationReport = validation_reports[-1]
    if last_report.verdict != ValidationVerdict.TRUE_POSITIVE:
        return None

    confirmed_findings: list[Finding] | None = state.get("confirmed_findings")
    if not confirmed_findings:
        return None

    matching_findings: list[Finding] = [
        f for f in confirmed_findings if f.payload_id == last_report.payload_id
    ]
    if not matching_findings:
        return None

    latest_finding: Finding = matching_findings[-1]
    if latest_finding.cvss is None:
        return TARGET_CVSS_ENGINE

    return None


__all__ = [
    "route_next_phase", "ALL_TARGETS",
    "TARGET_ERROR_HANDLER", "TARGET_SCOPE_ENFORCER", "TARGET_PLANNER",
    "TARGET_RECON_PARSER", "TARGET_CRAWLER_PARSER", "TARGET_WEB_FILTER",
    "TARGET_HYPOTHESIS_ANALYZER", "TARGET_PAYLOAD_GENERATOR",
    "TARGET_PAYLOAD_OPTIMIZER", "TARGET_EXECUTION_SANDBOX", "TARGET_VALIDATOR",
    "TARGET_KNOWLEDGE_RAG", "TARGET_CVSS_ENGINE", "TARGET_BUSINESS_IMPACT",
    "TARGET_REPORTER", "TARGET_MEMORY_SUMMARIZER",
]