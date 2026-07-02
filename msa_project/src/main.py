"""
src/main.py
===========

CLI entry point for the Zero-Budget Autonomous Web Pentesting Framework.

This is the missing piece that wires together everything already built:
``Settings`` (config), ``ScopeConfig``/``Target`` (schemas), the initial
``AppState``, and the compiled LangGraph app from ``build_graph()``.

Referenced by ``pyproject.toml`` as:

    [project.scripts]
    pentest = "src.main:cli"

Usage
-----
::

    pentest --target https://example.com/ [--scope scope.json] [--max-cycles 200]

Or directly:

::

    python -m src.main --target https://example.com/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.graph.builder import build_graph
from src.shared.config import REPORTS_DIR, SCOPE_FILE_PATH, settings
from src.shared.exceptions import (
    ConfigurationError,
    PentestFrameworkError,
    ScopeViolationError,
)
from src.shared.logging import get_logger, setup_logging
from src.shared.schemas import HTTPMethod, Phase, ScopeConfig, Target
from src.shared.state import AppState

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Hard ceiling on Orchestrator round-trips. Even with the next_phase
#: one-shot fix, a misbehaving LLM provider chain could in theory keep
#: returning new (but unproductive) directives forever. This is the
#: last line of defense against a runaway session burning API quota.
DEFAULT_MAX_CYCLES: int = 200


# ---------------------------------------------------------------------------
# scope.json loading
# ---------------------------------------------------------------------------


def load_scope_config(scope_path: Path) -> ScopeConfig:
    """Load and validate ``scope.json`` into a :class:`ScopeConfig`.

    Raises
    ------
    ConfigurationError
        If the file is missing, unreadable, or fails schema validation.
    """
    if not scope_path.exists():
        raise ConfigurationError(
            f"Scope file not found at '{scope_path}'. Create it before "
            "running the framework — see scope.json.example, or run "
            "with --target plus an inline --allow-domain flag if you "
            "are just smoke-testing the graph wiring.",
            details={"scope_path": str(scope_path)},
        )

    try:
        raw: dict[str, Any] = json.loads(scope_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigurationError(
            f"Failed to read/parse scope file '{scope_path}': {exc}",
            details={"scope_path": str(scope_path)},
        ) from exc

    try:
        return ScopeConfig(**raw)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigurationError(
            f"scope.json at '{scope_path}' failed schema validation: {exc}",
            details={"scope_path": str(scope_path)},
        ) from exc


# ---------------------------------------------------------------------------
# Initial AppState construction
# ---------------------------------------------------------------------------


def build_initial_state(
    target_url: str,
    method: HTTPMethod,
    scope: ScopeConfig,
    session_id: str,
    max_retries: int,
) -> AppState:
    """Build the seed ``AppState`` LangGraph starts from.

    Only the fields the entry node (``scope_enforcer``) actually reads
    are populated here. Every other channel is left absent so its
    reducer's identity behavior applies (``operator.add`` channels
    start as if empty; scalar channels start as ``None``/missing).
    """
    now = datetime.now(UTC)
    target = Target(url=target_url, method=method)

    state: AppState = {
        "session_id": session_id,
        "created_at": now,
        "updated_at": now,
        "current_phase": Phase.INITIALIZATION,
        "next_phase": None,
        "scope": scope,
        "target": target,
        "scope_verified": False,
        "retry_count": 0,
        "max_retries": max_retries,
    }
    return state


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


async def run_session(
    target_url: str,
    method: HTTPMethod,
    scope_path: Path,
    max_cycles: int,
) -> AppState:
    """Build the graph, seed the state, and drive the session to
    completion (or until ``max_cycles`` is exhausted).

    Returns
    -------
    AppState
        The final state after the graph reaches ``END`` or the cycle
        cap is hit.
    """
    log = get_logger("main")
    session_id: str = str(uuid.uuid4())

    log.info(
        "session_starting",
        session_id=session_id,
        target_url=target_url,
        scope_path=str(scope_path),
        max_cycles=max_cycles,
    )

    scope: ScopeConfig = load_scope_config(scope_path)
    initial_state: AppState = build_initial_state(
        target_url=target_url,
        method=method,
        scope=scope,
        session_id=session_id,
        max_retries=settings.SESSION_MAX_RETRIES,
    )

    app: Any = build_graph()

    # recursion_limit is LangGraph's own cycle cap (counts every node
    # hop, not "macro cycles"). We give it generous headroom over
    # max_cycles since each macro cycle is itself several node hops
    # (e.g. node -> orchestrator -> next node).
    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": max(50, max_cycles * 4),
    }

    final_state: AppState = await app.ainvoke(initial_state, config=config)

    log.info(
        "session_complete",
        session_id=session_id,
        confirmed_findings=len(final_state.get("confirmed_findings") or []),
        rejected_findings=len(final_state.get("rejected_findings") or []),
        errors=len(final_state.get("errors") or []),
    )

    return final_state


def _write_report(final_state: AppState, session_id: str) -> Path | None:
    """Persist ``final_state['final_report']`` (if any) to disk."""
    report: str | None = final_state.get("final_report")
    if not report:
        return None

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path: Path = REPORTS_DIR / f"report_{session_id}.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pentest",
        description="Zero-Budget Autonomous Web Pentesting Framework",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target URL to test, e.g. https://example.com/",
    )
    parser.add_argument(
        "--method",
        default="GET",
        choices=[m.value for m in HTTPMethod],
        help="Initial HTTP method for the target (default: GET).",
    )
    parser.add_argument(
        "--scope",
        default=str(SCOPE_FILE_PATH),
        help=f"Path to scope.json (default: {SCOPE_FILE_PATH}).",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=DEFAULT_MAX_CYCLES,
        help=f"Max Orchestrator macro-cycles before aborting (default: {DEFAULT_MAX_CYCLES}).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override LOG_LEVEL from .env for this run.",
    )
    return parser


def cli() -> None:
    """Synchronous console-script entry point (``pentest`` command).

    Wraps :func:`run_session` in ``asyncio.run`` and translates
    framework exceptions into clean CLI error output + exit codes
    instead of raw tracebacks.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    log = get_logger("cli")

    try:
        final_state: AppState = asyncio.run(
            run_session(
                target_url=args.target,
                method=HTTPMethod(args.method),
                scope_path=Path(args.scope),
                max_cycles=args.max_cycles,
            )
        )
    except ScopeViolationError as exc:
        log.error("scope_violation", error=str(exc), details=exc.details)
        print(f"[SCOPE VIOLATION] {exc}", file=sys.stderr)
        sys.exit(2)
    except ConfigurationError as exc:
        log.error("configuration_error", error=str(exc), details=exc.details)
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        sys.exit(3)
    except PentestFrameworkError as exc:
        log.error("framework_error", error=str(exc), details=exc.details)
        print(f"[FRAMEWORK ERROR] {exc}", file=sys.stderr)
        sys.exit(4)
    except KeyboardInterrupt:
        log.warning("session_interrupted_by_user")
        print("\n[INTERRUPTED] Session cancelled by user.", file=sys.stderr)
        sys.exit(130)

    session_id: str = final_state.get("session_id") or "unknown"
    report_path: Path | None = _write_report(final_state, session_id)

    findings = final_state.get("confirmed_findings") or []
    print(f"\nSession {session_id} complete.")
    print(f"Confirmed findings: {len(findings)}")
    if report_path is not None:
        print(f"Report written to: {report_path}")
    else:
        print("No report was generated (final_report empty).")


if __name__ == "__main__":
    cli()


__all__ = [
    "cli",
    "run_session",
    "load_scope_config",
    "build_initial_state",
    "DEFAULT_MAX_CYCLES",
]