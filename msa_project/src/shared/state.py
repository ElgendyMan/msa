"""
shared/state.py
===============

LangGraph ``AppState`` definition — the single typed state object every
node in the 16-node framework reads from and writes to.

Why a TypedDict (not a Pydantic model)?
---------------------------------------
LangGraph's ``StateGraph`` expects a ``TypedDict``. It uses the dict's key
set as the channel contract between nodes and uses
``Annotated[T, reducer]`` to decide how a node's return value is merged
into the running state.

Why ``Annotated[list[X], operator.add]``?
-----------------------------------------
By default, LangGraph OVERWRITES a channel when a node returns a value
for it. For accumulating channels (hypotheses, findings, execution
results, errors, ...) we instead supply ``operator.add`` as the reducer so
that a node's return value is APPENDED to whatever was already there.
This is the canonical LangGraph pattern for evidence accumulation.

Why ``Annotated[list[BaseMessage], add_messages]``?
---------------------------------------------------
LangGraph ships a special ``add_messages`` reducer that handles the
message-merging semantics LLMs need (deduplication by ``id``, replacement
on edit, append otherwise). Use it for any channel that accumulates
LangChain messages.

This file is deliberately small: it only declares the state shape. All
data structures referenced here are defined in ``shared/schemas.py`` so
they can be reused outside LangGraph (FastAPI endpoints, CLI tools, test
fixtures, ...) without dragging in LangGraph itself.
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.shared.schemas import (
    BusinessImpact,
    CVSSResult,
    CrawlerResult,
    ExecutionResult,
    Finding,
    Hypothesis,
    Payload,
    Phase,
    RAGContext,
    ReconResult,
    ScopeConfig,
    Target,
    ValidationReport,
)


def merge_findings(existing: list[Finding], new: list[Finding]) -> list[Finding]:
    """Merge incoming findings into the existing list by unique ``id``.

    This replaces the default ``operator.add`` reducer for
    ``confirmed_findings`` to prevent duplicates. When the CVSS Engine or
    Business Impact Writer returns an updated finding with the same ``id``,
    this function replaces the old entry rather than appending a second copy.

    Parameters
    ----------
    existing:
        The current list of findings already in the state channel.
    new:
        The list of findings returned by the most recently executed node.

    Returns
    -------
    list[Finding]
        A deduplicated list where any finding whose ``id`` matches an entry in
        ``new`` is replaced by the newer version.
    """
    merged: dict[str, Finding] = {f.id: f for f in existing}
    for item in new:
        merged[item.id] = item  # overwrite / insert
    return list(merged.values())


class AppState(TypedDict, total=False):
    """Canonical LangGraph state shared by every node in the framework.

    ``total=False`` so that any node can return a *partial* dict containing
    only the keys it actually modified — LangGraph will merge it into the
    running state using the per-key reducer. This is what makes the
    Single-Responsibility Principle enforceable: each node touches only
    its own slice of state.

    Field groups
    ------------
    * Identity & provenance  — ``session_id``, ``created_at``, ``updated_at``
    * Phase management       — ``current_phase``, ``next_phase``,
                               ``phase_history``
    * Scope & target         — ``scope``, ``target``, ``scope_verified``
    * Recon                  — ``raw_recon_output``, ``recon_data``
    * Crawling               — ``raw_crawler_output``, ``crawler_data``
    * Memory                 — ``memory_summary``, ``context_window_pressure``
    * Hypothesis pipeline    — ``hypotheses``, ``active_hypothesis_id``
    * Payload pipeline       — ``payloads``, ``active_payload_id``
    * WAF detection          — ``waf_detected``, ``waf_signature``
    * Execution & validation — ``execution_results``, ``validation_reports``
    * Knowledge RAG          — ``rag_context``
    * Scoring & impact       — ``cvss_results``, ``business_impacts``
    * Findings & reporting   — ``confirmed_findings``, ``rejected_findings``,
                               ``final_report``
    * LLM message log        — ``messages`` (LangGraph-managed)
    * Error handling         — ``errors``, ``retry_count``, ``max_retries``
    """

    # ---- Identity & provenance ----
    session_id: str
    created_at: datetime
    updated_at: datetime

    # ---- Phase management ----
    current_phase: Phase
    next_phase: Phase | None
    phase_history: Annotated[list[Phase], operator.add]

    # ---- Scope & target ----
    scope: ScopeConfig
    target: Target
    scope_verified: bool

    # ---- Recon (raw input -> structured) ----
    raw_recon_output: str
    recon_data: ReconResult

    # ---- Crawling (raw input -> structured) ----
    raw_crawler_output: str
    crawler_data: CrawlerResult

    # ---- Memory summarizer ----
    memory_summary: str
    context_window_pressure: float

    # ---- Hypothesis pipeline ----
    hypotheses: Annotated[list[Hypothesis], operator.add]
    active_hypothesis_id: str | None

    # ---- Payload pipeline ----
    payloads: Annotated[list[Payload], operator.add]
    active_payload_id: str | None

    # ---- WAF detection ----
    waf_detected: bool
    waf_signature: str | None

    # ---- Execution & validation ----
    execution_results: Annotated[list[ExecutionResult], operator.add]
    validation_reports: Annotated[list[ValidationReport], operator.add]

    # ---- Knowledge RAG ----
    rag_context: RAGContext | None

    # ---- Scoring & impact ----
    cvss_results: Annotated[list[CVSSResult], operator.add]
    business_impacts: Annotated[list[BusinessImpact], operator.add]

    # ---- Findings & reporting ----
    confirmed_findings: Annotated[list[Finding], merge_findings]
    rejected_findings: Annotated[list[Finding], operator.add]
    final_report: str | None

    # ---- LLM message log (LangGraph-managed; appends across turns) ----
    messages: Annotated[list[BaseMessage], add_messages]

    # ---- Error handling ----
    errors: Annotated[list[dict[str, Any]], operator.add]
    retry_count: int
    max_retries: int
