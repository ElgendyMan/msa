"""
src/graph/builder.py
====================

The **capstone** module that assembles the 16-node LangGraph
``StateGraph`` into a compiled, runnable application.

This module wires together every node, edge, and conditional routing
rule defined across the framework:

- **16 agent nodes** (Nodes 1-16) are registered via
  ``workflow.add_node()``.
- **1 dummy error_handler node** clears the error list so the graph
  doesn't infinite-loop on persistent errors.
- **1 conditional edge** from the Orchestrator uses
  :func:`~src.agents.orchestrator.route_next_phase` to route to the
  next node based on the current state.
- **15 normal edges** from every non-Orchestrator, non-entry node back
  to the Orchestrator, forming the core loop:

      Node → Orchestrator → (conditional) → Next Node → Orchestrator → ...

The entry point is ``"scope_enforcer"`` (Node 1). Scope enforcement
must run first — nothing else executes until the target is verified
as in-scope. After scope enforcement, the graph flows to the
Orchestrator, which uses its deterministic rules + the Planner's
``next_phase`` to route through the pipeline.

Graph topology
--------------

::

    START
      ↓
    scope_enforcer ──→ orchestrator ←────────────────────────┐
                         │                                     │
                         ├──→ error_handler ──→ orchestrator   │
                         ├──→ recon_parser ────→ orchestrator   │
                         ├──→ crawler_parser ──→ orchestrator   │
                         ├──→ web_filter ──────→ orchestrator   │
                         ├──→ planner ─────────→ orchestrator   │
                         ├──→ hypothesis_analyzer → orchestrator│
                         ├──→ payload_generator → orchestrator  │
                         ├──→ payload_optimizer → orchestrator  │
                         ├──→ execution_sandbox → orchestrator  │
                         ├──→ validator ────────→ orchestrator  │
                         ├──→ knowledge_rag ────→ orchestrator  │
                         ├──→ cvss_engine ──────→ orchestrator  │
                         ├──→ business_impact ──→ orchestrator  │
                         ├──→ reporter ─────────→ orchestrator  │
                         ├──→ memory_summarizer → orchestrator  │
                         └──→ END (when route_next_phase returns "complete")

The Orchestrator is the **hub** of the graph. Every node (except the
entry point and the error handler) flows back to it. The Orchestrator's
:func:`~src.agents.orchestrator.route_next_phase` function reads the
state and returns the name of the next node to route to.

Conditional edges
-----------------
The Orchestrator uses ``add_conditional_edges`` with
:func:`~src.agents.orchestrator.route_next_phase` as the routing
function. The routing map maps every target in
:data:`~src.agents.orchestrator.ALL_TARGETS` to its corresponding node
name, plus maps ``"complete"`` to ``END``.

Error handler
-------------
The ``"error_handler"`` node is a simple async function that returns
``{"errors": []}`` — clearing the error list so the Orchestrator's
Rule 1 (route to error_handler if errors exist) doesn't fire again.
The real error-handling logic (retry, route to Payload Optimizer,
abort) is implemented in the Orchestrator's routing rules, not in the
error handler itself. The error handler's job is purely to break the
error loop by clearing the list.

    TODO (future): The error handler should inspect the last error and
    decide whether to:
    - Route to the Payload Optimizer (for WAFBlockError).
    - Retry the failed node (for transient errors).
    - Abort the session (for unrecoverable errors).
    For now, it simply clears the errors and lets the Orchestrator
    re-plan.

Usage
-----
::

    from src.graph.builder import build_graph

    app = build_graph()
    result = await app.ainvoke(initial_state)
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.business_impact import evaluate_impact
from src.agents.crawler_executor import run_crawler
from src.agents.crawler_parser import parse_crawler_data
from src.agents.cvss_engine import calculate_cvss
from src.agents.execution_sandbox import execute_payload
from src.agents.hypothesis_analyzer import analyze_hypotheses
from src.agents.knowledge_rag import retrieve_knowledge
from src.agents.memory_summarizer import summarize_memory
from src.agents.orchestrator import ALL_TARGETS, route_next_phase
from src.agents.payload_generator import generate_payloads
from src.agents.payload_optimizer import optimize_payload
from src.agents.planner import plan_next_step
from src.agents.recon_executor import run_recon
from src.agents.recon_parser import parse_recon_data
from src.agents.reporter import generate_report
from src.agents.scope_enforcer import enforce_scope
from src.agents.validator import validate_execution
from src.agents.web_filter import filter_web_only
from src.shared.logging import get_logger
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Node name constants
# ---------------------------------------------------------------------------

#: Every node name in the graph. Used for validation and documentation.
NODE_SCOPE_ENFORCER: str = "scope_enforcer"
NODE_ORCHESTRATOR: str = "orchestrator"
NODE_ERROR_HANDLER: str = "error_handler"
NODE_PLANNER: str = "planner"
NODE_RECON_EXECUTOR: str = "recon_executor"
NODE_RECON_PARSER: str = "recon_parser"
NODE_CRAWLER_EXECUTOR: str = "crawler_executor"
NODE_CRAWLER_PARSER: str = "crawler_parser"
NODE_WEB_FILTER: str = "web_filter"
NODE_HYPOTHESIS_ANALYZER: str = "hypothesis_analyzer"
NODE_PAYLOAD_GENERATOR: str = "payload_generator"
NODE_PAYLOAD_OPTIMIZER: str = "payload_optimizer"
NODE_EXECUTION_SANDBOX: str = "execution_sandbox"
NODE_VALIDATOR: str = "validator"
NODE_KNOWLEDGE_RAG: str = "knowledge_rag"
NODE_CVSS_ENGINE: str = "cvss_engine"
NODE_BUSINESS_IMPACT: str = "business_impact"
NODE_REPORTER: str = "reporter"
NODE_MEMORY_SUMMARIZER: str = "memory_summarizer"

#: The set of all node names registered in the graph.
ALL_NODES: frozenset[str] = frozenset(
    {
        NODE_SCOPE_ENFORCER,
        NODE_ORCHESTRATOR,
        NODE_ERROR_HANDLER,
        NODE_PLANNER,
        NODE_RECON_EXECUTOR,
        NODE_RECON_PARSER,
        NODE_CRAWLER_EXECUTOR,
        NODE_CRAWLER_PARSER,
        NODE_WEB_FILTER,
        NODE_HYPOTHESIS_ANALYZER,
        NODE_PAYLOAD_GENERATOR,
        NODE_PAYLOAD_OPTIMIZER,
        NODE_EXECUTION_SANDBOX,
        NODE_VALIDATOR,
        NODE_KNOWLEDGE_RAG,
        NODE_CVSS_ENGINE,
        NODE_BUSINESS_IMPACT,
        NODE_REPORTER,
        NODE_MEMORY_SUMMARIZER,
    }
)

#: The entry point node. Scope enforcement MUST run first.
ENTRY_POINT: str = NODE_SCOPE_ENFORCER


# ---------------------------------------------------------------------------
# Error handler (dummy node)
# ---------------------------------------------------------------------------


async def _error_handler(state: AppState) -> dict[str, Any]:
    """Dummy error handler that clears the error list.

    The Orchestrator's Rule 1 routes here when ``state["errors"]`` is
    non-empty. The error handler clears the list so Rule 1 doesn't fire
    again on the next routing cycle, which would create an infinite loop.

    The REAL error-handling logic lives in the Orchestrator's routing
    rules:
    - :class:`WAFBlockError` → route to Payload Optimizer (Rule 6).
    - :class:`ValidationInconclusiveError` → route to Payload Optimizer.
    - :class:`LLMRateLimitError` → the Orchestrator could implement
      back-off here (TODO: future enhancement).
    - :class:`LLMOutputParsingError` → the Orchestrator could retry
      the failed node (TODO: future enhancement).

    For now, the error handler simply clears the errors and lets the
    Orchestrator re-plan via the Planner fallback.

    Parameters
    ----------
    state:
        The current :class:`~src.shared.state.AppState`. The error
        handler reads ``state["errors"]`` for logging but does NOT
        use the error details for routing decisions.

    Returns
    -------
    dict
        ``{"errors": []}`` — clears the error list so the Orchestrator's
        Rule 1 doesn't fire again.
    """
    log = get_logger("error_handler")

    errors: list[dict[str, Any]] | None = state.get("errors")
    error_count: int = len(errors) if errors else 0

    if errors:
        last_error: dict[str, Any] = errors[-1]
        log.warning(
            "error_handler_clearing_errors",
            error_count=error_count,
            last_error_type=last_error.get("exception_type", "unknown"),
            last_error_message=str(last_error.get("message", ""))[:200],
        )
    else:
        log.info("error_handler_no_errors_to_clear")

    # Clear the errors list so the Orchestrator's Rule 1 (route to
    # error_handler if errors exist) doesn't fire again. The
    # Orchestrator will then fall through to its other rules or the
    # Planner fallback.
    return {"errors": []}


# ---------------------------------------------------------------------------
# Conditional edges routing map
# ---------------------------------------------------------------------------


def _build_routing_map() -> dict[str, str]:
    """Build the conditional edges routing map for the Orchestrator.

    The map translates :func:`~src.agents.orchestrator.route_next_phase`
    return values into LangGraph node names (or ``END``).

    - Every target in :data:`~src.agents.orchestrator.ALL_TARGETS` maps
      to its corresponding node name (which is the same string — the
      Orchestrator returns node names directly).
    - The special value ``"complete"`` (returned by the Planner when
      the engagement is finished) maps to ``END``.

    Returns
    -------
    dict[str, str]
        A mapping from routing target → graph node name (or ``END``).
    """
    routing_map: dict[str, str] = {}

    # Every ALL_TARGETS value maps to itself (the Orchestrator returns
    # node names directly).
    for target in ALL_TARGETS:
        routing_map[target] = target

    # "complete" is a special Planner output that ends the graph.
    routing_map["complete"] = END

    return routing_map


# ---------------------------------------------------------------------------
# Public: build_graph
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """Build and compile the 16-node LangGraph application.

    This function:
    1. Creates a ``StateGraph(AppState)``.
    2. Registers all 16 agent nodes + the dummy error_handler.
    3. Sets the entry point to ``"scope_enforcer"``.
    4. Adds a normal edge from ``scope_enforcer`` → ``orchestrator``.
    5. Adds conditional edges from ``orchestrator`` using
       :func:`~src.agents.orchestrator.route_next_phase`.
    6. Adds normal edges from every other node back to ``orchestrator``
       (the core loop).
    7. Compiles the graph and returns it.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph application that can be invoked via
        ``await app.ainvoke(initial_state)`` or streamed via
        ``app.astream(initial_state)``.

    Raises
    ------
    ValueError
        If the routing map doesn't cover all possible return values
        from :func:`~src.agents.orchestrator.route_next_phase`.
    """
    log = get_logger("graph_builder")

    log.info("graph_build_started")

    # ---------------------------------------------------------------
    # 1. Initialize the workflow.
    # ---------------------------------------------------------------
    workflow: StateGraph = StateGraph(AppState)

    # ---------------------------------------------------------------
    # 2. Register all nodes.
    # ---------------------------------------------------------------
    # Node 1: Scope Enforcer (entry point)
    workflow.add_node(NODE_SCOPE_ENFORCER, enforce_scope)

    # Node 2: Orchestrator (conditional edge router)
    # The Orchestrator is NOT a standard node — it's a conditional edge.
    # However, LangGraph requires a node to exist at the source of
    # conditional edges. We register a pass-through node that does
    # nothing (returns empty dict) so the conditional edge has a source.
    # The ACTUAL routing logic is in route_next_phase, which is passed
    # to add_conditional_edges.
    async def _orchestrator_passthrough(state: AppState) -> dict[str, Any]:
        """Pass-through node for the Orchestrator.

        The Orchestrator's routing logic is implemented in
        :func:`~src.agents.orchestrator.route_next_phase`, which is a
        conditional edge function, not a node function. However,
        LangGraph requires a node to exist at the source of conditional
        edges. This pass-through node satisfies that requirement
        without modifying state.

        The routing decision is made by ``route_next_phase`` AFTER
        this node runs (but the node itself doesn't need to do
        anything).
        """
        return {}

    workflow.add_node(NODE_ORCHESTRATOR, _orchestrator_passthrough)

    # Dummy error handler
    workflow.add_node(NODE_ERROR_HANDLER, _error_handler)

    # Node 3: Planner
    workflow.add_node(NODE_PLANNER, plan_next_step)

    # Node 4: Web Filter
    workflow.add_node(NODE_WEB_FILTER, filter_web_only)

    # Node 5a: Recon Executor (NEW — produces raw_recon_output)
    workflow.add_node(NODE_RECON_EXECUTOR, run_recon)

    # Node 5b: Recon Parser
    workflow.add_node(NODE_RECON_PARSER, parse_recon_data)

    # Node 6a: Crawler Executor (NEW — produces raw_crawler_output)
    workflow.add_node(NODE_CRAWLER_EXECUTOR, run_crawler)

    # Node 6b: Crawler Parser
    workflow.add_node(NODE_CRAWLER_PARSER, parse_crawler_data)

    # Node 7: Memory Summarizer
    workflow.add_node(NODE_MEMORY_SUMMARIZER, summarize_memory)

    # Node 8: Hypothesis Analyzer
    workflow.add_node(NODE_HYPOTHESIS_ANALYZER, analyze_hypotheses)

    # Node 9: Payload Generator
    workflow.add_node(NODE_PAYLOAD_GENERATOR, generate_payloads)

    # Node 10: Payload Optimizer
    workflow.add_node(NODE_PAYLOAD_OPTIMIZER, optimize_payload)

    # Node 11: Execution Sandbox
    workflow.add_node(NODE_EXECUTION_SANDBOX, execute_payload)

    # Node 12: Validator
    workflow.add_node(NODE_VALIDATOR, validate_execution)

    # Node 13: Knowledge RAG
    workflow.add_node(NODE_KNOWLEDGE_RAG, retrieve_knowledge)

    # Node 14: CVSS Engine
    workflow.add_node(NODE_CVSS_ENGINE, calculate_cvss)

    # Node 15: Business Impact Writer
    workflow.add_node(NODE_BUSINESS_IMPACT, evaluate_impact)

    # Node 16: Reporter
    workflow.add_node(NODE_REPORTER, generate_report)

    log.info("graph_nodes_registered", count=len(ALL_NODES))

    # ---------------------------------------------------------------
    # 3. Set the entry point.
    # ---------------------------------------------------------------
    workflow.set_entry_point(ENTRY_POINT)

    log.info("graph_entry_point_set", entry_point=ENTRY_POINT)

    # ---------------------------------------------------------------
    # 4. Add the entry → orchestrator edge.
    # ---------------------------------------------------------------
    # After scope enforcement, flow to the Orchestrator for routing.
    workflow.add_edge(NODE_SCOPE_ENFORCER, NODE_ORCHESTRATOR)

    # ---------------------------------------------------------------
    # 5. Add the error_handler → orchestrator edge.
    # ---------------------------------------------------------------
    # After the error handler clears errors, flow back to the
    # Orchestrator for re-planning.
    workflow.add_edge(NODE_ERROR_HANDLER, NODE_ORCHESTRATOR)

    # ---------------------------------------------------------------
    # 6. Add conditional edges from the Orchestrator.
    # ---------------------------------------------------------------
    routing_map: dict[str, str] = _build_routing_map()

    # Executor nodes are reachable via next_phase (Rule 3) but are NOT
    # in orchestrator.ALL_TARGETS yet (added in the orchestrator.py step).
    # We add them here so the routing map is complete before that step lands.
    routing_map[NODE_RECON_EXECUTOR] = NODE_RECON_EXECUTOR
    routing_map[NODE_CRAWLER_EXECUTOR] = NODE_CRAWLER_EXECUTOR

    workflow.add_conditional_edges(
        NODE_ORCHESTRATOR,
        route_next_phase,
        routing_map,
    )

    log.info(
        "graph_conditional_edges_added",
        source=NODE_ORCHESTRATOR,
        routing_targets=len(routing_map),
    )

    # ---------------------------------------------------------------
    # 7. Add normal edges from every other node back to the Orchestrator.
    # ---------------------------------------------------------------
    # Executor nodes are intentionally EXCLUDED from this list: they
    # flow directly to their respective parser, not back to the
    # Orchestrator.  The parser then loops back to the Orchestrator.
    #
    #   orchestrator → recon_executor → recon_parser → orchestrator
    #   orchestrator → crawler_executor → crawler_parser → orchestrator
    nodes_that_loop_back: list[str] = [
        NODE_PLANNER,
        NODE_RECON_PARSER,
        NODE_CRAWLER_PARSER,
        NODE_WEB_FILTER,
        NODE_HYPOTHESIS_ANALYZER,
        NODE_PAYLOAD_GENERATOR,
        NODE_PAYLOAD_OPTIMIZER,
        NODE_EXECUTION_SANDBOX,
        NODE_VALIDATOR,
        NODE_KNOWLEDGE_RAG,
        NODE_CVSS_ENGINE,
        NODE_BUSINESS_IMPACT,
        NODE_REPORTER,
        NODE_MEMORY_SUMMARIZER,
    ]

    for node_name in nodes_that_loop_back:
        workflow.add_edge(node_name, NODE_ORCHESTRATOR)

    # Direct executor → parser edges (bypass the Orchestrator).
    workflow.add_edge(NODE_RECON_EXECUTOR, NODE_RECON_PARSER)
    workflow.add_edge(NODE_CRAWLER_EXECUTOR, NODE_CRAWLER_PARSER)

    log.info(
        "graph_loop_edges_added",
        loop_edge_count=len(nodes_that_loop_back),
        direct_edges=2,
        target=NODE_ORCHESTRATOR,
    )

    # ---------------------------------------------------------------
    # 8. Compile and return.
    # ---------------------------------------------------------------
    app: Any = workflow.compile()

    log.info(
        "graph_build_complete",
        total_nodes=len(ALL_NODES),
        total_edges=2 + len(nodes_that_loop_back),  # entry→orch + err→orch + loops
        conditional_edges=len(routing_map),
        entry_point=ENTRY_POINT,
    )

    return app


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "build_graph",
    "ALL_NODES",
    "ENTRY_POINT",
    "NODE_SCOPE_ENFORCER",
    "NODE_ORCHESTRATOR",
    "NODE_ERROR_HANDLER",
    "NODE_PLANNER",
    "NODE_RECON_PARSER",
    "NODE_CRAWLER_PARSER",
    "NODE_WEB_FILTER",
    "NODE_HYPOTHESIS_ANALYZER",
    "NODE_PAYLOAD_GENERATOR",
    "NODE_PAYLOAD_OPTIMIZER",
    "NODE_EXECUTION_SANDBOX",
    "NODE_VALIDATOR",
    "NODE_KNOWLEDGE_RAG",
    "NODE_CVSS_ENGINE",
    "NODE_BUSINESS_IMPACT",
    "NODE_REPORTER",
    "NODE_MEMORY_SUMMARIZER",
]