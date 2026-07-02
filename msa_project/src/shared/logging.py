"""
shared/logging.py
=================

Structured logging configuration for the Zero-Budget Autonomous Web
Pentesting Framework.

This module provides three public entry points:

    >>> from src.shared.logging import setup_logging, get_logger
    >>> setup_logging()                      # call once at app startup
    >>> log = get_logger("scope_enforcer")   # call from anywhere
    >>> log = log.bind(session_id="abc123", node_name="scope_enforcer",
    ...               target_url="https://example.com/")
    >>> log.info("scope_verified")
    2026-06-29 14:32:11 [info     ] scope_verified       node_name=scope_enforcer session_id=abc123 target_url=https://example.com/

Design
------
- Uses :mod:`structlog` for the high-level API (bound context, key-value
  pairs, renderers) and the standard :mod:`logging` module as the
  underlying sink. This combination gives us:
    * Structured key-value context on every log line.
    * Compatibility with any library that uses ``logging.getLogger``
      (httpx, playwright, langchain, ...) — we can silence them via
      standard ``Logger.setLevel()``.
    * A beautiful colorized console renderer for local development via
      :class:`structlog.dev.ConsoleRenderer`.
- :func:`setup_logging` is **idempotent**. Calling it multiple times
  (e.g. in tests) does not duplicate handlers or reset bindings.
- :func:`get_logger` returns a :class:`structlog.stdlib.BoundLogger`
  with a ``.bind()`` method for adding contextual key-value pairs.
  Every agent should bind ``session_id``, ``node_name``, and
  ``target_url`` at the top of its node function — see the usage
  pattern below.

Standard context keys
---------------------
The framework defines a closed set of context keys that agents SHOULD
bind consistently. Keeping this set small and stable makes log
filtering and dashboards tractable.

    session_id      str     The LangGraph session ID. Bound once per
                            session, never changes.
    node_name       str     The name of the LangGraph node emitting the
                            log. Bound at the top of each node fn.
    target_url      str     The current target URL. Bound per request.
    hypothesis_id   str     (optional) The hypothesis being worked on.
    payload_id      str     (optional) The payload being executed.
    finding_id      str     (optional) The finding being scored.

Noise reduction
---------------
Third-party libraries are notoriously chatty at INFO level. The
:func:`_silence_noisy_libraries` helper forces their loggers to
``WARNING`` (or ``ERROR`` for the worst offenders) so the operator's
console shows only framework-emitted lines. This is applied inside
:func:`setup_logging` and cannot be overridden by env vars — if a
developer really wants to see ``httpx`` DEBUG output, they can call
``logging.getLogger("httpx").setLevel(logging.DEBUG)`` after
``setup_logging()`` returns.

Dependency handling
-------------------
If :mod:`structlog` is not installed, :func:`setup_logging` raises
:class:`~src.shared.exceptions.DependencyMissingError` with an
actionable install hint. The module itself imports cleanly without
structlog so that ``src.shared.__init__`` can re-export the function
symbols without crashing on a fresh checkout. This is the same pattern
used by :mod:`src.shared.llm`.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Final

from src.shared.config import settings
from src.shared.exceptions import DependencyMissingError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Name of the root framework logger. All framework loggers are
#: descendants of this name, so setting its level propagates downward.
FRAMEWORK_ROOT_LOGGER_NAME: Final[str] = "src"


#: Standard context keys that agents SHOULD bind consistently. This is
#: a closed set — adding a new key here is a documentation change.
STANDARD_CONTEXT_KEYS: Final[tuple[str, ...]] = (
    "session_id",
    "node_name",
    "target_url",
    "hypothesis_id",
    "payload_id",
    "finding_id",
    "phase",
    "duration_ms",
    "attempt",
    "error_type",
)


#: Third-party libraries that are chatty at INFO/DEBUG level. We force
#: them to WARNING so the operator's console shows only framework-
#: emitted lines. The value is the minimum level to enforce.
_NOISY_LIBRARIES: Final[dict[str, int]] = {
    # HTTP clients — emit a line per request at INFO
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "urllib3": logging.WARNING,
    "urllib3.connectionpool": logging.WARNING,
    "requests": logging.WARNING,
    "aiohttp": logging.WARNING,
    "aiohttp.access": logging.WARNING,
    # Asyncio internals — DEBUG output is enormous and unhelpful
    "asyncio": logging.WARNING,
    # LangChain + LiteLLM — chain-of-thought is logged at INFO
    "langchain": logging.WARNING,
    "langchain_core": logging.WARNING,
    "langchain_openai": logging.WARNING,
    "langchain_google_genai": logging.WARNING,
    "langgraph": logging.WARNING,
    "LiteLLM": logging.WARNING,
    "LiteLLM Router": logging.WARNING,
    "LiteLLM Proxy": logging.WARNING,
    # OpenAI SDK — retries are logged at INFO
    "openai": logging.WARNING,
    "openai._base_client": logging.WARNING,
    # Google GenAI SDK
    "google": logging.WARNING,
    "google.genai": logging.WARNING,
    "google.ai.generativelanguage": logging.WARNING,
    # Playwright — emits a line per browser action
    "playwright": logging.WARNING,
    # Qdrant client
    "qdrant_client": logging.WARNING,
    # Watchfiles / uvicorn / fastapi lifecycle noise
    "watchfiles": logging.WARNING,
    "uvicorn.access": logging.WARNING,
    "uvicorn.error": logging.WARNING,
    "fastapi": logging.WARNING,
    # Pydantic — emits deprecation warnings at WARNING already, but
    # also some INFO lines on schema rebuilds
    "pydantic": logging.WARNING,
}


#: Sentinel returned by :func:`_get_bool_flag` helpers. We don't use
#: ``True``/``False`` directly to avoid confusion with the env value.
_SETUP_COMPLETE_FLAG: Final[str] = "_pentest_logging_setup_complete"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(
    *,
    level: str | int | None = None,
    force: bool = False,
) -> None:
    """Configure structured logging for the entire framework.

    Call this exactly once at application startup, before any agent
    code runs. Idempotent: subsequent calls are no-ops unless
    ``force=True``.

    Parameters
    ----------
    level:
        Optional override for the root framework logger level. Accepts
        a level name (``"DEBUG"``, ``"INFO"``, ...) or numeric level.
        If None, falls back to :attr:`settings.LOG_LEVEL`.
    force:
        If True, re-run the full configuration even if a previous call
        has already configured logging. Useful in tests where the
        LOG_LEVEL needs to be reset between test cases.

    Raises
    ------
    DependencyMissingError
        If :mod:`structlog` is not installed.
    """
    # Lazy import — see module docstring for rationale.
    try:
        import structlog  # noqa: F401  (imported for side-effect + isinstance check below)
    except ImportError as exc:
        raise DependencyMissingError(
            "Cannot configure structured logging: the 'structlog' "
            "package is not installed. Install it with "
            "'pip install structlog>=24.1.0'.",
            dependency="structlog",
            install_hint="pip install structlog>=24.1.0",
        ) from exc

    # Idempotency guard. We mark the root logger with a sentinel
    # attribute so re-calls are cheap and side-effect-free.
    root = logging.getLogger()
    already_configured = getattr(root, _SETUP_COMPLETE_FLAG, False)
    if already_configured and not force:
        return

    # ---------------------------------------------------------------
    # 1. Resolve the effective log level.
    # ---------------------------------------------------------------
    effective_level = _resolve_level(level)

    # ---------------------------------------------------------------
    # 2. Configure stdlib logging as the sink.
    # ---------------------------------------------------------------
    _configure_stdlib_root(effective_level)

    # ---------------------------------------------------------------
    # 3. Configure structlog's processor chain.
    # ---------------------------------------------------------------
    _configure_structlog(effective_level)

    # ---------------------------------------------------------------
    # 4. Silence noisy third-party libraries. This runs AFTER stdlib
    #    configuration so the level overrides stick.
    # ---------------------------------------------------------------
    _silence_noisy_libraries()

    # ---------------------------------------------------------------
    # 5. Mark setup complete.
    # ---------------------------------------------------------------
    setattr(root, _SETUP_COMPLETE_FLAG, True)

    # Emit a single startup line so the operator can confirm logging
    # is wired up. We use a fresh structlog logger here (not via
    # get_logger) so this works even if get_logger has not yet been
    # called by any agent.
    boot_logger = structlog.stdlib.get_logger("logging").bind(
        component="logging",
        event="logging_configured",
        level=logging.getLevelName(effective_level),
    )
    # Use stdlib .log() to avoid any processor-chain surprises before
    # structlog is fully wired.
    boot_logger.info("logging_configured", level_name=logging.getLevelName(effective_level))


def get_logger(name: str, **initial_context: Any) -> Any:
    """Return a bound :mod:`structlog` logger.

    Parameters
    ----------
    name:
        Logger name. Convention: use the LangGraph node name
        (``"scope_enforcer"``, ``"web_filter"``, ...) so log lines are
        easy to filter by node. The name is automatically prefixed
        with the framework root (``src.``) so it inherits the
        framework's log level.
    **initial_context:
        Optional key-value pairs to bind immediately. Equivalent to::

            log = get_logger("foo")
            log = log.bind(session_id="abc", node_name="foo")

    Returns
    -------
    structlog.stdlib.BoundLogger
        A bound logger with the standard structlog API (``.info()``,
        ``.debug()``, ``.warning()``, ``.error()``, ``.critical()``,
        ``.bind()``, ``.new()``).

    Notes
    -----
    If :func:`setup_logging` has not been called yet, the returned
    logger still works — structlog falls back to its default
    configuration. This is intentional: it lets unit tests import
    ``get_logger`` without booting the full logging stack. However,
    the output will not be colorized and may be redundant; production
    code should always call :func:`setup_logging` first.

    The returned logger is a *new* :class:`BoundLogger` instance each
    call. Binding context does not mutate global state — see
    :func:`bind_context` for a global-context helper if you need that.
    """
    # Lazy import — same rationale as setup_logging.
    try:
        import structlog
    except ImportError:
        # If structlog is missing AND setup_logging has not been called,
        # we cannot return a real structlog logger. Fall back to a thin
        # adapter around stdlib logging so the framework at least
        # produces *some* output. This is the only graceful-degradation
        # path in the framework — every other module raises
        # DependencyMissingError on missing structlog.
        return _StdlibFallbackLogger(name, **initial_context)

    # Prefix with framework root so the level propagates.
    full_name = name if name.startswith(FRAMEWORK_ROOT_LOGGER_NAME + ".") else f"{FRAMEWORK_ROOT_LOGGER_NAME}.{name}"
    logger = structlog.stdlib.get_logger(full_name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


def bind_context(**kwargs: Any) -> None:
    """Bind context variables GLOBALLY — every subsequent log line from
    every logger will include them.

    Useful for ``session_id`` which is constant for the lifetime of a
    LangGraph session and would otherwise need to be passed to every
    ``get_logger`` call.

    Use sparingly — global mutable state is a footgun. The recommended
    pattern is:

    1. At session start, call ``bind_context(session_id=session_id)``.
    2. Inside each node, call ``log = get_logger(node_name).bind(
          node_name=node_name, target_url=str(target.url))`` for the
          per-node context.

    Parameters
    ----------
    **kwargs:
        Key-value pairs to add to structlog's global context. Pass
        ``None`` for a key to remove it from the global context.
    """
    try:
        import structlog
    except ImportError:
        # No-op fallback — see get_logger docstring.
        return

    # structlog.contextvars.merge_contextvars reads from a ContextVar
    # store. bind_contextvars / unbind_contextvars are the
    # process-global equivalents. We use these so the global context
    # survives across asyncio task boundaries within a single process.
    for key, value in kwargs.items():
        if value is None:
            structlog.contextvars.unbind_contextvars(key)
        else:
            structlog.contextvars.bind_contextvars(**{key: value})


def reset_context() -> None:
    """Clear all globally-bound context variables.

    Primarily useful in tests to isolate context between test cases.
    In production, the global context is typically only set once per
    session and reset is not needed.
    """
    try:
        import structlog
    except ImportError:
        return
    structlog.contextvars.clear_contextvars()


# ---------------------------------------------------------------------------
# Internal configuration helpers
# ---------------------------------------------------------------------------


def _resolve_level(level: str | int | None) -> int:
    """Convert a level name / number / None into a numeric level.

    Falls back to :attr:`settings.LOG_LEVEL` if ``level`` is None.
    """
    if level is None:
        level = settings.LOG_LEVEL
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        normalized = level.upper().strip()
        # logging.getLevelName returns the int for a name, or a string
        # for an unknown name. Guard against the latter.
        resolved = logging.getLevelName(normalized)
        if isinstance(resolved, int):
            return resolved
        raise ValueError(
            f"Unknown log level name: {level!r}. "
            f"Valid names: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    raise TypeError(
        f"level must be str, int, or None — got {type(level).__name__}"
    )


def _configure_stdlib_root(level: int) -> None:
    """Configure the stdlib root logger as the sink for structlog.

    - Sets the root level to ``level``.
    - Installs a single :class:`~logging.StreamHandler` writing to
      ``stderr`` (so stdout remains free for actual program output).
    - Replaces any pre-existing handlers to avoid duplicate output on
      re-configuration (e.g. when ``force=True`` is passed to
      :func:`setup_logging`).
    """
    root = logging.getLogger()

    # Remove existing handlers — we want exactly one. This is safe
    # because setup_logging is documented as the single point of
    # configuration.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    # The actual formatting is done by structlog's ConsoleRenderer;
    # the stdlib handler just passes the pre-formatted message through.
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)

    root.setLevel(level)

    # The framework's own logger follows the root level (no override).
    logging.getLogger(FRAMEWORK_ROOT_LOGGER_NAME).setLevel(level)


def _configure_structlog(level: int) -> None:
    """Configure structlog's processor chain.

    Architecture
    ------------
    We use the structlog-recommended pattern for integrating with
    stdlib logging: structlog's own processors build up an event dict
    but do NOT render it — instead the last processor
    (:func:`structlog.stdlib.ProcessorFormatter.wrap_for_formatter`)
    hands the dict to a :class:`ProcessorFormatter` installed on the
    stdlib handler, which applies the final renderer.

    This lets BOTH structlog-emitted log lines AND foreign log records
    (httpx, playwright, langchain, ...) flow through the same final
    renderer — so they get the same colorized format.

    Processor chain (structlog-emitted lines)
    -----------------------------------------
    1. ``merge_contextvars``      — pulls in any global context bound
       via :func:`bind_context`.
    2. ``add_log_level``          — adds ``level`` to every event dict.
    3. ``TimeStamper``            — adds ``timestamp`` in ISO 8601 UTC.
    4. ``add_logger_name``        — adds ``logger`` = the logger name.
    5. ``StackInfoRenderer``      — formats ``stack_info`` if present.
    6. ``format_exc_info``        — formats ``exc_info`` into
       ``exception`` string.
    7. ``wrap_for_formatter``     — hands the dict to the stdlib
       :class:`ProcessorFormatter` for final rendering (NOT a renderer
       here — that would cause double rendering).

    The :class:`ProcessorFormatter` is configured with
    ``foreign_pre_chain`` set to the same shared processors (minus the
    final ``wrap_for_formatter``) so foreign log records get the same
    timestamp / level / logger_name enrichment before final rendering.

    Renderer
    --------
    :class:`structlog.dev.ConsoleRenderer` is the final renderer for
    both structlog and foreign records. It produces colorized,
    column-aligned output suitable for local terminal use. If/when we
    add a server mode, a JSON renderer can be wired up here based on
    a setting.
    """
    import structlog

    # Shared processors — applied to BOTH structlog-emitted lines and
    # foreign log records (via foreign_pre_chain). Do NOT include a
    # renderer here, and do NOT include ``wrap_for_formatter`` — that
    # is structlog-only and goes in the structlog.configure() chain.
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Configure structlog. The final processor is
    # ``wrap_for_formatter`` — this DOES NOT render the dict, it hands
    # it to the stdlib ProcessorFormatter for rendering. Without this,
    # we would get double-rendered output (once by structlog's chain,
    # once by the ProcessorFormatter on the stdlib handler).
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Wire the shared processors into stdlib's formatter so foreign
    # log records (httpx, playwright, ...) get the same enrichment
    # before the final renderer runs.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processor=structlog.dev.ConsoleRenderer(colors=True),
    )

    # Apply the formatter to the handler we installed in
    # _configure_stdlib_root. setup_logging is documented as the
    # single point of configuration, so we can assume there is exactly
    # one handler at this point.
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(formatter)


def _silence_noisy_libraries() -> None:
    """Force noisy third-party loggers to WARNING or ERROR.

    This runs AFTER stdlib root configuration so the level overrides
    are not reset. The levels are hard-coded — operators who need
    finer control can call
    ``logging.getLogger("httpx").setLevel(logging.DEBUG)`` after
    :func:`setup_logging` returns.
    """
    for logger_name, min_level in _NOISY_LIBRARIES.items():
        logging.getLogger(logger_name).setLevel(min_level)


# ---------------------------------------------------------------------------
# Graceful-degradation fallback (used only when structlog is missing)
# ---------------------------------------------------------------------------


class _StdlibFallbackLogger:
    """Minimal stdlib-only logger that mimics structlog's BoundLogger
    API. Used ONLY when structlog is not installed so the framework
    can still produce *some* log output.

    This is NOT a feature-complete replacement — it does not support
    processor chains, contextvars, or colorized output. It exists
    solely to prevent a hard crash in environments where structlog
    is pending installation.

    The API surface mimics the subset of structlog.BoundLogger that
    the framework actually uses: ``.bind()``, ``.new()``,
    ``.info()``, ``.debug()``, ``.warning()``, ``.error()``,
    ``.critical()``.
    """

    __slots__ = ("_name", "_context")

    def __init__(self, name: str, **context: Any) -> None:
        self._name = name
        self._context: dict[str, Any] = dict(context)

    def bind(self, **kwargs: Any) -> "_StdlibFallbackLogger":
        """Return a new logger with additional bound context."""
        new_context = {**self._context, **kwargs}
        return _StdlibFallbackLogger(self._name, **new_context)

    def new(self, **kwargs: Any) -> "_StdlibFallbackLogger":
        """Return a new logger with fresh context (replaces existing)."""
        return _StdlibFallbackLogger(self._name, **kwargs)

    def _emit(self, level: int, event: str, **kw: Any) -> None:
        logger = logging.getLogger(self._name)
        # ``exc_info`` is a stdlib logging special-case: it must be
        # passed as a kwarg to ``logger.log()`` (not embedded in the
        # message) so the stdlib formatter can render the traceback.
        # We pop it from the context dict before rendering the rest.
        exc_info = kw.pop("exc_info", None)
        merged = {**self._context, **kw}
        # Render the context as ``key=value`` pairs for readability.
        if merged:
            ctx_str = " ".join(f"{k}={v!r}" for k, v in merged.items())
            message = f"{event}  {ctx_str}"
        else:
            message = event
        if exc_info is not None:
            logger.log(level, message, exc_info=exc_info)
        else:
            logger.log(level, message)

    def debug(self, event: str, **kw: Any) -> None:
        self._emit(logging.DEBUG, event, **kw)

    def info(self, event: str, **kw: Any) -> None:
        self._emit(logging.INFO, event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._emit(logging.WARNING, event, **kw)

    warn = warning  # structlog alias

    def error(self, event: str, **kw: Any) -> None:
        self._emit(logging.ERROR, event, **kw)

    def critical(self, event: str, **kw: Any) -> None:
        self._emit(logging.CRITICAL, event, **kw)

    fatal = critical  # stdlib alias

    def exception(self, event: str, **kw: Any) -> None:
        """Log at ERROR level with exception info attached."""
        kw.setdefault("exc_info", True)
        self._emit(logging.ERROR, event, **kw)


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "setup_logging",
    "get_logger",
    "bind_context",
    "reset_context",
    "FRAMEWORK_ROOT_LOGGER_NAME",
    "STANDARD_CONTEXT_KEYS",
]
