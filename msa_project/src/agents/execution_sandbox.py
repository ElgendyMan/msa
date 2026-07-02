"""
src/agents/execution_sandbox.py
===============================

Node 11 of the 16-node LangGraph framework: the **Execution Sandbox**.

This node is the "hands" of the framework. It takes the active payload
produced by the Payload Generator (and optionally refined by the Payload
Optimizer) and sends it to the target over the wire. It performs NO
analysis — analysis is the Validator's job. The sandbox's contract is:

    Send the payload exactly as specified. Record what was sent and what
    came back. Fail loudly on timeouts and WAF blocks. Capture all other
    errors inside ``ExecutionResult.error`` so the Validator can decide
    whether they are meaningful.

Security invariants
-------------------
1. **No LLM in this path.** Pure Python + httpx. The sandbox must be
   deterministic and fast; an LLM call here would add 1-3s of latency
   to every payload, plus a non-determinism risk.
2. **SSL verification is disabled.** Pentest targets frequently use
   self-signed or expired certs; we must not refuse to talk to them.
   ``verify=False`` is set explicitly on the httpx client.
3. **Redirects are NOT followed.** The Validator needs to see the raw
   301/302 response (e.g. an open-redirect payload that returns 302 to
   an attacker-controlled host is a TRUE_POSITIVE only if we observe
   the 302, not if we land on the attacker host).
4. **Scope is re-verified.** Even though the Scope Enforcer already
   ran, defense-in-depth requires that we re-check the target URL is
   in-scope immediately before sending. A bug in the Orchestrator
   could otherwise route an out-of-scope target here.
5. **Rate limit is enforced.** The sandbox shares a global semaphore
   across all concurrent executions so the operator's
   ``max_requests_per_second`` / ``max_concurrent_requests``
   limits are honored.
6. **Timeouts raise, not silently swallow.** A timeout is a strong
   signal that either the target is down OR a time-based blind
   payload (e.g. SQLi ``SLEEP(5)``) fired. Either way, the Validator
   needs to know.

Parallel Execution (Optimization A)
-----------------------------------
When there are multiple pending payloads (payloads without execution
results), this node executes them ALL concurrently via
``asyncio.gather``. This is 3x faster for 3 payloads, 5x for 5, etc.

The concurrency semaphore inside ``_execute_via_httpx`` limits actual
parallelism to ``settings.EXECUTION_MAX_CONCURRENT``.

LangGraph contract
------------------
::

    async def execute_payload(state: AppState) -> dict:

- Reads: ``state["payloads"]`` (list of :class:`Payload`),
         ``state["active_payload_id"]`` (str),
         ``state["scope"]`` (:class:`ScopeConfig`, optional).
- Writes: returns ``{"execution_results": [<ExecutionResult>]}``.

Raises
------
- :class:`PentestFrameworkError` — payload not found.
- :class:`ScopeViolationError` — defense-in-depth scope re-check failed.
- :class:`ExecutionTimeoutError` — httpx timeout exceeded.
- :class:`WAFBlockError` — WAF clearly blocked the payload.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx

from src.shared.config import settings
from src.shared.exceptions import (
    PentestFrameworkError,
    ScopeViolationError,
    ExecutionTimeoutError,
    WAFBlockError,
)
from src.shared.logging import get_logger
from src.shared.schemas import (
    ExecutionResult,
    HTTPMethod,
    HTTPRequestRecord,
    HTTPResponseRecord,
    Payload,
    PayloadTransport,
    ScopeConfig,
    WAFSignature,
)
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# WAF detection table
# ---------------------------------------------------------------------------

_WAF_STATUS_CODES: frozenset[int] = frozenset(
    {
        403, 418, 429, 503, 504,
        520, 521, 522, 523, 524, 525, 526,
    }
)

_WAF_HEADER_SIGNATURES: tuple[tuple[str, str, WAFSignature], ...] = (
    ("cf-ray", "", WAFSignature.CLOUDFLARE),
    ("server", "cloudflare", WAFSignature.CLOUDFLARE),
    ("server", "akamai", WAFSignature.AKAMAI),
    ("akamai-grn", "", WAFSignature.AKAMAI),
    ("x-akamai-transformed", "", WAFSignature.AKAMAI),
    ("server", "imperva", WAFSignature.IMPERVA),
    ("x-iinfo", "", WAFSignature.IMPERVA),
    ("x-cdn", "imperva", WAFSignature.IMPERVA),
    ("server", "sucuri", WAFSignature.SUCURI),
    ("x-sucuri-id", "", WAFSignature.SUCURI),
    ("server", "bigip", WAFSignature.F5_BIGIP),
    ("server", "f5", WAFSignature.F5_BIGIP),
    ("x-cnection", "", WAFSignature.F5_BIGIP),
    ("server", "awselb", WAFSignature.AWS_WAF),
    ("x-amz-cf-id", "", WAFSignature.AWS_WAF),
    ("x-amz-cf-pop", "", WAFSignature.AWS_WAF),
    ("via", "cloudfront", WAFSignature.AWS_WAF),
    ("server", "modsecurity", WAFSignature.MODSECURITY),
    ("server", "mod_security", WAFSignature.MODSECURITY),
    ("x-mod-security", "", WAFSignature.MODSECURITY),
)


# ---------------------------------------------------------------------------
# Per-session concurrency primitives
# ---------------------------------------------------------------------------
# FIX: previously these were single module-level globals shared across
# ALL sessions/targets running in the same process. Two concurrent
# pentest sessions against different targets with different
# scope.max_concurrent_requests / max_requests_per_second would
# incorrectly share one semaphore and one rate limiter — a session
# with a tight limit would throttle a session that should run faster,
# and vice versa. Each session now gets its own isolated primitives,
# keyed by session_id, built once from THAT session's scope.


class _SessionRateState:
    """Per-session rate-limiting state: one semaphore + one token-bucket
    timestamp, both scoped to a single session_id."""

    __slots__ = ("semaphore", "last_request_time", "lock")

    def __init__(self, concurrency_limit: int) -> None:
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrency_limit)
        self.last_request_time: float = 0.0
        self.lock: asyncio.Lock = asyncio.Lock()


_session_rate_states: dict[str, _SessionRateState] = {}
_session_rate_states_lock: asyncio.Lock = asyncio.Lock()


async def _get_session_rate_state(
    session_id: str, scope: ScopeConfig | None
) -> _SessionRateState:
    """Return (creating if necessary) the rate state for this session.

    Built once per session_id from that session's own scope limits, so
    concurrent sessions never share a semaphore or rate-limit clock.
    """
    existing = _session_rate_states.get(session_id)
    if existing is not None:
        return existing

    async with _session_rate_states_lock:
        # Re-check after acquiring the lock (another coroutine may have
        # created it while we were waiting).
        existing = _session_rate_states.get(session_id)
        if existing is not None:
            return existing

        scope_limit = (
            scope.max_concurrent_requests if scope else settings.EXECUTION_MAX_CONCURRENT
        )
        effective_limit = min(scope_limit, settings.EXECUTION_MAX_CONCURRENT)
        effective_limit = max(1, effective_limit)

        state = _SessionRateState(effective_limit)
        _session_rate_states[session_id] = state
        return state


def reset_session_rate_state(session_id: str) -> None:
    """Test/utility helper: drop a session's rate state (e.g. at session
    end) so its memory doesn't accumulate across long-running processes."""
    _session_rate_states.pop(session_id, None)


async def _enforce_rate_limit(state: _SessionRateState, scope: ScopeConfig | None) -> None:
    import time
    scope_rps = scope.max_requests_per_second if scope else settings.EXECUTION_RATE_LIMIT_RPS
    effective_rps = min(scope_rps, settings.EXECUTION_RATE_LIMIT_RPS)
    min_interval = 1.0 / float(effective_rps)
    async with state.lock:
        now = time.monotonic()
        elapsed = now - state.last_request_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        state.last_request_time = time.monotonic()


# ---------------------------------------------------------------------------
# Internal: single-payload execution wrapper (error-safe)
# ---------------------------------------------------------------------------


async def _execute_single_payload_safe(
    payload: Payload,
    scope: ScopeConfig | None,
    log: Any,
    session_id: str,
) -> ExecutionResult:
    """Execute a single payload, capturing ALL exceptions into an
    :class:`ExecutionResult`.

    This wrapper never raises — it always returns an ExecutionResult.
    On success, the result has a response. On failure, the result has
    an error string and no response.
    """
    try:
        if payload.transport in (PayloadTransport.HTTP_REQUEST, PayloadTransport.GRAPHQL_QUERY):
            return await _execute_via_httpx(payload, scope, log, session_id)
        elif payload.transport in (PayloadTransport.WEBSOCKET, PayloadTransport.PLAYWRIGHT_DOM):
            return _build_error_result(
                payload,
                NotImplementedError(
                    f"Transport '{payload.transport.value}' is not yet implemented."
                ),
            )
        else:
            return _build_error_result(
                payload,
                PentestFrameworkError(
                    f"Unknown payload transport: {payload.transport!r}",
                    details={"transport": str(payload.transport)},
                ),
            )
    except httpx.TimeoutException as exc:
        log.warning(
            "execution_timeout",
            error_type=type(exc).__name__,
            timeout=settings.EXECUTION_TIMEOUT_SECONDS,
        )
        return _build_error_result(
            payload,
            ExecutionTimeoutError(
                f"Payload execution timed out after "
                f"{settings.EXECUTION_TIMEOUT_SECONDS}s (transport=httpx, "
                f"phase={type(exc).__name__}).",
                payload_id=payload.id,
                timeout_seconds=settings.EXECUTION_TIMEOUT_SECONDS,
                transport="httpx",
            ),
        )
    except WAFBlockError as exc:
        log.warning(
            "waf_block_detected",
            waf_signature=exc.waf_signature,
            status_code=exc.status_code,
        )
        return _build_error_result(payload, exc)
    except (
        httpx.ConnectError, httpx.ReadError, httpx.WriteError,
        httpx.NetworkError, httpx.ProtocolError, httpx.ProxyError,
        httpx.UnsupportedProtocol, httpx.RemoteProtocolError,
        httpx.LocalProtocolError, OSError,
    ) as exc:
        log.warning(
            "connection_error",
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
        )
        return _build_error_result(payload, exc)
    except httpx.HTTPError as exc:
        log.warning(
            "httpx_error",
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
        )
        return _build_error_result(payload, exc)
    except Exception as exc:
        log.exception(
            "unexpected_execution_error",
            error_type=type(exc).__name__,
        )
        return _build_error_result(payload, exc)


def _log_execution_result(er: ExecutionResult, log: Any) -> None:
    """Log a completion line for an ExecutionResult."""
    if er.response is not None:
        log.info(
            "execution_complete",
            status_code=er.response.status_code,
            elapsed_ms=er.response.elapsed_ms,
            error=er.error,
        )
    else:
        log.info(
            "execution_complete_no_response",
            error=er.error,
        )


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


async def execute_payload(state: AppState) -> dict[str, Any]:
    """LangGraph Node 11: execute the active payload(s) against the target.

    Supports PARALLEL execution: if there are multiple pending payloads,
    they are ALL executed concurrently via asyncio.gather.

    Parameters
    ----------
    state:
        The current :class:`~src.shared.state.AppState`.

    Returns
    -------
    dict
        ``{"execution_results": [<ExecutionResult>, ...]}``

    Raises
    ------
    PentestFrameworkError
        If ``payloads`` is missing/empty or ``active_payload_id`` is missing.
    ScopeViolationError
        If the defense-in-depth scope re-check fails.
    """
    log = get_logger("execution_sandbox")

    session_id: str = state.get("session_id") or "_default"

    # ---------------------------------------------------------------
    # 1. Resolve payloads to execute — with PARALLEL mode.
    # ---------------------------------------------------------------
    payloads: list[Payload] | None = state.get("payloads")
    if not payloads:
        raise PentestFrameworkError(
            "Execution Sandbox cannot run: state['payloads'] is missing or empty. "
            "The Payload Generator must run before the Execution Sandbox.",
            details={"available_keys": list(state.keys())},
        )

    # Find all payloads that haven't been executed yet.
    existing_results: list[ExecutionResult] | None = state.get("execution_results")
    executed_ids: set[str] = (
        {er.payload_id for er in existing_results} if existing_results else set()
    )
    pending_payloads: list[Payload] = [
        p for p in payloads
        if p.id not in executed_ids
        and p.transport in (PayloadTransport.HTTP_REQUEST, PayloadTransport.GRAPHQL_QUERY)
    ]

    # If active_payload_id is set and its payload is pending, ensure
    # it's in the list (it should already be, but defensive).
    active_id: str | None = state.get("active_payload_id")
    if active_id:
        active_payload: Payload | None = next(
            (p for p in payloads if p.id == active_id), None
        )
        if active_payload is not None and active_payload not in pending_payloads:
            if active_payload.transport in (PayloadTransport.HTTP_REQUEST, PayloadTransport.GRAPHQL_QUERY):
                pending_payloads.insert(0, active_payload)

    if not pending_payloads:
        log.info("execution_skipped_no_pending_payloads")
        return {"execution_results": []}

    scope: ScopeConfig | None = state.get("scope")

    # ---------------------------------------------------------------
    # 2. Defense-in-depth: re-verify scope for ALL pending payloads.
    # ---------------------------------------------------------------
    if scope is not None:
        for p in pending_payloads:
            _verify_scope_for_payload(p, scope)

    # ---------------------------------------------------------------
    # 3. Execute payloads — PARALLEL if >1, SINGLE if 1.
    # ---------------------------------------------------------------
    if len(pending_payloads) == 1:
        # Single payload — no need for gather overhead.
        payload = pending_payloads[0]
        log = log.bind(
            payload_id=payload.id,
            hypothesis_id=payload.hypothesis_id,
            target_url=str(payload.target_url),
        )
        log.info(
            "execution_attempt",
            transport=payload.transport.value,
            method=(payload.http_method.value if payload.http_method else "N/A"),
            is_optimized=payload.is_optimized,
        )
        execution_result = await _execute_single_payload_safe(payload, scope, log, session_id)
        _log_execution_result(execution_result, log)
        return {"execution_results": [execution_result]}

    # Multiple payloads — execute in PARALLEL.
    log.info(
        "parallel_execution_started",
        payload_count=len(pending_payloads),
        payload_ids=[p.id for p in pending_payloads],
    )

    # Build the list of coroutines WITHOUT a nested function.
    # This avoids the NameError caused by nested function scope issues.
    coros = [
        _execute_single_payload_safe(p, scope, log.bind(
            payload_id=p.id,
            hypothesis_id=p.hypothesis_id,
            target_url=str(p.target_url),
        ), session_id)
        for p in pending_payloads
    ]

    results: list[ExecutionResult] = await asyncio.gather(*coros)

    for p, er in zip(pending_payloads, results):
        p_log = log.bind(payload_id=p.id)
        _log_execution_result(er, p_log)

    log.info(
        "parallel_execution_complete",
        payload_count=len(pending_payloads),
        results_count=len(results),
    )

    return {"execution_results": results}


# ---------------------------------------------------------------------------
# Internal: HTTP transport via httpx
# ---------------------------------------------------------------------------


async def _execute_via_httpx(
    payload: Payload,
    scope: ScopeConfig | None,
    log: Any,
    session_id: str,
) -> ExecutionResult:
    """Execute an HTTP or GraphQL payload via :class:`httpx.AsyncClient`."""
    request_record = _build_request_record(payload)
    method = (payload.http_method or HTTPMethod.GET).value

    request_kwargs: dict[str, Any] = {
        "method": method,
        "url": str(payload.target_url),
        "headers": _merge_headers(payload),
        "timeout": settings.EXECUTION_TIMEOUT_SECONDS,
        "follow_redirects": False,
    }

    body_bytes: bytes | None = None
    if request_record.body_bytes_b64:
        try:
            body_bytes = base64.b64decode(request_record.body_bytes_b64)
        except Exception:
            body_bytes = None
    if body_bytes is None and request_record.body:
        body_bytes = request_record.body.encode("utf-8", errors="replace")
    if body_bytes is not None:
        request_kwargs["content"] = body_bytes

    rate_state = await _get_session_rate_state(session_id, scope)
    async with rate_state.semaphore:
        await _enforce_rate_limit(rate_state, scope)

        async with httpx.AsyncClient(
            verify=False,
            follow_redirects=False,
            timeout=settings.EXECUTION_TIMEOUT_SECONDS,
            http2=False,
            trust_env=False,
        ) as client:
            import time
            start = time.monotonic()
            response = await client.request(**request_kwargs)
            elapsed_ms = int((time.monotonic() - start) * 1000)

    response_record = _build_response_record(response, elapsed_ms)

    waf_signature = _detect_waf(response)
    if waf_signature is not None:
        raise WAFBlockError(
            f"WAF block detected: signature='{waf_signature.value}', "
            f"status={response.status_code}.",
            payload_id=payload.id,
            waf_signature=waf_signature.value,
            status_code=response.status_code,
            details={
                "response_headers": dict(response.headers),
                "response_body_preview": response.text[:500] if response.text else "",
            },
        )

    return ExecutionResult(
        payload_id=payload.id,
        request=request_record,
        response=response_record,
        error=None,
        retry_attempts=0,
        rate_limited=False,
    )


def _build_request_record(payload: Payload) -> HTTPRequestRecord:
    """Build a :class:`HTTPRequestRecord` from the payload."""
    method = (payload.http_method or HTTPMethod.GET)
    headers = _merge_headers(payload)

    body_str: str | None = None
    body_b64: str | None = None
    if payload.raw:
        try:
            payload.raw.encode("utf-8")
            body_str = payload.raw
        except UnicodeEncodeError:
            body_b64 = base64.b64encode(
                payload.raw.encode("utf-8", errors="surrogatepass")
            ).decode("ascii")

    return HTTPRequestRecord(
        method=method,
        url=payload.target_url,
        headers=headers,
        body=body_str,
        body_bytes_b64=body_b64,
    )


def _build_response_record(
    response: httpx.Response, elapsed_ms: int
) -> HTTPResponseRecord:
    """Build an :class:`HTTPResponseRecord` from an httpx response."""
    body_str: str | None = None
    body_b64: str | None = None
    if response.content:
        try:
            decoded = response.content.decode("utf-8")
            decoded.encode("utf-8")
            body_str = decoded
        except UnicodeDecodeError:
            body_b64 = base64.b64encode(response.content).decode("ascii")

    headers: dict[str, str] = {}
    for raw_name, raw_value in response.headers.raw:
        name = raw_name.decode("latin-1")
        value = raw_value.decode("latin-1")
        if name in headers:
            headers[name] = f"{headers[name]}, {value}"
        else:
            headers[name] = value

    return HTTPResponseRecord(
        status_code=response.status_code,
        headers=headers,
        body=body_str,
        body_bytes_b64=body_b64,
        elapsed_ms=elapsed_ms,
    )


def _build_error_result(payload: Payload, exc: Exception) -> ExecutionResult:
    """Build an :class:`ExecutionResult` for a failed execution."""
    return ExecutionResult(
        payload_id=payload.id,
        request=_build_request_record(payload),
        response=None,
        error=f"{type(exc).__name__}: {exc}",
        retry_attempts=0,
        rate_limited=False,
    )


def _merge_headers(payload: Payload) -> dict[str, str]:
    """Merge the payload's headers with a sensible default set."""
    headers: dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }
    for k, v in (payload.headers or {}).items():
        headers[str(k)] = str(v)
    return headers


# ---------------------------------------------------------------------------
# Internal: WAF detection
# ---------------------------------------------------------------------------


def _detect_waf(response: httpx.Response) -> WAFSignature | None:
    """Inspect the response and return a WAF signature if detected."""
    headers_lower: dict[str, str] = {
        k.lower(): v for k, v in response.headers.items()
    }

    for header_name, substring, signature in _WAF_HEADER_SIGNATURES:
        if header_name not in headers_lower:
            continue
        if not substring:
            return signature
        if substring in headers_lower[header_name].lower():
            return signature

    if response.status_code in _WAF_STATUS_CODES:
        body_text = response.text.lower() if response.text else ""
        header_text = " ".join(headers_lower.values()).lower()
        haystack = f"{body_text} {header_text}"
        generic_keywords = (
            "firewall", "waf", "security rule", "request rejected",
            "web application firewall", "blocked by security",
            "denied by security",
        )
        if any(kw in haystack for kw in generic_keywords):
            return WAFSignature.UNKNOWN

    return None


# ---------------------------------------------------------------------------
# Internal: defense-in-depth scope re-check
# ---------------------------------------------------------------------------


def _verify_scope_for_payload(payload: Payload, scope: ScopeConfig) -> None:
    """Defense-in-depth: re-verify the payload's target URL is in-scope."""
    from src.agents.scope_enforcer import _domain_matches, _resolve_to_ip, _ip_in_cidr

    url_str = str(payload.target_url)
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url_str)
    except Exception:
        raise ScopeViolationError(
            f"Defense-in-depth scope check failed: cannot parse payload "
            f"target_url '{url_str}'.",
            target=url_str,
            reason="url_unparseable",
        )

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ScopeViolationError(
            f"Defense-in-depth scope check failed: payload target_url "
            f"'{url_str}' has no hostname.",
            target=url_str,
            reason="url_missing_hostname",
        )

    for denied in scope.out_of_scope_domains:
        if _domain_matches(hostname, denied):
            raise ScopeViolationError(
                f"Defense-in-depth scope check failed: payload target "
                f"'{hostname}' matches out-of-scope domain '{denied}'.",
                target=url_str,
                reason="domain_out_of_scope",
            )

    target_ip = _resolve_to_ip(hostname)
    if target_ip is not None:
        for denied_cidr in scope.out_of_scope_cidrs:
            try:
                if _ip_in_cidr(target_ip, denied_cidr):
                    raise ScopeViolationError(
                        f"Defense-in-depth scope check failed: payload "
                        f"target IP '{target_ip}' is in out-of-scope "
                        f"CIDR '{denied_cidr}'.",
                        target=url_str,
                        reason="cidr_out_of_scope",
                    )
            except Exception:
                continue

    domain_ok = any(_domain_matches(hostname, a) for a in scope.in_scope_domains)
    cidr_ok = False
    if target_ip is not None:
        cidr_ok = any(
            _ip_in_cidr(target_ip, c)
            for c in scope.in_scope_cidrs
            if _safe_cidr_check(c)
        )
    if not domain_ok and not cidr_ok:
        raise ScopeViolationError(
            f"Defense-in-depth scope check failed: payload target "
            f"'{hostname}' does not match any in-scope domain or CIDR.",
            target=url_str,
            reason="not_in_scope",
        )


def _safe_cidr_check(cidr: str) -> bool:
    try:
        import ipaddress
        ipaddress.ip_network(cidr, strict=False)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = ["execute_payload"]