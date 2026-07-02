"""
src/agents/scope_enforcer.py
============================

Node 1 of the 16-node LangGraph framework: the **Scope Enforcer**.

This is the framework's most safety-critical node. It is the ONLY gate
between the operator's intent (declared in ``scope.json`` →
``AppState["scope"]``) and any subsequent node that issues network
traffic. Every other node in the graph MUST be able to assume that
``AppState["scope_verified"] == True`` implies the target has been
positively confirmed as in-scope.

Security invariants
-------------------
1. **No LLM in the matching path.** All scope matching is performed by
   deterministic pure-Python code (``ipaddress``, ``urllib.parse``).
   This is non-negotiable: an LLM hallucinating "yes, that host is
   in-scope" is the worst possible failure mode for a pentest tool.
2. **Deny overrides allow.** ``out_of_scope_*`` is checked BEFORE
   ``in_scope_*``. A target that matches both lists is rejected.
3. **Fail closed.** ``Target.is_in_scope`` defaults to ``False`` in the
   schema; this node must explicitly prove every check passes before
   returning ``scope_verified=True``.
4. **Legal attestation is mandatory.** If
   ``scope.legal_acknowledged`` is False, the node refuses regardless
   of how perfectly the target matches the in-scope lists. This
   protects the operator from running the framework against a target
   they do not have written authorization to test.
5. **Every failure raises** ``ScopeViolationError`` with structured
   ``details`` so the Orchestrator can route to the ERROR phase and
   the operator can see exactly which check failed and why.

LangGraph contract
------------------
::

    def enforce_scope(state: AppState) -> dict:

- Reads: ``state["target"]``, ``state["scope"]``.
- Writes: returns ``{"scope_verified": True}`` on success. On failure,
  raises ``ScopeViolationError`` — LangGraph catches the exception via
  the graph's error handler and routes to the ERROR phase.
- Does NOT mutate any other state channel. This is the
  Single-Responsibility Principle: scope enforcement is the ONLY thing
  this node does.

Matching semantics
------------------
- **Domain match**: case-insensitive. ``example.com`` matches
  ``example.com``, ``EXAMPLE.COM``, and ``api.example.com`` (subdomain
  match), but NOT ``notexample.com`` (suffix-without-dot is rejected
  to prevent the classic ``evil.com`` ↔ ``notevil.com`` confusion).
- **CIDR match**: ``ipaddress`` module. Both IPv4 and IPv6 are
  supported. A bare IP literal in the target hostname is matched
  against both ``in_scope_cidrs`` and ``out_of_scope_cidrs``.
- **Port match**: exact integer equality against
  ``scope.allowed_ports``. The framework is web-only; non-web ports
  (22, 21, 445, 3389, ...) are never permitted here. The separate
  Web Filter node handles dropping non-web services from recon data.
"""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse

from src.shared.exceptions import ScopeViolationError
from src.shared.schemas import ScopeConfig, Target
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


def enforce_scope(state: AppState) -> dict[str, Any]:
    """LangGraph Node 1: enforce the operator's declared scope against
    the current target.

    Parameters
    ----------
    state:
        The current :class:`~src.shared.state.AppState`. Must contain
        ``target`` (a :class:`~src.shared.schemas.Target`) and ``scope``
        (a :class:`~src.shared.schemas.ScopeConfig`).

    Returns
    -------
    dict
        ``{"scope_verified": True}`` — LangGraph merges this partial
        dict into the running state. No other channels are touched.

    Raises
    ------
    ScopeViolationError
        If any of the following fail (in order):

        1. ``scope.legal_acknowledged`` is not True.
        2. The target URL cannot be parsed.
        3. The target port is not in ``scope.allowed_ports``.
        4. The target hostname matches ``scope.out_of_scope_domains``.
        5. The target IP matches ``scope.out_of_scope_cidrs``.
        6. The target hostname does NOT match ``scope.in_scope_domains``.
        7. The target IP does NOT match ``scope.in_scope_cidrs``.

        A target that matches BOTH in-scope and out-of-scope lists is
        rejected (deny-overrides-allow).
    """
    target: Target = _require(state, "target")
    scope: ScopeConfig = _require(state, "scope")

    # ---------------------------------------------------------------
    # Check 1: legal attestation. Refuse without it, no exceptions.
    # ---------------------------------------------------------------
    if not scope.legal_acknowledged:
        raise ScopeViolationError(
            "Legal attestation is required: scope.legal_acknowledged is "
            "False. The framework refuses to operate against any target "
            "until the operator attests that written authorization for "
            "testing exists. Set 'legal_acknowledged: true' in scope.json.",
            target=str(target.url),
            reason="legal_attestation_missing",
        )

    # ---------------------------------------------------------------
    # Check 2: parse the target URL. Fail loudly on malformed URLs.
    # ---------------------------------------------------------------
    hostname, port = _extract_host_and_port(target)

    # ---------------------------------------------------------------
    # Check 3: port must be in the web-allowlist.
    # ---------------------------------------------------------------
    if port not in scope.allowed_ports:
        raise ScopeViolationError(
            f"Target port {port} is not in the allowed_ports list "
            f"({sorted(scope.allowed_ports)}). The framework is "
            f"web-only; non-web ports are rejected at the scope gate.",
            target=str(target.url),
            reason="port_not_allowed",
            details={"port": port, "allowed_ports": sorted(scope.allowed_ports)},
        )

    # ---------------------------------------------------------------
    # Check 4: deny-overrides-allow — out-of-scope DOMAINS first.
    # ---------------------------------------------------------------
    for denied in scope.out_of_scope_domains:
        if _domain_matches(hostname, denied):
            raise ScopeViolationError(
                f"Target hostname '{hostname}' matches out-of-scope "
                f"domain '{denied}'. Out-of-scope entries take "
                f"precedence over in-scope (deny-overrides-allow).",
                target=str(target.url),
                reason="domain_out_of_scope",
                details={"hostname": hostname, "matched_denied_domain": denied},
            )

    # ---------------------------------------------------------------
    # Pre-validate ALL CIDR strings (in_scope AND out_of_scope) up
    # front. We do this BEFORE checking any IP membership so a
    # malformed CIDR in scope.json fails loudly regardless of whether
    # the current target resolves to an IP. Silent skip of a malformed
    # rule would be a security hole: an operator's typo in
    # ``out_of_scope_cidrs`` could silently disable an exclusion.
    # ---------------------------------------------------------------
    for cidr in list(scope.in_scope_cidrs) + list(scope.out_of_scope_cidrs):
        _validate_cidr_string(cidr)  # raises ScopeViolationError if malformed

    # ---------------------------------------------------------------
    # Check 5: deny-overrides-allow — out-of-scope CIDRs.
    # ---------------------------------------------------------------
    target_ip = _resolve_to_ip(hostname)
    if target_ip is not None:
        for denied_cidr in scope.out_of_scope_cidrs:
            if _ip_in_cidr(target_ip, denied_cidr):
                raise ScopeViolationError(
                    f"Target IP '{target_ip}' (resolved from '{hostname}') "
                    f"falls inside out-of-scope CIDR '{denied_cidr}'. "
                    f"Out-of-scope entries take precedence over in-scope "
                    f"(deny-overrides-allow).",
                    target=str(target.url),
                    reason="cidr_out_of_scope",
                    details={
                        "hostname": hostname,
                        "resolved_ip": str(target_ip),
                        "matched_denied_cidr": denied_cidr,
                    },
                )

    # ---------------------------------------------------------------
    # Check 6: must match at least one in-scope DOMAIN...
    # ---------------------------------------------------------------
    domain_in_scope = any(
        _domain_matches(hostname, allowed) for allowed in scope.in_scope_domains
    )

    # ---------------------------------------------------------------
    # Check 7: ...OR at least one in-scope CIDR.
    # ---------------------------------------------------------------
    cidr_in_scope = False
    if target_ip is not None:
        cidr_in_scope = any(
            _ip_in_cidr(target_ip, allowed_cidr)
            for allowed_cidr in scope.in_scope_cidrs
        )

    if not domain_in_scope and not cidr_in_scope:
        raise ScopeViolationError(
            f"Target '{hostname}' (ip={target_ip}) does not match any "
            f"in_scope_domains ({scope.in_scope_domains}) nor any "
            f"in_scope_cidrs ({scope.in_scope_cidrs}). A target must be "
            f"positively confirmed as in-scope; absence from the "
            f"out-of-scope list is not sufficient.",
            target=str(target.url),
            reason="not_in_scope",
            details={
                "hostname": hostname,
                "resolved_ip": str(target_ip) if target_ip else None,
                "in_scope_domains": list(scope.in_scope_domains),
                "in_scope_cidrs": list(scope.in_scope_cidrs),
            },
        )

    # ---------------------------------------------------------------
    # All checks passed. Return the partial state update.
    # LangGraph merges this into the running state via the default
    # (overwrite) reducer for the ``scope_verified`` channel.
    # ---------------------------------------------------------------
    return {"scope_verified": True}


# ---------------------------------------------------------------------------
# Internal helpers — pure Python, deterministic, no I/O except DNS
# ---------------------------------------------------------------------------


def _require(state: AppState, key: str) -> Any:
    """Extract a required key from ``state`` and raise a clear error if
    it is missing. LangGraph nodes receive the *full* accumulated state
    by contract, but defensive programming here prevents a confusing
    KeyError deep inside the matching logic if the Orchestrator ever
    routes here with malformed state."""
    if key not in state or state[key] is None:
        raise ScopeViolationError(
            f"Scope Enforcer cannot run: required state key '{key}' is "
            f"missing or None. This indicates an Orchestrator routing "
            f"bug — scope enforcement should only be invoked after the "
            f"target and scope have been loaded onto the state.",
            target="(unknown)",
            reason="state_incomplete",
            details={"missing_key": key, "available_keys": list(state.keys())},
        )
    return state[key]


def _extract_host_and_port(target: Target) -> tuple[str, int]:
    """Parse the target URL into ``(hostname, port)``.

    The hostname is lowercased because DNS is case-insensitive and we
    want ``EXAMPLE.com`` and ``example.com`` to compare equal.

    Port resolution order:
    1. Explicit port in the URL (``https://host:8443/``).
    2. Scheme default (``http`` → 80, ``https`` → 443, ``ws`` → 80,
       ``wss`` → 443).
    3. If neither yields a port, the URL is malformed; raise
       ``ScopeViolationError``.
    """
    url_str: str = str(target.url)

    try:
        parsed = urlparse(url_str)
    except Exception as exc:  # noqa: BLE001
        raise ScopeViolationError(
            f"Failed to parse target URL '{url_str}': {exc}",
            target=url_str,
            reason="url_unparseable",
            details={"url": url_str, "cause": str(exc)},
        ) from exc

    raw_hostname = parsed.hostname
    if not raw_hostname:
        raise ScopeViolationError(
            f"Target URL '{url_str}' has no hostname component. The "
            f"framework cannot enforce scope against a URL without a "
            f"concrete target host.",
            target=url_str,
            reason="url_missing_hostname",
            details={"url": url_str},
        )

    hostname = raw_hostname.lower()

    # Explicit port in URL wins.
    if parsed.port is not None:
        return hostname, parsed.port

    # Fall back to scheme default.
    scheme_defaults = {
        "http": 80,
        "https": 443,
        "ws": 80,
        "wss": 443,
    }
    scheme = (parsed.scheme or "").lower()
    if scheme in scheme_defaults:
        return hostname, scheme_defaults[scheme]

    raise ScopeViolationError(
        f"Target URL '{url_str}' has no explicit port and uses unknown "
        f"scheme '{scheme}'. Cannot determine the target port for "
        f"scope enforcement. Use http/https/ws/wss or specify a port "
        f"explicitly.",
        target=url_str,
        reason="port_indeterminable",
        details={"url": url_str, "scheme": scheme},
    )


def _domain_matches(hostname: str, pattern: str) -> bool:
    """Return True iff ``hostname`` matches the domain ``pattern``.

    Matching rules (deterministic, case-insensitive):
    - Exact match: ``hostname == pattern`` (both lowercased).
    - Subdomain match: ``hostname`` ends with ``.`` + ``pattern``.
      The leading dot is REQUIRED to prevent the
      ``evil.com`` ↔ ``notevil.com`` confusion. ``notevil.com`` must
      NOT match the pattern ``evil.com``.

    Examples
    --------
    >>> _domain_matches("example.com", "example.com")
    True
    >>> _domain_matches("api.example.com", "example.com")
    True
    >>> _domain_matches("notexample.com", "example.com")
    False
    >>> _domain_matches("EXAMPLE.COM", "example.com")
    True
    """
    hostname = hostname.lower().rstrip(".")
    pattern = pattern.lower().rstrip(".")
    if not hostname or not pattern:
        return False
    if hostname == pattern:
        return True
    return hostname.endswith("." + pattern)


def _resolve_to_ip(hostname: str) -> ipaddress._BaseAddress | None:
    """Return an IP address object for ``hostname``, or None if it
    cannot be resolved to a concrete IP without network I/O.

    - If ``hostname`` is already an IP literal (v4 or v6), return it
      immediately as an ``ipaddress`` object. No DNS, no I/O.
    - Otherwise (it's a DNS name), return None. We deliberately do NOT
      do a forward DNS lookup here because:
        (a) it would introduce network I/O into a "deterministic"
            code path;
        (b) DNS results can change between the scope check and the
            actual request, so the check would be unreliable anyway;
        (c) the operator is expected to express IP-based scope via
            ``in_scope_cidrs`` AND domain-based scope via
            ``in_scope_domains`` for the same target if they want
            both checks to fire.

    Downstream consumers (Execution Sandbox) SHOULD re-resolve and
    re-check scope immediately before sending the request, using the
    same ``_ip_in_cidr`` helper.
    """
    # Strip IPv6 brackets that urlparse may leave in the hostname field
    # for URLs like ``http://[::1]:8080/``.
    cleaned = hostname.strip("[]")

    # Try IPv4 first (faster path), then IPv6.
    for parser in (ipaddress.ip_address,):
        try:
            return parser(cleaned)
        except ValueError:
            continue
    return None


def _ip_in_cidr(ip: ipaddress._BaseAddress, cidr: str) -> bool:
    """Return True iff ``ip`` falls inside the CIDR ``cidr``.

    Assumes ``cidr`` has already been validated by
    :func:`_validate_cidr_string`. If callers bypass that pre-validation,
    a ``ValueError`` from ``ipaddress.ip_network`` will propagate — that
    is intentional and indicates a code bug, not a user-facing error.

    Accepts both ``1.2.3.0/24`` and bare ``1.2.3.4`` (treated as
    ``/32`` for v4, ``/128`` for v6).
    """
    network = ipaddress.ip_network(cidr, strict=False)

    # ``in`` operator on ipaddress objects handles v4/v6 mismatch
    # correctly (returns False instead of raising), so we don't need
    # an explicit version check.
    return ip in network


def _validate_cidr_string(cidr: str) -> None:
    """Validate that ``cidr`` is a syntactically-correct CIDR string.

    Raises :class:`ScopeViolationError` with ``reason="malformed_cidr"``
    if the string is empty or cannot be parsed by
    :func:`ipaddress.ip_network`. This pre-validation runs for EVERY
    CIDR in scope.json regardless of whether the current target
    resolves to an IP, so a typo in an out-of-scope rule cannot
    silently disable a security exclusion.
    """
    if not cidr or not cidr.strip():
        raise ScopeViolationError(
            "Empty CIDR string encountered in scope configuration. "
            "Remove the empty entry from in_scope_cidrs / "
            "out_of_scope_cidrs in scope.json.",
            target="(scope config)",
            reason="malformed_cidr",
            details={"cidr": cidr},
        )
    try:
        ipaddress.ip_network(cidr.strip(), strict=False)
    except ValueError as exc:
        raise ScopeViolationError(
            f"Malformed CIDR '{cidr}' in scope configuration: {exc}. "
            f"Fix the entry in scope.json; the framework refuses to "
            f"silently skip a malformed rule.",
            target="(scope config)",
            reason="malformed_cidr",
            details={"cidr": cidr, "cause": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = ["enforce_scope"]
