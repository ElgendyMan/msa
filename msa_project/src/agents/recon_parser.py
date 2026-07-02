"""
src/agents/recon_parser.py
==========================

Node 5 of the 16-node LangGraph framework: the **Recon Parser**.

This is the first LLM-backed node in the pipeline. Its job is to take
the raw, messy, heterogeneous recon output produced by external tools
(Nmap XML, Nmap text, Subfinder JSON, httpx JSON, or any mix thereof)
and convert it into a clean :class:`~src.shared.schemas.ReconResult`
Pydantic object that downstream nodes can reason about deterministically.

Why an LLM and not regex / XML parsers?
---------------------------------------
Real-world recon output is heterogeneous:

- Nmap emits XML by default but operators frequently paste the
  human-readable text output (-oA without the .xml extension).
- Subfinder emits JSON or plain newline-delimited domains.
- Operators sometimes paste mixed logs from multiple tools, sometimes
  with their own annotations ("# this is the API server").

Writing a parser for every possible combination would be a never-ending
maintenance burden. A capable LLM with a strict structured-output
schema handles this gracefully and is robust to format drift.

What the LLM is — and is NOT — allowed to do
--------------------------------------------
ALLOWED:
- Extract hosts, IP addresses, open ports, service names, banners,
  technologies, subdomains, and WAF fingerprints that appear in the
  raw text.
- Infer ``service_name`` from a banner when Nmap reports it as
  ``unknown`` but the banner clearly identifies the service.
- Set ``is_web=False`` for ALL services — web classification is the
  Web Filter node's job, not the parser's.

FORBIDDEN:
- Inventing hosts, ports, or services that do not appear in the text.
- "Correcting" data (e.g. changing an obvious typo in a banner).
- Inferring scope (e.g. adding a subdomain because "it probably exists").
- Marking any service as ``is_web=True``.

The system prompt enforces these constraints; the
``with_structured_output`` schema enforces them at the Pydantic layer.

LangGraph contract
------------------
::

    async def parse_recon_data(state: AppState) -> dict:

- Reads: ``state["raw_recon_output"]`` (str),
         ``state["target"]`` (:class:`~src.shared.schemas.Target`).
- Writes: returns ``{"recon_data": <ReconResult>}``.

Raises
------
- :class:`PentestFrameworkError` — missing or empty raw input.
- :class:`LLMOutputParsingError` — the LLM returned malformed JSON or
  JSON that failed Pydantic validation. Carries the raw LLM output
  for forensics.
- :class:`LLMRateLimitError` — the LLM provider returned 429 / quota
  exceeded and the built-in retry budget is exhausted.
"""

from __future__ import annotations

from typing import Any

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from src.shared.exceptions import (
    LLMOutputParsingError,
    LLMRateLimitError,
    PentestFrameworkError,
)
from src.shared.llm import gemini_flash
from src.shared.logging import get_logger
from src.shared.schemas import ReconResult, Target, WAFSignature
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT: str = """\
You are an expert security data parser embedded in an automated web
pentesting framework. Your sole job is to convert raw, heterogeneous
reconnaissance output into a structured JSON document that conforms
exactly to the provided schema.

INPUT SOURCES YOU MUST HANDLE
-----------------------------
The raw recon text may be any one of, or an unstructured mix of:
- Nmap XML output (-oX)
- Nmap human-readable text output (default terminal output)
- Subfinder JSON output (-json)
- Subfinder plain text output (newline-delimited domains)
- httpx JSON output
- httpx text output
- Operator-annotated mixed logs (lines starting with '#' are comments)

You must parse whatever you receive. Do not reject any input format.

EXTRACTION RULES
----------------
1. HOSTS: Extract every distinct host (hostname OR IP address) that
   appears in the input. A host may appear multiple times across
   different tool outputs — deduplicate by (hostname, ip_address)
   pair, merging services into the single host entry.

2. SERVICES: For each host, extract every open port and its service
   information. Fields:
   - port (int, required): the open TCP/UDP port number.
   - protocol (enum: http | https | ws | wss): infer from the service
     name. Use 'http' for plain HTTP, 'https' for TLS-wrapped HTTP,
     'ws' for plain WebSocket, 'wss' for TLS-wrapped WebSocket. Use
     'http' as the default for ambiguous web services.
   - service_name (str, required): the Nmap service name (e.g.
     'http', 'https', 'ssh', 'mysql'). If Nmap reports 'unknown' but
     the banner clearly identifies the service, infer it.
   - service_version (str | null): the version string if reported.
   - product (str | null): the product name if reported (e.g. 'nginx').
   - banner (str | null): the raw banner string if reported.
   - is_web (bool, MUST be False): set this to False for EVERY
     service. A downstream deterministic node (the Web Filter) will
     classify which services are web targets. Do NOT pre-classify.

3. SUBDOMAINS: Extract every subdomain string. Deduplicate.

4. TECHNOLOGIES_DETECTED: Extract every technology fingerprint
   (e.g. 'nginx/1.21', 'PHP/8.1', 'WordPress 6.2', 'React'). These
   typically come from httpx -tech-detect output or Nmap service
   banners.

5. WAF_SIGNATURE (enum: cloudflare | aws_waf | akamai | f5_bigip |
   imperva | sucuri | modsecurity | unknown | none): If the input
   contains evidence of a WAF (httpx -waf output, or a Server header
   like 'cloudflare', or a cf-ray header), classify it. Use 'none'
   if no WAF evidence is present. Use 'unknown' if WAF evidence is
   present but you cannot classify it.

CRITICAL PROHIBITIONS — VIOLATING THESE IS A P0 BUG
----------------------------------------------------
1. NEVER INVENT DATA. If a field is not present in the input, set it
   to null (for nullable fields) or omit the service / host entirely
   (for required fields). Do not guess, do not infer from "common
   sense", do not "fill in" missing ports because "they're usually
   open".

2. NEVER MARK is_web=True. Every service must have is_web=False.
   Web classification is the Web Filter node's responsibility.

3. NEVER INFER SCOPE. Do not add a subdomain because "it probably
   exists". Only extract what is literally in the text.

4. NEVER CORRECT DATA. If a banner has a typo, preserve the typo.
   If a port number is impossible (>65535), omit that service and
   move on — do not "fix" it.

5. NEVER INCLUDE THE TARGET URL IN web_endpoints. The Web Filter
   node will derive web_endpoints from the surviving services. Leave
   web_endpoints as an empty list — it will be populated downstream.

OUTPUT FORMAT
-------------
Return a single JSON object matching the ReconResult schema. Do NOT
wrap it in markdown code fences. Do NOT add commentary before or
after the JSON. Do NOT return a JSON array — the schema requires a
single object.
"""


_USER_PROMPT_TEMPLATE: str = """\
Parse the following raw reconnaissance output into the ReconResult \
schema. Remember: is_web=False for every service, web_endpoints must \
be an empty list, and never invent data.

TARGET URL (for context only — do not include in web_endpoints):
{target_url}

RAW RECON OUTPUT:
---BEGIN RAW OUTPUT---
{raw_output}
---END RAW OUTPUT---

Return the JSON object now.
"""


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


async def parse_recon_data(state: AppState) -> dict[str, Any]:
    """LangGraph Node 5: parse raw recon text into a structured
    :class:`~src.shared.schemas.ReconResult`.

    Parameters
    ----------
    state:
        The current :class:`~src.shared.state.AppState`. Must contain
        ``raw_recon_output`` (str) and ``target``
        (:class:`~src.shared.schemas.Target`).

    Returns
    -------
    dict
        ``{"recon_data": <ReconResult>}`` — a fresh frozen Pydantic
        instance ready to be merged into the ``recon_data`` channel
        (which uses the default overwrite reducer).

    Raises
    ------
    PentestFrameworkError
        If ``raw_recon_output`` is missing, None, or empty-after-strip.
    LLMOutputParsingError
        If the LLM returns malformed JSON or JSON that fails Pydantic
        validation against :class:`ReconResult`. The raw LLM output
        is preserved in ``error.raw_output`` and
        ``error.details["raw_output_preview"]``.
    LLMRateLimitError
        If the LLM provider returns 429 / quota-exceeded and the
        built-in retry budget (``settings.LLM_MAX_RETRIES``) is
        exhausted.
    """
    log = get_logger("recon_parser")

    # ---------------------------------------------------------------
    # 1. Read + validate inputs.
    # ---------------------------------------------------------------
    raw_output: str | None = state.get("raw_recon_output")
    if raw_output is None or not isinstance(raw_output, str) or not raw_output.strip():
        raise PentestFrameworkError(
            "Recon Parser cannot run: state['raw_recon_output'] is missing, "
            "None, or empty. The Orchestrator must populate this channel "
            "with the raw text output of Nmap / Subfinder / httpx before "
            "routing here.",
            details={
                "available_keys": list(state.keys()),
                "raw_recon_output_present": "raw_recon_output" in state,
                "raw_recon_output_type": type(state.get("raw_recon_output")).__name__,
                "raw_recon_output_length": (
                    len(raw_output) if isinstance(raw_output, str) else 0
                ),
            },
        )

    target: Target | None = state.get("target")
    target_url_str: str = str(target.url) if target is not None else "(unknown)"

    # Bind context for every subsequent log line.
    log = log.bind(target_url=target_url_str)

    log.info(
        "recon_parse_started",
        input_size_bytes=len(raw_output),
        input_size_kb=round(len(raw_output) / 1024.0, 2),
        input_lines=raw_output.count("\n") + 1,
    )

    # ---------------------------------------------------------------
    # 2. Build the structured-output LLM.
    # ---------------------------------------------------------------
    # ``with_structured_output`` enforces the Pydantic schema at the
    # LangChain layer. Gemini supports this via its native
    # ``response_schema`` feature; langchain-google-genai translates
    # the Pydantic model into a Gemini-compatible schema automatically.
    #
    # We rebuild the bound LLM on every call rather than caching at
    # module import — this keeps the LLM instance fresh and avoids
    # surprising shared-state bugs if the LLM config changes between
    # sessions.
    structured_llm = gemini_flash.with_structured_output(ReconResult)

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        target_url=target_url_str,
        raw_output=raw_output,
    )

    # ---------------------------------------------------------------
    # 3. Invoke the LLM and convert errors to framework exceptions.
    # ---------------------------------------------------------------
    try:
        parsed_result: ReconResult = await structured_llm.ainvoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except OutputParserException as exc:
        # LangChain's structured-output layer raises OutputParserException
        # when the LLM returns JSON that fails Pydantic validation. We
        # extract the raw output (LangChain stuffs it into the exception
        # message or .lc_attributes) and re-raise as LLMOutputParsingError.
        raw_llm_output = _extract_raw_output_from_parser_exception(exc)
        log.warning(
            "llm_output_parse_failed",
            error_type=type(exc).__name__,
            raw_output_length=len(raw_llm_output),
        )
        raise LLMOutputParsingError(
            f"Recon Parser LLM returned output that could not be parsed "
            f"into ReconResult: {exc}",
            raw_output=raw_llm_output,
            schema_name="ReconResult",
            details={
                "input_size_bytes": len(raw_output),
                "lc_exception_type": type(exc).__name__,
            },
        ) from exc
    except ValidationError as exc:
        # Defensive: should not happen because with_structured_output
        # catches validation errors internally and raises
        # OutputParserException, but Pydantic can in some configurations
        # raise ValidationError directly.
        log.warning(
            "pydantic_validation_failed",
            error_type=type(exc).__name__,
            error_count=len(exc.errors()),
        )
        raise LLMOutputParsingError(
            f"Recon Parser LLM output failed Pydantic validation against "
            f"ReconResult: {exc}",
            raw_output="(Pydantic raised ValidationError directly; no "
                       "raw LLM string available)",
            schema_name="ReconResult",
            details={
                "input_size_bytes": len(raw_output),
                "validation_errors": exc.errors()[:10],  # cap for log safety
            },
        ) from exc
    except Exception as exc:
        # Last-resort classification. We inspect the exception's
        # repr / message for rate-limit indicators (429, rate limit,
        # quota) and route to LLMRateLimitError if found; otherwise
        # we wrap in LLMOutputParsingError with the exception's str
        # as the "raw output" (better than nothing for forensics).
        exc_repr = repr(exc)
        exc_str = str(exc)
        haystack = f"{exc_repr} {exc_str}".lower()

        if _is_rate_limit_error(haystack):
            log.warning(
                "llm_rate_limited",
                error_type=type(exc).__name__,
                error_message=exc_str[:200],
            )
            raise LLMRateLimitError(
                f"Recon Parser LLM call hit a rate limit or quota: {exc}",
                provider="google",
                model="gemini-2.5-flash",
                retry_after_seconds=_extract_retry_after(exc),
            ) from exc

        # Truly unexpected error. Treat as a parse failure with the
        # exception text as the "raw output" — this preserves the
        # diagnostic info without crashing the whole session.
        log.exception(
            "llm_call_unexpected_error",
            error_type=type(exc).__name__,
        )
        raise LLMOutputParsingError(
            f"Recon Parser LLM call failed unexpectedly: {exc}",
            raw_output=f"(exception: {exc_repr})",
            schema_name="ReconResult",
            details={
                "input_size_bytes": len(raw_output),
                "exception_type": type(exc).__name__,
                "exception_repr": exc_repr[:500],
            },
        ) from exc

    # ---------------------------------------------------------------
    # 4. Post-process: enforce the "is_web=False" invariant.
    # ---------------------------------------------------------------
    # The system prompt instructs the LLM to set is_web=False for every
    # service, but we do NOT trust the LLM. We forcibly reset every
    # service's is_web flag here so the Web Filter node receives clean
    # input regardless of what the LLM returned.
    parsed_result = _enforce_is_web_false(parsed_result)

    # ---------------------------------------------------------------
    # 5. Log success metrics and return.
    # ---------------------------------------------------------------
    host_count = len(parsed_result.hosts)
    subdomain_count = len(parsed_result.subdomains)
    service_count = sum(len(h.services) for h in parsed_result.hosts)
    endpoint_count = len(parsed_result.web_endpoints)
    waf_sig = parsed_result.waf_signature.value if parsed_result.waf_signature else "none"

    log.info(
        "recon_parse_complete",
        hosts_extracted=host_count,
        subdomains_extracted=subdomain_count,
        services_extracted=service_count,
        web_endpoints_extracted=endpoint_count,
        technologies_detected=len(parsed_result.technologies_detected),
        waf_signature=waf_sig,
    )

    return {"recon_data": parsed_result}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _enforce_is_web_false(result: ReconResult) -> ReconResult:
    """Forcibly set ``is_web=False`` on every service in ``result``.

    The system prompt instructs the LLM to do this, but defense-in-depth
    requires we verify it. If any service has ``is_web=True``, we
    rebuild the :class:`ReconResult` with corrected services.

    Because the schemas are frozen (Pydantic ``frozen=True``), we use
    ``model_copy(update={...})`` at every level — host services lists
    and individual ServiceInfo instances.
    """
    needs_rebuild = any(
        any(svc.is_web for svc in host.services)
        for host in result.hosts
    )
    if not needs_rebuild:
        return result

    new_hosts = []
    for host in result.hosts:
        if not any(svc.is_web for svc in host.services):
            new_hosts.append(host)
            continue
        new_services = [
            svc.model_copy(update={"is_web": False}) if svc.is_web else svc
            for svc in host.services
        ]
        new_hosts.append(host.model_copy(update={"services": new_services}))

    return result.model_copy(update={"hosts": new_hosts})


def _extract_raw_output_from_parser_exception(exc: OutputParserException) -> str:
    """Best-effort extraction of the raw LLM output from a LangChain
    :class:`OutputParserException`.

    LangChain stuffs the raw output into different attributes depending
    on which parser raised the exception. We try the most common
    locations in order and fall back to the exception's string repr.
    """
    # Common attribute names across LangChain parser versions.
    for attr_name in ("raw_output", "raw", "output", "text", "content"):
        candidate = getattr(exc, attr_name, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate

    # Some parsers embed the raw output in the exception message
    # between known markers. Try to extract it.
    msg = str(exc)
    for marker_pair in (
        ("Raw output: ", ""),
        ("Got: ", ""),
        ("Invalid JSON: ", ""),
    ):
        start_marker, _ = marker_pair
        if start_marker in msg:
            return msg.split(start_marker, 1)[1]

    # Last resort: the full exception string.
    return msg


def _is_rate_limit_error(haystack: str) -> bool:
    """Return True if the lowercased exception text indicates a rate
    limit or quota-exceeded error from the LLM provider.

    The matching is intentionally broad — false positives here only
    mean we raise ``LLMRateLimitError`` instead of
    ``LLMOutputParsingError``, and the Orchestrator's error handler
    can re-classify if needed. False negatives are worse: a rate
    limit treated as a parse error would not trigger back-off.

    The haystack is lowercased before matching so callers can pass in
    the raw exception text in any case (``"429 Too Many Requests"``,
    ``"RESOURCE_EXHAUSTED"``, etc.) without pre-normalizing.
    """
    if not haystack:
        return False
    normalized = haystack.lower()
    rate_limit_indicators = (
        "429",
        "rate limit",
        "rate_limit",
        "ratelimit",
        "quota",
        "quota exceeded",
        "resourceexhausted",
        "resource_exhausted",
        "resource exhausted",
        "too many requests",
        "throttled",
        "throttling",
        "retry-after",
        "retry_after",
        "service unavailable",
    )
    return any(indicator in normalized for indicator in rate_limit_indicators)


def _extract_retry_after(exc: Exception) -> float | None:
    """Best-effort extraction of a ``Retry-After`` value (in seconds)
    from an LLM provider exception.

    Returns None if no value can be extracted. The Orchestrator uses
    this hint to decide how long to back off before retrying.
    """
    # Check common attribute names.
    for attr_name in ("retry_after", "retry_after_seconds", "retry_after_secs"):
        candidate = getattr(exc, attr_name, None)
        if isinstance(candidate, (int, float)) and candidate > 0:
            return float(candidate)

    # Check the exception message for "retry after N seconds" patterns.
    import re
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


__all__ = ["parse_recon_data"]
