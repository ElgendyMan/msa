"""
src/agents/crawler_parser.py
============================

Node 6 of the 16-node LangGraph framework: the **Crawler Parser**.

This node is the LLM-backed counterpart to the Recon Parser (Node 5).
Where the Recon Parser ingests Nmap/Subfinder output, the Crawler
Parser ingests Playwright crawl output: the messy, semi-structured
log of URLs visited, HTML forms discovered, JavaScript files loaded,
cookies set, and DOM storage entries populated during a Playwright
crawl of the target web application.

Why an LLM and not a Playwright HAR parser?
-------------------------------------------
A Playwright crawl can produce output in several formats depending on
how the operator ran it:

- Raw HAR (HTTP Archive) JSON — verbose, nested, includes every
  request/response pair.
- Playwright's ``page.content()`` HTML dumps — useful for form
  extraction but require HTML parsing to find form fields.
- Operator-annotated mixed logs ("# this is the login form",
  pasted curl commands, manual notes about discovered endpoints).
- ``playwright-cli`` text output (request log with method/url/status).

A capable LLM handles all of these without per-format parsers. The
structured-output schema enforces what the LLM is allowed to return;
the system prompt enforces what the LLM is allowed to infer.

What the LLM is — and is NOT — allowed to do
--------------------------------------------
ALLOWED:
- Extract URLs, parameters, forms, JS files, cookies, and storage
  entries that literally appear in the crawl output.
- Infer ``location`` (query / body / header / cookie / etc.) for a
  parameter from the context in which it appears (e.g. a parameter
  in a ``<form method="POST">`` field with ``enctype="multipart"``
  is ``body_multipart``).
- Infer ``param_type`` (string / int / bool / json / file) from the
  parameter's value or its declared form-field type.

FORBIDDEN:
- Inventing endpoints or parameters that do not appear in the crawl
  output (the Hypothesis Analyzer's job is to reason about
  *possible* injection points; it cannot do that if the Crawler
  Parser hallucinated them).
- Pre-classifying parameters as ``is_reflected=True`` or
  ``is_injectable=True``. These flags are the Hypothesis Analyzer's
  responsibility — it will evaluate them with full request/response
  context, not the Crawler Parser with only crawl output.
- Inventing secrets in JS files. If the crawl output includes a
  secret (e.g. an AWS key fingerprint), preserve the redacted
  description; do NOT extract the raw secret value.

Defense-in-depth
----------------
The system prompt instructs the LLM to set ``is_reflected=False`` and
``is_injectable=False`` for every parameter. The
:func:`_enforce_flags_false` helper forcibly resets them post-LLM.
This three-layer enforcement (prompt → schema → post-process) matches
the Recon Parser's ``is_web=False`` pattern and is the framework's
standard approach to LLM-controlled flags.

LangGraph contract
------------------
::

    async def parse_crawler_data(state: AppState) -> dict:

- Reads: ``state["raw_crawler_output"]`` (str),
         ``state["target"]`` (:class:`~src.shared.schemas.Target`).
- Writes: returns ``{"crawler_data": <CrawlerResult>}``.

Raises
------
- :class:`PentestFrameworkError` — missing or empty raw input.
- :class:`LLMOutputParsingError` — LLM returned malformed JSON or
  JSON that failed Pydantic validation.
- :class:`LLMRateLimitError` — LLM provider returned 429 / quota
  exceeded.
"""

from __future__ import annotations

import re
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
from src.shared.schemas import CrawlerResult, FormInfo, Parameter, Target
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT: str = """\
You are an expert web crawler data parser embedded in an automated web
pentesting framework. Your sole job is to convert raw Playwright crawl
output into a structured JSON document that conforms exactly to the
provided CrawlerResult schema.

INPUT SOURCES YOU MUST HANDLE
-----------------------------
The raw crawler text may be any one of, or an unstructured mix of:
- Playwright HAR (HTTP Archive) JSON output
- Playwright page.content() HTML dumps
- Playwright request log text output (method/url/status lines)
- Operator-annotated mixed logs (lines starting with '#' are comments)
- cURL commands pasted by the operator (for manual endpoint additions)
- Browser DevTools "Copy as fetch" / "Copy as cURL" output

You must parse whatever you receive. Do not reject any input format.

EXTRACTION RULES
----------------
1. URLS: Extract every distinct URL discovered during the crawl.
   Include the target URL itself if it appears in the crawl output.
   Deduplicate by exact URL string. Preserve the original URL form
   (do not normalize trailing slashes, query-string order, etc.).

2. PARAMETERS: Extract every parameter you observe. A parameter may
   appear in:
   - Query string (?key=value)
   - URL path (/api/users/{id})
   - HTTP request header (Cookie, Authorization, X-Custom-Header, etc.)
   - Cookie value (parse the Cookie header into individual cookies)
   - Form body (application/x-www-form-urlencoded)
   - JSON body (application/json — flatten top-level keys only)
   - XML body (application/xml — flatten top-level elements only)
   - Multipart body (multipart/form-data — file uploads)

   For each parameter, set:
   - name (str, required): the parameter name.
   - location (enum, required): query | path | header | cookie |
     body_form | body_json | body_xml | body_multipart.
   - value (str | null): the parameter value if observed. Set to null
     if the parameter was observed but had no value (e.g. a flag-style
     query param like ?debug).
   - param_type (str | null): inferred type — 'string', 'int', 'bool',
     'json', 'file', 'xml', 'null'. Infer from the value or the
     declared form-field type (e.g. <input type="number"> → 'int').
     Set to null if you cannot infer the type.
   - is_reflected (bool, MUST be False): set this to False for EVERY
     parameter. A downstream node (the Hypothesis Analyzer) will
     evaluate reflection by replaying the request and inspecting the
     response. Do NOT pre-classify.
   - is_injectable (bool, MUST be False): set this to False for EVERY
     parameter. The Hypothesis Analyzer decides injectability, not
     the parser.

3. FORMS: Extract every HTML form discovered. For each form:
   - action (HttpUrl, required): the form's action URL.
   - method (enum: GET | POST | PUT | PATCH | DELETE | HEAD |
     OPTIONS | CONNECT | TRACE, required): the form's HTTP method.
   - fields (list[Parameter]): each form field as a Parameter with
     location='body_form' (or 'body_multipart' if enctype is
     multipart/form-data). All fields MUST have is_reflected=False
     and is_injectable=False.
   - has_csrf_token (bool): True if the form contains a field whose
     name looks like a CSRF token (csrf_token, _csrf, authenticity_token,
     __RequestVerificationToken, etc.).
   - enctype (str | null): the form's encoding type if specified.

4. JS_FILES: Extract every JavaScript file referenced by the
   application. For each:
   - url (HttpUrl, required): the JS file URL.
   - content_sha256 (str | null): the SHA-256 of the file content, if
     the crawl output includes it.
   - size_bytes (int | null): file size in bytes, if known.
   - endpoints_discovered (list[str]): any API endpoints you can
     extract from the JS source (e.g. fetch('/api/users'), axios.get
     ('/v1/profile'), hardcoded URL strings). ONLY include endpoints
     that literally appear in the JS source — do NOT infer.
   - secrets_discovered (list[str]): REDACTED descriptions of any
     secrets found (e.g. 'aws_access_key_id pattern match', 'generic
     API key pattern match'). NEVER include the raw secret value —
     only the pattern description.
   - interesting_patterns (list[str]): any other interesting patterns
     (e.g. 'eval() call', 'innerHTML assignment', 'document.write',
     'jQuery $ selector', 'React component', 'Vue directive').

5. COOKIES: Extract every cookie set during the crawl. Each cookie is
   a dict with at minimum 'name' and 'value' keys; include 'domain',
   'path', 'secure', 'httpOnly', 'sameSite' if available.

6. STORAGE_ENTRIES: Extract every localStorage, sessionStorage, and
   IndexedDB entry populated during the crawl. Each entry is a dict
   with at minimum 'storage_type' ('local' | 'session' | 'indexeddb'),
   'key', and 'value'. Include 'origin' if available.

7. CRAWL_DEPTH (int): the maximum link-follow depth reached during
   the crawl, if reported in the output. Set to 0 if unknown.

8. CRAWL_DURATION_SECONDS (float): the crawl duration in seconds, if
   reported in the output. Set to 0.0 if unknown.

CRITICAL PROHIBITIONS — VIOLATING THESE IS A P0 BUG
----------------------------------------------------
1. NEVER INVENT ENDPOINTS OR PARAMETERS. If a URL or parameter does
   not appear in the crawl output, do not include it. Do not "guess"
   that an endpoint "probably exists" because it follows a REST
   convention. Do not "add" a parameter because "most forms have a
   CSRF token" — only include it if you literally observe it.

2. NEVER MARK is_reflected=True OR is_injectable=True. Every
   parameter — top-level AND inside forms — MUST have both flags set
   to False. The Hypothesis Analyzer evaluates these flags with full
   request/response context; the parser does not have that context.

3. NEVER INCLUDE RAW SECRET VALUES. If you discover a secret in a JS
   file (e.g. an AWS access key), include only a redacted description
   ('aws_access_key_id pattern match') in secrets_discovered. NEVER
   include the actual key value.

4. NEVER NORMALIZE URLS. Preserve the original URL form. Do not strip
   trailing slashes, reorder query parameters, percent-decode, or
   "fix" URL encoding. Downstream nodes need the exact form.

5. NEVER INFER FORMS FROM HTML TEMPLATES. If the crawl output includes
   HTML, only extract <form> elements that are actually rendered (i.e.
   appear in page.content()). Do not extract forms from <template>
   elements or commented-out HTML.

OUTPUT FORMAT
-------------
Return a single JSON object matching the CrawlerResult schema. Do NOT
wrap it in markdown code fences. Do NOT add commentary before or
after the JSON. Do NOT return a JSON array — the schema requires a
single object.
"""


_USER_PROMPT_TEMPLATE: str = """\
Parse the following raw Playwright crawl output into the CrawlerResult \
schema. Remember: is_reflected=False and is_injectable=False for every \
parameter (including form fields), and never invent endpoints or \
parameters.

TARGET URL (for context only):
{target_url}

RAW CRAWLER OUTPUT:
---BEGIN RAW OUTPUT---
{raw_output}
---END RAW OUTPUT---

Return the JSON object now.
"""


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


async def parse_crawler_data(state: AppState) -> dict[str, Any]:
    """LangGraph Node 6: parse raw Playwright crawl output into a
    structured :class:`~src.shared.schemas.CrawlerResult`.

    Parameters
    ----------
    state:
        The current :class:`~src.shared.state.AppState`. Must contain
        ``raw_crawler_output`` (str) and ``target``
        (:class:`~src.shared.schemas.Target`).

    Returns
    -------
    dict
        ``{"crawler_data": <CrawlerResult>}`` — a fresh frozen
        Pydantic instance ready to be merged into the ``crawler_data``
        channel (which uses the default overwrite reducer).

    Raises
    ------
    PentestFrameworkError
        If ``raw_crawler_output`` is missing, None, or empty-after-strip.
    LLMOutputParsingError
        If the LLM returns malformed JSON or JSON that fails Pydantic
        validation against :class:`CrawlerResult`. The raw LLM output
        is preserved in ``error.raw_output`` and
        ``error.details["raw_output_preview"]``.
    LLMRateLimitError
        If the LLM provider returns 429 / quota-exceeded and the
        built-in retry budget (``settings.LLM_MAX_RETRIES``) is
        exhausted.
    """
    log = get_logger("crawler_parser")

    # ---------------------------------------------------------------
    # 1. Read + validate inputs.
    # ---------------------------------------------------------------
    raw_output: Any = state.get("raw_crawler_output")
    if raw_output is None or not isinstance(raw_output, str) or not raw_output.strip():
        raise PentestFrameworkError(
            "Crawler Parser cannot run: state['raw_crawler_output'] is missing, "
            "None, or empty. The Orchestrator must populate this channel "
            "with the raw Playwright crawl output (HAR JSON, page.content() "
            "HTML, request log text, or mixed annotated logs) before "
            "routing here.",
            details={
                "available_keys": list(state.keys()),
                "raw_crawler_output_present": "raw_crawler_output" in state,
                "raw_crawler_output_type": type(state.get("raw_crawler_output")).__name__,
                "raw_crawler_output_length": (
                    len(raw_output) if isinstance(raw_output, str) else 0
                ),
            },
        )

    target: Target | None = state.get("target")
    target_url_str: str = str(target.url) if target is not None else "(unknown)"

    # Bind context for every subsequent log line.
    log = log.bind(target_url=target_url_str)

    log.info(
        "crawler_parse_started",
        input_size_bytes=len(raw_output),
        input_size_kb=round(len(raw_output) / 1024.0, 2),
        input_lines=raw_output.count("\n") + 1,
    )

    # ---------------------------------------------------------------
    # 2. Build the structured-output LLM.
    # ---------------------------------------------------------------
    structured_llm = gemini_flash.with_structured_output(CrawlerResult)

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        target_url=target_url_str,
        raw_output=raw_output,
    )

    # ---------------------------------------------------------------
    # 3. Invoke the LLM and convert errors to framework exceptions.
    # ---------------------------------------------------------------
    try:
        parsed_result: CrawlerResult = await structured_llm.ainvoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except OutputParserException as exc:
        raw_llm_output = _extract_raw_output_from_parser_exception(exc)
        log.warning(
            "llm_output_parse_failed",
            error_type=type(exc).__name__,
            raw_output_length=len(raw_llm_output),
        )
        raise LLMOutputParsingError(
            f"Crawler Parser LLM returned output that could not be parsed "
            f"into CrawlerResult: {exc}",
            raw_output=raw_llm_output,
            schema_name="CrawlerResult",
            details={
                "input_size_bytes": len(raw_output),
                "lc_exception_type": type(exc).__name__,
            },
        ) from exc
    except ValidationError as exc:
        log.warning(
            "pydantic_validation_failed",
            error_type=type(exc).__name__,
            error_count=len(exc.errors()),
        )
        raise LLMOutputParsingError(
            f"Crawler Parser LLM output failed Pydantic validation against "
            f"CrawlerResult: {exc}",
            raw_output="(Pydantic raised ValidationError directly; no "
                       "raw LLM string available)",
            schema_name="CrawlerResult",
            details={
                "input_size_bytes": len(raw_output),
                "validation_errors": exc.errors()[:10],
            },
        ) from exc
    except Exception as exc:
        exc_repr = repr(exc)
        exc_str = str(exc)
        haystack = f"{exc_repr} {exc_str}"

        if _is_rate_limit_error(haystack):
            log.warning(
                "llm_rate_limited",
                error_type=type(exc).__name__,
                error_message=exc_str[:200],
            )
            raise LLMRateLimitError(
                f"Crawler Parser LLM call hit a rate limit or quota: {exc}",
                provider="google",
                model="gemini-2.5-flash",
                retry_after_seconds=_extract_retry_after(exc),
            ) from exc

        log.exception(
            "llm_call_unexpected_error",
            error_type=type(exc).__name__,
        )
        raise LLMOutputParsingError(
            f"Crawler Parser LLM call failed unexpectedly: {exc}",
            raw_output=f"(exception: {exc_repr})",
            schema_name="CrawlerResult",
            details={
                "input_size_bytes": len(raw_output),
                "exception_type": type(exc).__name__,
                "exception_repr": exc_repr[:500],
            },
        ) from exc

    # ---------------------------------------------------------------
    # 4. Post-process: enforce the is_reflected=False / is_injectable=False
    #    invariant on EVERY parameter (top-level AND inside forms).
    # ---------------------------------------------------------------
    parsed_result = _enforce_flags_false(parsed_result)

    # ---------------------------------------------------------------
    # 5. Log success metrics and return.
    # ---------------------------------------------------------------
    url_count = len(parsed_result.urls)
    param_count = len(parsed_result.parameters)
    form_count = len(parsed_result.forms)
    form_field_count = sum(len(f.fields) for f in parsed_result.forms)
    js_file_count = len(parsed_result.js_files)
    js_endpoint_count = sum(len(j.endpoints_discovered) for j in parsed_result.js_files)
    cookie_count = len(parsed_result.cookies)
    storage_count = len(parsed_result.storage_entries)

    log.info(
        "crawler_parse_complete",
        urls_extracted=url_count,
        parameters_extracted=param_count,
        forms_extracted=form_count,
        form_fields_extracted=form_field_count,
        js_files_extracted=js_file_count,
        js_endpoints_extracted=js_endpoint_count,
        cookies_extracted=cookie_count,
        storage_entries_extracted=storage_count,
        crawl_depth=parsed_result.crawl_depth,
        crawl_duration_seconds=parsed_result.crawl_duration_seconds,
    )

    return {"crawler_data": parsed_result}


# ---------------------------------------------------------------------------
# Internal helpers — defense-in-depth flag enforcement
# ---------------------------------------------------------------------------


def _enforce_flags_false(result: CrawlerResult) -> CrawlerResult:
    """Forcibly set ``is_reflected=False`` and ``is_injectable=False``
    on EVERY parameter in ``result``, including parameters nested
    inside forms.

    The system prompt instructs the LLM to do this, but defense-in-depth
    requires we verify it. If ANY parameter (top-level or form-embedded)
    has either flag set to True, we rebuild the :class:`CrawlerResult`
    with corrected parameters.

    Because the schemas are frozen (Pydantic ``frozen=True``), we use
    ``model_copy(update={...})`` at every level — CrawlerResult,
    FormInfo, and Parameter instances.

    Parameters
    ----------
    result:
        The :class:`CrawlerResult` returned by the LLM.

    Returns
    -------
    CrawlerResult
        Either the same instance (if no corrections were needed) or a
        new frozen instance with all flags correctly reset.
    """
    # Check if any parameter anywhere needs correction.
    needs_rebuild = _any_flag_true(result)
    if not needs_rebuild:
        return result

    # --- Rebuild top-level parameters ---
    new_params: list[Parameter] = [
        _fix_param_flags(p) for p in result.parameters
    ]

    # --- Rebuild forms (which contain their own parameters) ---
    new_forms: list[FormInfo] = [
        f.model_copy(
            update={"fields": [_fix_param_flags(p) for p in f.fields]}
        )
        if any(_param_needs_fix(p) for p in f.fields)
        else f
        for f in result.forms
    ]

    return result.model_copy(
        update={
            "parameters": new_params,
            "forms": new_forms,
        }
    )


def _any_flag_true(result: CrawlerResult) -> bool:
    """Return True if any parameter (top-level or inside a form) has
    ``is_reflected=True`` or ``is_injectable=True``."""
    for p in result.parameters:
        if p.is_reflected or p.is_injectable:
            return True
    for f in result.forms:
        for p in f.fields:
            if p.is_reflected or p.is_injectable:
                return True
    return False


def _param_needs_fix(p: Parameter) -> bool:
    """Return True if a single parameter has either flag set to True."""
    return p.is_reflected or p.is_injectable


def _fix_param_flags(p: Parameter) -> Parameter:
    """Return a corrected copy of ``p`` with both flags forced to False.

    If ``p`` already has both flags False, returns ``p`` unchanged
    (no unnecessary copy).
    """
    if not _param_needs_fix(p):
        return p
    return p.model_copy(
        update={
            "is_reflected": False,
            "is_injectable": False,
        }
    )


# ---------------------------------------------------------------------------
# Internal helpers — error classification (shared logic with recon_parser)
# ---------------------------------------------------------------------------


def _extract_raw_output_from_parser_exception(exc: OutputParserException) -> str:
    """Best-effort extraction of the raw LLM output from a LangChain
    :class:`OutputParserException`.

    Same logic as :func:`src.agents.recon_parser._extract_raw_output_from_parser_exception`.
    Duplicated here rather than imported to keep the agent modules
    independent (a future refactor could move this to a shared utility
    module if the duplication becomes a maintenance burden).
    """
    for attr_name in ("raw_output", "raw", "output", "text", "content"):
        candidate = getattr(exc, attr_name, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate

    msg = str(exc)
    for start_marker in ("Raw output: ", "Got: ", "Invalid JSON: "):
        if start_marker in msg:
            return msg.split(start_marker, 1)[1]

    return msg


def _is_rate_limit_error(haystack: str) -> bool:
    """Return True if the lowercased exception text indicates a rate
    limit or quota-exceeded error from the LLM provider.

    The haystack is lowercased before matching so callers can pass in
    the raw exception text in any case. Matching is intentionally
    broad — false positives here only mean we raise
    :class:`LLMRateLimitError` instead of :class:`LLMOutputParsingError`,
    and the Orchestrator's error handler can re-classify if needed.
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

    Returns None if no value can be extracted.
    """
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


__all__ = ["parse_crawler_data"]
