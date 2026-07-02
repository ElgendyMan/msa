"""
src/agents/hypothesis_analyzer.py
=================================

Node 8 of the 16-node LangGraph framework: the **Hypothesis Analyzer** —
the "Brain" that identifies potential vulnerabilities.

This node consumes the structured attack surface produced by the Recon
Parser (Node 5) and Crawler Parser (Node 6) and generates a list of
plausible vulnerability hypotheses for the Payload Generator (Node 9)
to act on. Each hypothesis is an unverified suspicion — the Validator
(Node 12) is the arbiter of truth.

Why DeepSeek R1?
----------------
The Architect specified DeepSeek R1 (``deepseek-reasoner``) for this
node because hypothesis generation requires *reasoning*: the LLM must
correlate technologies, parameter locations, form fields, and JS
endpoints to identify which attack classes are plausible. R1's
chain-of-thought capability lets it work through the attack surface
step-by-step before emitting hypotheses.

R1 does NOT support ``with_structured_output()``, so we use a private
wrapper model :class:`_HypothesisList` with
:class:`PydanticOutputParser` and inject
``parser.get_format_instructions()`` into the system prompt. R1's
output typically contains a ``<think>...</think>`` reasoning block
followed by the final JSON; we preprocess the output to extract the
JSON before parsing (same pattern as the Validator).

Defense-in-depth post-processing
--------------------------------
The LLM is instructed to follow several constraints, but we do NOT
trust it. Two post-processing helpers enforce them:

1. **Confidence clamping** (:func:`_enforce_confidence_range`): R1
   might return ``confidence=1.0`` (overconfident) for a hypothesis
   that is merely plausible. We clamp every hypothesis's confidence
   to ``[0.0, 0.7]`` — hypotheses are unverified suspicions, never
   certainties. Values in range are left unchanged; values > 0.7 are
   clamped to 0.7 with a warning log.

2. **Hallucinated parameter removal**
   (:func:`_remove_hallucinated_parameters`): The LLM might invent a
   parameter name that does not appear in the crawler data. For each
   hypothesis, if ``target_parameter`` is set but its ``name`` does
   not match any parameter observed in ``crawler_data`` (top-level or
   form-embedded), we set ``target_parameter`` to ``None`` and log a
   warning. The hypothesis is kept (the vulnerability might still be
   plausible at the URL level), but the hallucinated parameter
   reference is removed.

3. **Empty-hypothesis-list is valid**: if the LLM returns zero
   hypotheses, that is a legitimate result (the attack surface might
   not have obvious vulnerabilities). We do NOT raise an error.

LangGraph contract
------------------
::

    async def analyze_hypotheses(state: AppState) -> dict:

- Reads: ``state["crawler_data"]`` (mandatory),
         ``state["recon_data"]`` (optional but recommended),
         ``state["target"]`` (optional, used for logging context).
- Writes: returns ``{"hypotheses": [<Hypothesis>, ...]}`` — a list
  ready to be merged into the ``hypotheses`` channel via the
  ``operator.add`` reducer. Multiple calls to this node accumulate
  hypotheses in the state.

Raises
------
- :class:`PentestFrameworkError` — if ``crawler_data`` is missing.
- :class:`LLMOutputParsingError` — LLM output could not be parsed
  into :class:`_HypothesisList`.
- :class:`LLMRateLimitError` — LLM provider returned 429 / quota.
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
    PentestFrameworkError,
)
from src.shared.llm import deepseek_r1
from src.shared.logging import get_logger
from src.shared.schemas import (
    CrawlerResult,
    Hypothesis,
    ReconResult,
    Target,
    VulnerabilityCategory,
)
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Private wrapper model for R1 structured output
# ---------------------------------------------------------------------------


class _HypothesisList(BaseModel):
    """Private wrapper model for R1's structured output.

    R1 does not support ``with_structured_output()``, so we use
    :class:`PydanticOutputParser` with this wrapper. The parser's
    ``get_format_instructions()`` describes this schema to the LLM,
    and the LLM returns a JSON object with a single ``hypotheses`` key
    containing a list of :class:`Hypothesis` objects.

    The config matches the framework's standard Pydantic settings
    (``extra="forbid"``, ``str_strip_whitespace=True``) so malformed
    LLM output fails loudly rather than silently dropping fields.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=False,
    )

    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="List of vulnerability hypotheses generated from "
        "the attack surface analysis. Each hypothesis is an unverified "
        "suspicion — the Validator node will confirm or deny it.",
    )


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_TEMPLATE: str = """\
You are a Lead Penetration Tester. Your job is to analyze the attack \
surface of a web application and generate a list of plausible \
vulnerability hypotheses.

You must think step-by-step:
1. Examine the URLs discovered during crawling. Look for patterns \
that suggest injection points (query parameters, RESTful resource IDs, \
API endpoints).
2. Examine the parameters. For each parameter, consider: what \
vulnerability classes could apply given its location (query, body, \
header, cookie), its inferred type, and the endpoint it was observed on?
3. Examine the forms. Form fields are prime injection targets — \
especially login forms (SQLi, auth bypass), search forms (XSS), and \
file upload forms (path traversal, RCE).
4. Examine the JS files and their discovered endpoints. Look for \
undocumented API routes, hardcoded URL patterns, and interesting \
function calls (eval, innerHTML, document.write).
5. Examine the technologies detected. Specific technologies have \
known vulnerability classes (e.g. PHP → SQLi/LFI; React → XSS via \
dangerouslySetInnerHTML; Express → prototype pollution; GraphQL → \
introspection/batching attacks).
6. For each plausible vulnerability, formulate a hypothesis with a \
specific target parameter (if applicable) and a confidence score.

VULNERABILITY CATEGORIES TO CONSIDER:
- SQL Injection (sqli, sqli_blind, sqli_time)
- Cross-Site Scripting (xss_reflected, xss_stored, xss_dom)
- Server-Side Request Forgery (SSRF, ssrf)
- Broken Object Level Authorization / IDOR (bola, idor)
- Broken Function Level Authorization (bfla)
- Command Injection (command_injection)
- Path Traversal (path_traversal)
- Open Redirect (open_redirect)
- Server-Side Template Injection (ssti)
- XML External Entity (xxe)
- GraphQL vulnerabilities (graphql_introspection, graphql_batching)
- JWT vulnerabilities (jwt_none_alg, jwt_weak_secret)
- Insecure Deserialization (deserialization)
- CSRF (csrf)
- Business Logic flaws (business_logic)
- Race Conditions (race_condition)

CONFIDENCE SCORING RULES — VIOLATING THESE IS A P0 BUG:
1. Confidence must be in [0.3, 0.7]. These are unverified suspicions, \
not confirmed vulnerabilities.
2. NEVER use confidence 1.0. A hypothesis is a guess that needs \
proof, not a certainty.
3. 0.7 = "strong suspicion based on clear indicators" (e.g. a search \
parameter reflected unsanitized in HTML).
4. 0.5 = "moderate suspicion based on technology + parameter type" \
(e.g. a PHP app with a numeric ID parameter — SQLi is plausible but \
unproven).
5. 0.3 = "weak suspicion based on general attack surface" (e.g. any \
form might have CSRF, but we have no specific indicator).

CRITICAL PROHIBITIONS:
1. DO NOT write payloads. Your job is to identify WHAT might be \
vulnerable, not HOW to exploit it. The Payload Generator node will \
craft PoCs based on your hypotheses.
2. DO NOT hallucinate parameters. Only reference parameters that \
literally appear in the provided crawler data. If a hypothesis \
applies to a URL generally (not a specific parameter), set \
target_parameter to null.
3. DO NOT generate more than 10 hypotheses. Focus on the highest-\
signal suspicions. Quality over quantity.
4. DO NOT include hypotheses with confidence below 0.3 — they are \
too weak to be worth testing.
5. Each hypothesis must have a unique, specific reasoning. Do not \
generate 5 SQLi hypotheses that all say "parameter might be \
injectable" — differentiate them by target URL, parameter, or \
attack variant.

OUTPUT FORMAT:
{format_instructions}

Output ONLY the JSON object. Do not wrap it in markdown code fences. \
Do not add commentary. If you need to reason, do so in a \
<think>...</think> block BEFORE the JSON, then output the JSON on its own.
"""


_USER_PROMPT_TEMPLATE: str = """\
Analyze the following attack surface and generate vulnerability \
hypotheses. Think step-by-step, then output the JSON _HypothesisList.

=== TARGET ===
URL: {target_url}

=== TECHNOLOGIES DETECTED ===
{technologies}

=== URLS DISCOVERED ({url_count_total} total, showing first {url_count_shown}) ===
{urls}

=== PARAMETERS DISCOVERED ({param_count_total} total, showing first {param_count_shown}) ===
{parameters}

=== FORMS DISCOVERED ({form_count} total) ===
{forms}

=== JS FILES ({js_file_count} total, {js_endpoint_count} endpoints) ===
{js_files}

=== REQUIRED OUTPUT ===
Return a JSON object with a "hypotheses" key containing a list of \
Hypothesis objects. Each hypothesis must include:
- category: one of the VulnerabilityCategory enum values
- target_url: the URL where the vulnerability is suspected
- target_parameter: the specific Parameter object (name, location) \
from the crawler data, or null if the hypothesis applies to the URL \
generally
- confidence: float in [0.3, 0.7]
- reasoning: 1-2 sentences explaining WHY this vulnerability is \
plausible here
- evidence: list of specific evidence pointers (e.g. "parameter 'id' \
is numeric and appears in query string", "technology 'PHP/7.4' \
detected")
- prerequisites: list of conditions that must be true for the \
vulnerability to exist (e.g. "parameter must be passed to SQL query \
without parameterization")

Return the JSON object now.
"""


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


async def analyze_hypotheses(state: AppState) -> dict[str, Any]:
    """LangGraph Node 8: analyze the attack surface and generate
    vulnerability hypotheses.

    Parameters
    ----------
    state:
        The current :class:`~src.shared.state.AppState`. Must contain
        ``crawler_data`` (a :class:`~src.shared.schemas.CrawlerResult`).
        ``recon_data`` is optional but recommended (provides technology
        fingerprints). ``target`` is optional (used for logging context).

    Returns
    -------
    dict
        ``{"hypotheses": [<Hypothesis>, ...]}`` — a list of zero or
        more hypotheses, ready to be merged into the ``hypotheses``
        channel via the ``operator.add`` reducer.

    Raises
    ------
    PentestFrameworkError
        If ``crawler_data`` is missing.
    LLMOutputParsingError
        If the LLM output cannot be parsed into :class:`_HypothesisList`.
    LLMRateLimitError
        If the DeepSeek API returns 429 / quota-exceeded.
    """
    log = get_logger("hypothesis_analyzer")

    # ---------------------------------------------------------------
    # 1. Read + validate inputs.
    # ---------------------------------------------------------------
    crawler_data: CrawlerResult | None = state.get("crawler_data")
    if crawler_data is None:
        raise PentestFrameworkError(
            "Hypothesis Analyzer cannot run: state['crawler_data'] is "
            "missing. The Crawler Parser must run before the Hypothesis "
            "Analyzer.",
            details={
                "available_keys": list(state.keys()),
                "has_crawler_data": "crawler_data" in state,
            },
        )

    recon_data: ReconResult | None = state.get("recon_data")
    target: Target | None = state.get("target")
    target_url_str: str = str(target.url) if target is not None else "(unknown)"

    # Bind context for every subsequent log line.
    log = log.bind(target_url=target_url_str)

    log.info(
        "hypothesis_analysis_started",
        urls_available=len(crawler_data.urls),
        parameters_available=len(crawler_data.parameters),
        forms_available=len(crawler_data.forms),
        js_files_available=len(crawler_data.js_files),
        has_recon_data=recon_data is not None,
    )

    # ---------------------------------------------------------------
    # 2. Build the PydanticOutputParser and prompts.
    # ---------------------------------------------------------------
    parser: PydanticOutputParser = PydanticOutputParser(
        pydantic_object=_HypothesisList
    )

    system_prompt: str = _SYSTEM_PROMPT_TEMPLATE.format(
        format_instructions=parser.get_format_instructions()
    )

    user_prompt: str = _build_user_prompt(crawler_data, recon_data, target_url_str)

    # ---------------------------------------------------------------
    # 3. Invoke DeepSeek R1.
    # ---------------------------------------------------------------
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
            log.warning(
                "llm_rate_limited",
                error_type=type(exc).__name__,
                error_message=exc_str[:200],
            )
            raise LLMRateLimitError(
                f"Hypothesis Analyzer LLM call hit a rate limit or quota: {exc}",
                provider="deepseek",
                model="deepseek-reasoner",
                retry_after_seconds=_extract_retry_after(exc),
            ) from exc

        log.exception(
            "llm_call_unexpected_error",
            error_type=type(exc).__name__,
        )
        raise LLMOutputParsingError(
            f"Hypothesis Analyzer LLM call failed unexpectedly: {exc}",
            raw_output=f"(exception: {exc_repr})",
            schema_name="_HypothesisList",
            details={
                "exception_type": type(exc).__name__,
                "exception_repr": exc_repr[:500],
            },
        ) from exc

    raw_llm_output: str = response_message.content

    # ---------------------------------------------------------------
    # 4. Parse the LLM output (with R1 <think> preprocessing fallback).
    # ---------------------------------------------------------------
    parsed_list: _HypothesisList | None = None

    # First, try the parser directly.
    try:
        parsed_list = await parser.aparse(raw_llm_output)
    except (OutputParserException, ValidationError):
        log.debug("parser_direct_failed_trying_preprocessed")
        pass

    # If the direct parse failed, preprocess and try again.
    if parsed_list is None:
        cleaned: str = _preprocess_llm_output(raw_llm_output)
        try:
            parsed_list = _HypothesisList.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            log.warning(
                "llm_output_parse_failed",
                error_type=type(exc).__name__,
                raw_output_length=len(raw_llm_output),
            )
            raise LLMOutputParsingError(
                f"Hypothesis Analyzer LLM output could not be parsed "
                f"into _HypothesisList: {exc}",
                raw_output=raw_llm_output,
                schema_name="_HypothesisList",
                details={
                    "cleaned_output_preview": cleaned[:500],
                    "parse_error": str(exc)[:500],
                },
            ) from exc

    hypotheses: list[Hypothesis] = parsed_list.hypotheses

    # ---------------------------------------------------------------
    # 5. Defense-in-depth: enforce confidence range.
    # ---------------------------------------------------------------
    hypotheses = _enforce_confidence_range(hypotheses, log)

    # ---------------------------------------------------------------
    # 6. Defense-in-depth: remove hallucinated parameters.
    # ---------------------------------------------------------------
    hypotheses = _remove_hallucinated_parameters(hypotheses, crawler_data, log)

    # ---------------------------------------------------------------
    # 7. Log success metrics and return.
    # ---------------------------------------------------------------
    category_counts: dict[str, int] = {}
    for h in hypotheses:
        cat: str = h.category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1

    log.info(
        "hypothesis_analysis_complete",
        hypotheses_generated=len(hypotheses),
        category_summary=category_counts,
        avg_confidence=(
            round(sum(h.confidence for h in hypotheses) / len(hypotheses), 3)
            if hypotheses else 0.0
        ),
    )

    return {"hypotheses": hypotheses}


# ---------------------------------------------------------------------------
# Internal: user prompt construction
# ---------------------------------------------------------------------------

#: Maximum number of items to include in the prompt for each section.
#: Truncating keeps the prompt within R1's context window; the LLM
#: rarely needs more than the first N items to identify patterns.
_MAX_URLS_IN_PROMPT: int = 50
_MAX_PARAMS_IN_PROMPT: int = 50
_MAX_FORMS_IN_PROMPT: int = 20
_MAX_JS_FILES_IN_PROMPT: int = 20


def _build_user_prompt(
    crawler_data: CrawlerResult,
    recon_data: ReconResult | None,
    target_url: str,
) -> str:
    """Build the user prompt with the attack surface formatted for the LLM.

    Each section is truncated to a maximum number of items to stay
    within token limits. The total counts are shown so the LLM knows
    how much data was elided.
    """
    # --- Technologies ---
    if recon_data is not None and recon_data.technologies_detected:
        technologies: str = ", ".join(recon_data.technologies_detected)
    elif recon_data is not None:
        technologies = "(none detected)"
    else:
        technologies = "(recon data unavailable)"

    # --- URLs ---
    url_count_total: int = len(crawler_data.urls)
    urls_shown: list[str] = [str(u) for u in crawler_data.urls[:_MAX_URLS_IN_PROMPT]]
    if url_count_total > _MAX_URLS_IN_PROMPT:
        urls_str: str = "\n".join(f"  {u}" for u in urls_shown)
        urls_str += f"\n  ... ({url_count_total - _MAX_URLS_IN_PROMPT} more URLs elided)"
    elif urls_shown:
        urls_str = "\n".join(f"  {u}" for u in urls_shown)
    else:
        urls_str = "  (none)"

    # --- Parameters ---
    param_count_total: int = len(crawler_data.parameters)
    params_shown = crawler_data.parameters[:_MAX_PARAMS_IN_PROMPT]
    if params_shown:
        param_lines: list[str] = []
        for p in params_shown:
            val_str: str = f', value="{p.value}"' if p.value else ""
            type_str: str = f", type={p.param_type}" if p.param_type else ""
            param_lines.append(
                f"  name={p.name}, location={p.location.value}{type_str}{val_str}"
            )
        params_str = "\n".join(param_lines)
        if param_count_total > _MAX_PARAMS_IN_PROMPT:
            params_str += f"\n  ... ({param_count_total - _MAX_PARAMS_IN_PROMPT} more parameters elided)"
    else:
        params_str = "  (none)"

    # --- Forms ---
    form_count: int = len(crawler_data.forms)
    forms_shown = crawler_data.forms[:_MAX_FORMS_IN_PROMPT]
    if forms_shown:
        form_lines: list[str] = []
        for i, f in enumerate(forms_shown, 1):
            field_names: list[str] = [
                f"{p.name}({p.location.value})" for p in f.fields
            ]
            csrf_str: str = " [has CSRF token]" if f.has_csrf_token else ""
            enctype_str: str = f" [enctype={f.enctype}]" if f.enctype else ""
            form_lines.append(
                f"  Form {i}: action={f.action}, method={f.method.value}{csrf_str}{enctype_str}\n"
                f"    fields: {', '.join(field_names) if field_names else '(none)'}"
            )
        forms_str = "\n".join(form_lines)
    else:
        forms_str = "  (none)"

    # --- JS Files ---
    js_file_count: int = len(crawler_data.js_files)
    js_endpoint_count: int = sum(
        len(j.endpoints_discovered) for j in crawler_data.js_files
    )
    js_files_shown = crawler_data.js_files[:_MAX_JS_FILES_IN_PROMPT]
    if js_files_shown:
        js_lines: list[str] = []
        for j in js_files_shown:
            endpoints_str: str = ", ".join(j.endpoints_discovered) if j.endpoints_discovered else "(none)"
            js_lines.append(f"  {j.url}")
            if j.endpoints_discovered:
                js_lines.append(f"    endpoints: {endpoints_str}")
            if j.interesting_patterns:
                js_lines.append(f"    patterns: {', '.join(j.interesting_patterns)}")
        js_str = "\n".join(js_lines)
    else:
        js_str = "  (none)"

    return _USER_PROMPT_TEMPLATE.format(
        target_url=target_url,
        technologies=technologies,
        url_count_total=url_count_total,
        url_count_shown=min(url_count_total, _MAX_URLS_IN_PROMPT),
        urls=urls_str,
        param_count_total=param_count_total,
        param_count_shown=min(param_count_total, _MAX_PARAMS_IN_PROMPT),
        parameters=params_str,
        form_count=form_count,
        forms=forms_str,
        js_file_count=js_file_count,
        js_endpoint_count=js_endpoint_count,
        js_files=js_str,
    )


# ---------------------------------------------------------------------------
# Internal: defense-in-depth — confidence clamping
# ---------------------------------------------------------------------------


#: Maximum allowed confidence for any hypothesis. Hypotheses are
#: unverified suspicions; 0.7 reflects "strong suspicion" without
#: crossing into "certainty" territory.
_MAX_HYPOTHESIS_CONFIDENCE: float = 0.7

#: Minimum allowed confidence. Values below 0.0 are nonsensical (the
#: Hypothesis schema itself enforces ``ge=0.0``, but we keep this
#: guard for defense-in-depth in case the schema is ever relaxed).
_MIN_HYPOTHESIS_CONFIDENCE: float = 0.0


def _enforce_confidence_range(
    hypotheses: list[Hypothesis], log: Any
) -> list[Hypothesis]:
    """Clamp every hypothesis's confidence to the allowed range.

    The LLM is instructed to use [0.3, 0.7], but it may return 1.0
    (overconfident) or values slightly outside the schema's [0.0, 1.0]
    range. We clamp:
    - Values > 0.7 → 0.7 (overconfidence is the most dangerous case)
    - Values < 0.0 → 0.0 (negative confidence is nonsensical)
    - Values in [0.0, 0.7] → unchanged

    A warning log line is emitted for each clamped hypothesis so the
    operator can see when the LLM was overconfident.

    Parameters
    ----------
    hypotheses:
        The list of hypotheses returned by the LLM.
    log:
        The bound structlog logger (for warning emission).

    Returns
    -------
    list[Hypothesis]
        A new list with clamped confidence values. Hypotheses that
        were already in range are returned as-is (same instance);
        out-of-range hypotheses are replaced with ``model_copy``
        instances.
    """
    result: list[Hypothesis] = []
    clamped_count: int = 0

    for h in hypotheses:
        if h.confidence > _MAX_HYPOTHESIS_CONFIDENCE:
            clamped_count += 1
            log.warning(
                "confidence_clamped",
                hypothesis_id=h.id,
                original_confidence=h.confidence,
                clamped_confidence=_MAX_HYPOTHESIS_CONFIDENCE,
                category=h.category.value,
            )
            h = h.model_copy(update={"confidence": _MAX_HYPOTHESIS_CONFIDENCE})
        elif h.confidence < _MIN_HYPOTHESIS_CONFIDENCE:
            clamped_count += 1
            log.warning(
                "confidence_clamped",
                hypothesis_id=h.id,
                original_confidence=h.confidence,
                clamped_confidence=_MIN_HYPOTHESIS_CONFIDENCE,
                category=h.category.value,
            )
            h = h.model_copy(update={"confidence": _MIN_HYPOTHESIS_CONFIDENCE})
        result.append(h)

    if clamped_count > 0:
        log.info(
            "confidence_clamp_summary",
            total_clamped=clamped_count,
            total_hypotheses=len(hypotheses),
        )

    return result


# ---------------------------------------------------------------------------
# Internal: defense-in-depth — hallucinated parameter removal
# ---------------------------------------------------------------------------


def _remove_hallucinated_parameters(
    hypotheses: list[Hypothesis],
    crawler_data: CrawlerResult,
    log: Any,
) -> list[Hypothesis]:
    """Remove ``target_parameter`` references that don't exist in
    ``crawler_data``.

    The LLM is instructed to only reference parameters that appear in
    the crawler data, but it may hallucinate parameter names. For each
    hypothesis with a non-None ``target_parameter``, we verify that
    the parameter's ``name`` matches at least one parameter observed
    in ``crawler_data`` (either top-level or inside a form).

    If the parameter name is not found, we set ``target_parameter`` to
    ``None`` and log a warning. The hypothesis itself is kept — the
    vulnerability might still be plausible at the URL level — but the
    hallucinated parameter reference is removed.

    Parameters
    ----------
    hypotheses:
        The list of hypotheses (post-confidence-clamping).
    crawler_data:
        The crawler result containing the ground-truth parameter set.
    log:
        The bound structlog logger.

    Returns
    -------
    list[Hypothesis]
        A new list with hallucinated parameter references removed.
        Hypotheses with valid (or None) target_parameter are returned
        as-is; hypotheses with hallucinated parameters are replaced
        with ``model_copy`` instances.
    """
    # Build the set of valid parameter names from crawler_data.
    valid_param_names: set[str] = set()
    for p in crawler_data.parameters:
        if p.name:
            valid_param_names.add(p.name)
    for f in crawler_data.forms:
        for p in f.fields:
            if p.name:
                valid_param_names.add(p.name)

    result: list[Hypothesis] = []
    removed_count: int = 0

    for h in hypotheses:
        if h.target_parameter is None:
            result.append(h)
            continue

        if h.target_parameter.name not in valid_param_names:
            removed_count += 1
            log.warning(
                "hallucinated_parameter_removed",
                hypothesis_id=h.id,
                hallucinated_param_name=h.target_parameter.name,
                category=h.category.value,
                valid_param_count=len(valid_param_names),
            )
            h = h.model_copy(update={"target_parameter": None})

        result.append(h)

    if removed_count > 0:
        log.info(
            "hallucinated_parameter_summary",
            total_removed=removed_count,
            total_hypotheses=len(hypotheses),
        )

    return result


# ---------------------------------------------------------------------------
# Internal: R1 output preprocessing (duplicated from validator.py)
# ---------------------------------------------------------------------------


def _preprocess_llm_output(text: str) -> str:
    """Preprocess DeepSeek R1's output to extract the JSON payload.

    Duplicated from :func:`src.agents.validator._preprocess_llm_output`
    to keep agent modules independent. See the Crawler Parser docstring
    for the rationale on duplication vs. shared utilities.

    1. Removes ``<think>...</think>`` blocks (R1's reasoning output).
    2. Removes markdown code fences (```json ... ```).
    3. Extracts the substring between the first ``{`` and the last
       ``}`` as a last-resort JSON extraction.
    4. Strips whitespace.
    """
    # 1. Remove <think>...</think> blocks
    cleaned: str = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 2. Remove markdown code fences
    cleaned = re.sub(
        r"```(?:json)?\s*(.*?)\s*```",
        r"\1",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 3. Strip stray backticks
    cleaned = cleaned.strip("`").strip()

    # 4. Extract JSON substring if text doesn't start with {
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
    """Return True if the lowercased exception text indicates a rate
    limit or quota-exceeded error from the LLM provider.

    Duplicated from :func:`src.agents.validator._is_rate_limit_error`.
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
    """Best-effort extraction of a ``Retry-After`` value (in seconds).

    Duplicated from :func:`src.agents.validator._extract_retry_after`.
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


__all__ = ["analyze_hypotheses"]
