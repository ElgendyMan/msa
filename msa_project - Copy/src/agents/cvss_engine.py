"""
src/agents/cvss_engine.py
=========================

Node 14 of the 16-node LangGraph framework: the **CVSS Engine**.

This is the final pure-Python node in the framework. It takes a
confirmed :class:`~src.shared.schemas.Finding` (whose
``cvss.vector`` has already been populated by a previous node) and
calculates the deterministic CVSS 3.1 Base Score, Base Severity, and
Vector String.

Why pure Python (no LLM)?
-------------------------
CVSS scoring must be **100% deterministic**. An LLM hallucinating a
score of 9.8 when the vector actually yields 7.5 would undermine the
entire reporting pipeline. The CVSS 3.1 specification defines exact
equations; we implement them verbatim.

CVSS 3.1 Base Score equations
-----------------------------
The official specification (CVSS v3.1 Specification Document, Section
7.1) defines:

    ISCBase = 1 − [(1 − C) × (1 − I) × (1 − A)]

    ISC = Scope(Unchanged)  →  6.42 × ISCBase
    ISC = Scope(Changed)    →  7.52 × (ISCBase − 0.029) − 3.25 × (ISCBase − 0.02)^15

    ESC = 8.22 × AV × AC × PR × UI

    BaseScore = Scope(Unchanged)  →  Roundup(minimum(ISC + ESC, 10))
    BaseScore = Scope(Changed)    →  Roundup(minimum(1.08 × (ISC + ESC), 10))

Where the metric values are:

    Attack Vector (AV):           Network=0.85, Adjacent=0.62, Local=0.55, Physical=0.2
    Attack Complexity (AC):       Low=0.77, High=0.44
    Privileges Required (PR):     None=0.85, Low=0.62/0.68, High=0.27/0.5
                                  (first value: Scope Unchanged; second: Scope Changed)
    User Interaction (UI):        None=0.85, Required=0.62
    Confidentiality (C):          High=0.56, Low=0.22, None=0
    Integrity (I):                High=0.56, Low=0.22, None=0
    Availability (A):             High=0.56, Low=0.22, None=0

The **Roundup** function (Section 7.1.4) performs ceiling-at-2-decimals:

    Roundup(x) = ⌈x × 10⁵⌉ / 10⁵ × 10 / 10
               = (the input rounded up to 2 decimal places, with a
                  specific intermediate scaling to avoid float drift)

The exact algorithm from the spec:

    1. Multiply the input by 100000 (10^5).
    2. If the result is an integer (no fractional part), divide by 100000
       and return.
    3. Otherwise, truncate to integer, add 1, divide by 100000.
    4. Then round to 1 decimal place using the same logic.
    5. The final result has exactly 1 decimal place.

Severity rating scale (Section 7.4)
-----------------------------------
    Base Score 0.0         → INFO (None)
    Base Score 0.1 – 3.9   → LOW
    Base Score 4.0 – 6.9   → MEDIUM
    Base Score 7.0 – 8.9   → HIGH
    Base Score 9.0 – 10.0  → CRITICAL

Vector String format (Section 6)
--------------------------------
    CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H

LangGraph contract
------------------
::

    def calculate_cvss(state: AppState) -> dict:

- Reads: ``state["confirmed_findings"][-1]`` (a
  :class:`~src.shared.schemas.Finding` whose ``cvss.vector`` is
  populated).
- Writes: returns ``{"cvss_results": [<CVSSResult>]}`` — a list of
  exactly one element, ready to be merged into the ``cvss_results``
  channel via the ``operator.add`` reducer.

  The returned :class:`~src.shared.schemas.CVSSResult` has its
  ``finding_id`` set to the finding's ID so the Orchestrator can
  correlate them.

  Note: this node does NOT update the ``Finding`` object in place.
  The Orchestrator is responsible for taking the returned
  :class:`CVSSResult` and calling ``finding.model_copy(update={
  "cvss": cvss_result})`` on the appropriate finding. This keeps the
  CVSS Engine focused on a single responsibility: scoring.

Raises
------
- :class:`PentestFrameworkError` — if ``confirmed_findings`` is
  missing, empty, or the last finding has no ``cvss.vector``.
"""

from __future__ import annotations

import math
from typing import Any

from src.shared.exceptions import PentestFrameworkError
from src.shared.logging import get_logger
from src.shared.schemas import (
    AttackComplexity,
    AttackVector,
    CIA,
    CVSSResult,
    CVSSScope,
    CVSSVector,
    Finding,
    PrivilegesRequired,
    SeverityLevel,
    UserInteraction,
    VulnerabilityCategory,
)
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Metric value tables (CVSS 3.1 Specification, Section 7.1)
# ---------------------------------------------------------------------------

#: Attack Vector (AV) numeric values.
_AV_VALUES: dict[AttackVector, float] = {
    AttackVector.NETWORK: 0.85,
    AttackVector.ADJACENT: 0.62,
    AttackVector.LOCAL: 0.55,
    AttackVector.PHYSICAL: 0.20,
}

#: Attack Complexity (AC) numeric values.
_AC_VALUES: dict[AttackComplexity, float] = {
    AttackComplexity.LOW: 0.77,
    AttackComplexity.HIGH: 0.44,
}

#: User Interaction (UI) numeric values.
_UI_VALUES: dict[UserInteraction, float] = {
    UserInteraction.NONE: 0.85,
    UserInteraction.REQUIRED: 0.62,
}

#: Confidentiality / Integrity / Availability (C/I/A) numeric values.
#: These three metrics share the same value mapping.
_CIA_VALUES: dict[CIA, float] = {
    CIA.HIGH: 0.56,
    CIA.LOW: 0.22,
    CIA.NONE: 0.00,
}

#: Privileges Required (PR) numeric values.
#: PR depends on the Scope metric — different values for Scope Unchanged
#: vs Scope Changed. We store both as a tuple (unchanged, changed).
_PR_VALUES: dict[PrivilegesRequired, tuple[float, float]] = {
    PrivilegesRequired.NONE: (0.85, 0.85),
    PrivilegesRequired.LOW: (0.62, 0.68),
    PrivilegesRequired.HIGH: (0.27, 0.50),
}


# ---------------------------------------------------------------------------
# Metric abbreviations for vector string generation
# ---------------------------------------------------------------------------

#: Maps each AttackVector enum value to its single-letter abbreviation.
_AV_ABBR: dict[AttackVector, str] = {
    AttackVector.NETWORK: "N",
    AttackVector.ADJACENT: "A",
    AttackVector.LOCAL: "L",
    AttackVector.PHYSICAL: "P",
}

#: Maps each AttackComplexity enum value to its abbreviation.
_AC_ABBR: dict[AttackComplexity, str] = {
    AttackComplexity.LOW: "L",
    AttackComplexity.HIGH: "H",
}

#: Maps each PrivilegesRequired enum value to its abbreviation.
_PR_ABBR: dict[PrivilegesRequired, str] = {
    PrivilegesRequired.NONE: "N",
    PrivilegesRequired.LOW: "L",
    PrivilegesRequired.HIGH: "H",
}

#: Maps each UserInteraction enum value to its abbreviation.
_UI_ABBR: dict[UserInteraction, str] = {
    UserInteraction.NONE: "N",
    UserInteraction.REQUIRED: "R",
}

#: Maps each CVSSScope enum value to its abbreviation.
_S_ABBR: dict[CVSSScope, str] = {
    CVSSScope.UNCHANGED: "U",
    CVSSScope.CHANGED: "C",
}

#: Maps each CIA enum value to its abbreviation. Used for C, I, and A.
_CIA_ABBR: dict[CIA, str] = {
    CIA.HIGH: "H",
    CIA.LOW: "L",
    CIA.NONE: "N",
}



#: Maps each VulnerabilityCategory to a reasonable default CVSSVector.
#: These are conservative defaults; real scores should be refined per finding.
_DEFAULT_VECTORS: dict[VulnerabilityCategory, CVSSVector] = {
    VulnerabilityCategory.SQL_INJECTION: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.HIGH, availability=CIA.HIGH,
    ),
    VulnerabilityCategory.SQL_INJECTION_BLIND: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.HIGH,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.HIGH, availability=CIA.LOW,
    ),
    VulnerabilityCategory.SQL_INJECTION_TIME: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.HIGH,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.LOW, availability=CIA.LOW,
    ),
    VulnerabilityCategory.XSS_REFLECTED: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.REQUIRED,
        scope=CVSSScope.CHANGED, confidentiality=CIA.LOW,
        integrity=CIA.LOW, availability=CIA.NONE,
    ),
    VulnerabilityCategory.XSS_STORED: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.LOW, user_interaction=UserInteraction.REQUIRED,
        scope=CVSSScope.CHANGED, confidentiality=CIA.LOW,
        integrity=CIA.LOW, availability=CIA.NONE,
    ),
    VulnerabilityCategory.XSS_DOM: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.REQUIRED,
        scope=CVSSScope.CHANGED, confidentiality=CIA.LOW,
        integrity=CIA.LOW, availability=CIA.NONE,
    ),
    VulnerabilityCategory.SSRF: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.CHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.LOW, availability=CIA.NONE,
    ),
    VulnerabilityCategory.COMMAND_INJECTION: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.HIGH, availability=CIA.HIGH,
    ),
    VulnerabilityCategory.PATH_TRAVERSAL: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.NONE, availability=CIA.NONE,
    ),
    VulnerabilityCategory.IDOR: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.LOW, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.HIGH, availability=CIA.NONE,
    ),
    VulnerabilityCategory.BOLA: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.LOW, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.HIGH, availability=CIA.NONE,
    ),
    VulnerabilityCategory.BFLA: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.LOW, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.HIGH, availability=CIA.NONE,
    ),
    VulnerabilityCategory.SSTI: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.HIGH, availability=CIA.HIGH,
    ),
    VulnerabilityCategory.XXE: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.NONE, availability=CIA.NONE,
    ),
    VulnerabilityCategory.DESERIALIZATION: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.HIGH, availability=CIA.HIGH,
    ),
    VulnerabilityCategory.OPEN_REDIRECT: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.REQUIRED,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.LOW,
        integrity=CIA.LOW, availability=CIA.NONE,
    ),
    VulnerabilityCategory.CSRF: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.REQUIRED,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.NONE,
        integrity=CIA.HIGH, availability=CIA.NONE,
    ),
    VulnerabilityCategory.JWT_NONE_ALG: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.HIGH, availability=CIA.NONE,
    ),
    VulnerabilityCategory.JWT_WEAK_SECRET: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.HIGH,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.HIGH, availability=CIA.NONE,
    ),
    VulnerabilityCategory.AUTH_BYPASS: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.HIGH, availability=CIA.HIGH,
    ),
    VulnerabilityCategory.SENSITIVE_DATA_EXPOSURE: CVSSVector(
        attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
        scope=CVSSScope.UNCHANGED, confidentiality=CIA.HIGH,
        integrity=CIA.NONE, availability=CIA.NONE,
    ),
}

#: Safe generic fallback vector for unknown categories.
_GENERIC_VECTOR: CVSSVector = CVSSVector(
    attack_vector=AttackVector.NETWORK, attack_complexity=AttackComplexity.LOW,
    privileges_required=PrivilegesRequired.NONE, user_interaction=UserInteraction.NONE,
    scope=CVSSScope.UNCHANGED, confidentiality=CIA.LOW,
    integrity=CIA.LOW, availability=CIA.LOW,
)


def _infer_vector(finding: Finding) -> CVSSVector:
    """Return a CVSSVector for a finding.

    Priority order:
    1. If the finding already has ``cvss.vector`` set (pre-existing), use it.
    2. If the category maps to a default vector, use that.
    3. Fall back to the generic LOW/LOW/LOW vector.
    """
    if finding.cvss is not None and finding.cvss.vector is not None:
        return finding.cvss.vector
    return _DEFAULT_VECTORS.get(finding.category, _GENERIC_VECTOR)


def calculate_cvss(state: AppState) -> dict[str, Any]:
    """LangGraph Node 14: calculate CVSS 3.1 Base Score for confirmed findings
    that have not yet been scored.

    This node now **infers** the CVSSVector from the vulnerability category
    when no vector has been pre-populated on the finding. This fixes the
    chicken-and-egg problem where the engine previously raised an error
    because no other node ever set ``finding.cvss.vector``.

    Parameters
    ----------
    state:
        The current :class:`~src.shared.state.AppState`. Must contain at
        least one ``confirmed_findings`` entry whose ``cvss`` field is
        ``None`` (i.e., not yet scored).

    Returns
    -------
    dict
        ``{"cvss_results": [<CVSSResult>, ...]}`` — one result per
        unscored finding processed in this invocation.
    """
    log = get_logger("cvss_engine")

    # ---------------------------------------------------------------
    # 1. Resolve confirmed findings that still need scoring.
    # ---------------------------------------------------------------
    confirmed_findings: list[Finding] | None = state.get("confirmed_findings")
    if not confirmed_findings:
        raise PentestFrameworkError(
            "CVSS Engine cannot run: state['confirmed_findings'] is missing "
            "or empty. The Orchestrator must route a confirmed finding here "
            "only after the Validator has confirmed it as TRUE_POSITIVE.",
            details={"available_keys": list(state.keys())},
        )

    # Only score findings that don't already have a base_score.
    unscored: list[Finding] = [
        f for f in confirmed_findings
        if f.cvss is None or f.cvss.base_score == 0.0
    ]
    if not unscored:
        log.info("cvss_engine_skipped_all_findings_already_scored")
        return {"cvss_results": []}

    results: list[CVSSResult] = []
    for finding in unscored:
        log_f = log.bind(finding_id=finding.id, category=finding.category.value)

        vector: CVSSVector = _infer_vector(finding)

        log_f.info(
            "cvss_calculation_started",
            av=vector.attack_vector.value,
            ac=vector.attack_complexity.value,
            pr=vector.privileges_required.value,
            ui=vector.user_interaction.value,
            scope=vector.scope.value,
            c=vector.confidentiality.value,
            i=vector.integrity.value,
            a=vector.availability.value,
            inferred=(finding.cvss is None),
        )

        base_score: float = _calculate_base_score(vector)
        base_severity: SeverityLevel = _severity_rating(base_score)
        vector_string: str = _generate_vector_string(vector)

        cvss_result: CVSSResult = CVSSResult(
            finding_id=finding.id,
            vector=vector,
            base_score=base_score,
            base_severity=base_severity,
            vector_string=vector_string,
        )

        log_f.info(
            "cvss_calculation_complete",
            base_score=base_score,
            base_severity=base_severity.value,
            vector_string=vector_string,
        )

        results.append(cvss_result)

    return {"cvss_results": results}


# ---------------------------------------------------------------------------
# Internal: CVSS 3.1 Base Score calculation
# ---------------------------------------------------------------------------


def _calculate_base_score(vector: CVSSVector) -> float:
    """Calculate the CVSS 3.1 Base Score.

    Implements the exact equations from the CVSS v3.1 Specification
    Document, Section 7.1:

    1. Compute the Impact Subscore Base (ISCBase).
    2. Compute the Impact Subscore (ISC) — depends on Scope.
    3. Compute the Exploitability Subscore (ESC).
    4. Compute the Base Score — depends on Scope.
    5. Apply the Roundup function.
    """
    # --- Step 1: ISCBase ---
    c: float = _CIA_VALUES[vector.confidentiality]
    i: float = _CIA_VALUES[vector.integrity]
    a: float = _CIA_VALUES[vector.availability]

    isc_base: float = 1.0 - ((1.0 - c) * (1.0 - i) * (1.0 - a))

    # --- Step 2: ISC (depends on Scope) ---
    # Select the correct PR value based on Scope.
    pr_unchanged, pr_changed = _PR_VALUES[vector.privileges_required]
    pr_value: float = (
        pr_unchanged if vector.scope == CVSSScope.UNCHANGED else pr_changed
    )

    if vector.scope == CVSSScope.UNCHANGED:
        isc: float = 6.42 * isc_base
    else:
        # Scope Changed formula has a correction term.
        # The exponent 15 is applied to (ISCBase − 0.02)^15.
        # Guard against negative base (ISCBase is always >= 0, so
        # ISCBase - 0.02 could be slightly negative when ISCBase < 0.02,
        # which happens when C=I=A=0). In that case the power would
        # produce a complex number; we use abs() to avoid that, matching
        # the reference implementation's behavior.
        correction_base: float = isc_base - 0.02
        if correction_base < 0:
            # When ISCBase is 0 (all CIA = None), the impact is 0.
            # The formula would produce a negative ISC, which we clamp
            # to 0 to avoid nonsensical negative scores.
            isc = 0.0
        else:
            isc = 7.52 * (isc_base - 0.029) - 3.25 * (correction_base ** 15)

    # --- Step 3: ESC (Exploitability Subscore) ---
    av: float = _AV_VALUES[vector.attack_vector]
    ac: float = _AC_VALUES[vector.attack_complexity]
    ui: float = _UI_VALUES[vector.user_interaction]

    esc: float = 8.22 * av * ac * pr_value * ui

    # --- Step 4: Base Score (depends on Scope) ---
    if vector.scope == CVSSScope.UNCHANGED:
        raw_score: float = min(isc + esc, 10.0)
    else:
        raw_score = min(1.08 * (isc + esc), 10.0)

    # --- Step 5: Roundup ---
    # If ISC is 0 (no impact), the base score is 0.0 regardless of
    # exploitability. This matches the CVSS spec: "If the Impact Score
    # is 0, then the Base Score is 0."
    if isc <= 0.0:
        return 0.0

    return _roundup(raw_score)


# ---------------------------------------------------------------------------
# Internal: Roundup function (CVSS 3.1 Specification, Section 7.1.4)
# ---------------------------------------------------------------------------


def _roundup(x: float) -> float:
    """Apply the CVSS 3.1 Roundup function.

    The Roundup function rounds the input up to one decimal place using
    a specific intermediate scaling to avoid floating-point drift.

    Algorithm (from the spec):

    1. Multiply the input by 100,000 (10^5).
    2. Check if the result, when divided by 10,000 (10^4), is an integer.
       - If yes: the input already has at most 1 decimal place. Return
         input / 10^4 × 10^4 / 10^5 (i.e., the input unchanged but
         normalized).
       - If no: truncate to integer, add 1, then divide by 10^5.

    The spec's exact formulation (in pseudocode):

        function Roundup(input):
            int_input = round_to_nearest_integer(input * 100000)
            if (int_input % 10000) == 0:
                return int_input / 100000.0
            else:
                return (floor(int_input / 10000) + 1) / 10.0

    This is equivalent to "round up to 1 decimal place" but with a
    specific floating-point-safe implementation.

    Examples
    --------
    >>> _roundup(4.02)
    4.1
    >>> _roundup(4.00)
    4.0
    >>> _roundup(4.001)
    4.1
    >>> _roundup(9.99)
    10.0
    """
    # Step 1: Scale to 5 decimal places and round to nearest integer.
    # We use round() here to handle floating-point representation issues
    # (e.g., 4.02 might be stored as 4.0199999999...). The spec uses
    # round_to_nearest_integer; Python's round() with no ndigits does
    # banker's rounding, so we use math.floor(x + 0.5) for traditional
    # rounding.
    int_input: int = int(math.floor(x * 100000.0 + 0.5))

    # Step 2: Check if the value already has at most 1 decimal place.
    if int_input % 10000 == 0:
        return int_input / 100000.0
    else:
        return (math.floor(int_input / 10000) + 1) / 10.0


# ---------------------------------------------------------------------------
# Internal: Severity rating (CVSS 3.1 Specification, Section 7.4)
# ---------------------------------------------------------------------------


def _severity_rating(base_score: float) -> SeverityLevel:
    """Map a CVSS 3.1 Base Score to its qualitative severity rating.

    Rating scale (from the spec):

    ============  ===========
    Base Score    Severity
    ============  ===========
    0.0           None (INFO)
    0.1 – 3.9     Low
    4.0 – 6.9     Medium
    7.0 – 8.9     High
    9.0 – 10.0    Critical
    ============  ===========
    """
    if base_score <= 0.0:
        return SeverityLevel.INFO
    elif base_score <= 3.9:
        return SeverityLevel.LOW
    elif base_score <= 6.9:
        return SeverityLevel.MEDIUM
    elif base_score <= 8.9:
        return SeverityLevel.HIGH
    else:  # 9.0 – 10.0
        return SeverityLevel.CRITICAL


# ---------------------------------------------------------------------------
# Internal: Vector String generation (CVSS 3.1 Specification, Section 6)
# ---------------------------------------------------------------------------


def _generate_vector_string(vector: CVSSVector) -> str:
    """Generate the canonical CVSS 3.1 Vector String.

    Format::

        CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H

    The order of metrics in the vector string is fixed by the spec:
    AV, AC, PR, UI, S, C, I, A.
    """
    parts: list[str] = [
        "CVSS:3.1",
        f"AV:{_AV_ABBR[vector.attack_vector]}",
        f"AC:{_AC_ABBR[vector.attack_complexity]}",
        f"PR:{_PR_ABBR[vector.privileges_required]}",
        f"UI:{_UI_ABBR[vector.user_interaction]}",
        f"S:{_S_ABBR[vector.scope]}",
        f"C:{_CIA_ABBR[vector.confidentiality]}",
        f"I:{_CIA_ABBR[vector.integrity]}",
        f"A:{_CIA_ABBR[vector.availability]}",
    ]
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = ["calculate_cvss"]
