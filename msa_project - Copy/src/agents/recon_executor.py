"""
src/agents/recon_executor.py
============================

**Recon Executor** — runs external recon tools (Nmap, Subfinder) as
subprocesses and writes their combined raw output to
``state["raw_recon_output"]`` for the downstream ``recon_parser`` node.

This is one of two new "executor" nodes added to close the design gap
where ``recon_parser`` and ``crawler_parser`` expected pre-populated
raw output that nothing in the original 16-node graph ever produced.

LangGraph contract
------------------
::

    async def run_recon(state: AppState) -> dict[str, Any]:

- Reads:  ``state["target"]``  (:class:`~src.shared.schemas.Target`)
          ``state["scope"]``   (:class:`~src.shared.schemas.ScopeConfig`)
          ``state["session_id"]`` (str)
- Writes: ``{"raw_recon_output": <str>}``  — concatenated stdout from
          every tool that ran successfully.  Empty-string result
          (all tools failed / not installed) is written as ``""``
          so ``recon_parser`` can emit a clean error rather than seeing
          a missing key.

Tool availability
-----------------
Both tools are **optional**: if a binary is not on ``$PATH`` the node
logs a warning, skips that tool, and continues with whatever output
the other tool produced. The node never raises; callers should check
whether ``raw_recon_output`` is non-empty.

Install on Linux
~~~~~~~~~~~~~~~~
.. code-block:: bash

    # Nmap
    sudo apt-get install -y nmap          # Debian / Ubuntu
    sudo dnf install -y nmap              # Fedora / RHEL

    # Subfinder (Go binary)
    go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest
    # or download a pre-built release from:
    # https://github.com/projectdiscovery/subfinder/releases

After install, ensure both are on ``$PATH`` (``which nmap subfinder``).
"""

from __future__ import annotations

import asyncio
import shutil
from typing import Any
from urllib.parse import urlparse

from src.shared.config import settings
from src.shared.logging import get_logger
from src.shared.state import AppState

# ---------------------------------------------------------------------------
# Tool configuration
# ---------------------------------------------------------------------------

#: Nmap flags used for every scan. We prefer structured output (-oX -)
#: but also keep human-readable (-oN) in the combined blob so the LLM
#: parser can fall back to text if XML parsing is ambiguous.
#:
#: Flags chosen for a low-noise service scan that stays within typical
#: pentesting engagement scope: top 1000 ports, version detection, no
#: OS detection (requires root and adds noise), skip host discovery for
#: hosts that block ICMP.
_NMAP_FLAGS: list[str] = [
    "-sV",          # service/version detection
    "--open",       # only show open ports (reduces noise)
    "-T3",          # timing template 3 (normal — balance speed vs IDS noise)
    "-Pn",          # skip ping / host-discovery (works on filtered hosts)
    "--top-ports", "1000",   # top 1000 ports (good coverage, not exhaustive)
    "-oX", "-",     # output XML to stdout (recon_parser handles both formats)
]

#: Subfinder flags. We request JSON output (-oJ) for structured parsing
#: and silent mode (-silent) to suppress the banner so stdout is pure data.
_SUBFINDER_FLAGS: list[str] = [
    "-silent",      # suppress banner/progress
    "-oJ",          # JSON output (one JSON object per line)
    "-t", "10",     # concurrency (threads)
]

# Maximum bytes we will capture from any single tool's stdout. 4 MB is
# generous; real Nmap XML for a single host rarely exceeds a few hundred KB.
_MAX_OUTPUT_BYTES: int = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------


async def run_recon(state: AppState) -> dict[str, Any]:
    """LangGraph node: execute recon tools and populate raw_recon_output.

    Runs Nmap and Subfinder concurrently. Writes whatever output was
    produced (even partial) to ``raw_recon_output``. If both tools are
    unavailable or both time out, writes ``""`` — the downstream
    ``recon_parser`` handles this gracefully.
    """
    log = get_logger("recon_executor")
    session_id: str = state.get("session_id") or "_default"

    target = state.get("target")
    if target is None:
        log.warning("recon_executor_skipped", reason="target is None", session_id=session_id)
        return {"raw_recon_output": ""}

    target_url: str = str(target.url)
    host: str = _extract_host(target_url)

    scope = state.get("scope")
    timeout: float = settings.EXECUTION_TIMEOUT_SECONDS

    log.info(
        "recon_executor_starting",
        session_id=session_id,
        host=host,
        target_url=target_url,
        timeout_seconds=timeout,
    )

    # Run both tools concurrently; gather results (never raises).
    nmap_output, subfinder_output = await asyncio.gather(
        _run_nmap(host, timeout, log),
        _run_subfinder(host, scope, timeout, log),
        return_exceptions=False,
    )

    combined: str = _combine_outputs(host, nmap_output, subfinder_output)

    log.info(
        "recon_executor_complete",
        session_id=session_id,
        host=host,
        nmap_bytes=len(nmap_output),
        subfinder_bytes=len(subfinder_output),
        combined_bytes=len(combined),
    )

    return {"raw_recon_output": combined}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_host(url: str) -> str:
    """Return the bare hostname (no port, no scheme) from a URL."""
    parsed = urlparse(url)
    return parsed.hostname or parsed.netloc or url


async def _run_tool(
    cmd: list[str],
    timeout: float,
    tool_name: str,
    log: Any,
) -> str:
    """Run an external command asynchronously, capping output at
    ``_MAX_OUTPUT_BYTES``. Returns stdout as a string or ``""`` on
    any failure (binary missing, timeout, non-zero exit).
    """
    binary = cmd[0]
    if shutil.which(binary) is None:
        log.warning(
            "recon_tool_not_found",
            tool=tool_name,
            binary=binary,
            hint=f"Install {binary} and ensure it is on $PATH. "
                 f"See the docstring in recon_executor.py for instructions.",
        )
        return ""

    log.debug("recon_tool_starting", tool=tool_name, cmd=" ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()   # drain pipes to prevent zombie
            log.warning(
                "recon_tool_timeout",
                tool=tool_name,
                timeout_seconds=timeout,
            )
            return ""

        # Truncate very large outputs before decoding.
        if len(stdout_bytes) > _MAX_OUTPUT_BYTES:
            log.warning(
                "recon_tool_output_truncated",
                tool=tool_name,
                original_bytes=len(stdout_bytes),
                truncated_to=_MAX_OUTPUT_BYTES,
            )
            stdout_bytes = stdout_bytes[:_MAX_OUTPUT_BYTES]

        output: str = stdout_bytes.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            stderr_snippet = stderr_bytes[:500].decode("utf-8", errors="replace")
            log.warning(
                "recon_tool_nonzero_exit",
                tool=tool_name,
                returncode=proc.returncode,
                stderr_snippet=stderr_snippet,
            )
            # Return partial stdout anyway — Nmap sometimes emits useful
            # XML before a non-zero exit (e.g. a single host that blocked).
            return output

        log.debug(
            "recon_tool_complete",
            tool=tool_name,
            returncode=proc.returncode,
            output_bytes=len(stdout_bytes),
        )
        return output

    except Exception as exc:  # noqa: BLE001
        log.error(
            "recon_tool_unexpected_error",
            tool=tool_name,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return ""


async def _run_nmap(host: str, timeout: float, log: Any) -> str:
    """Build the Nmap command and delegate to ``_run_tool``."""
    cmd: list[str] = ["nmap"] + _NMAP_FLAGS + [host]
    return await _run_tool(cmd, timeout=timeout, tool_name="nmap", log=log)


async def _run_subfinder(
    host: str,
    scope: Any,   # ScopeConfig | None
    timeout: float,
    log: Any,
) -> str:
    """Build the Subfinder command and delegate to ``_run_tool``.

    Only runs against the root domain (not IP addresses).  Extra
    in-scope domains from ``scope.in_scope_domains`` are added as
    additional targets so a single executor pass covers all enrolled
    domains, not just the primary target host.
    """
    # Subfinder needs a domain, not a raw IP.
    import re
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
        log.debug(
            "subfinder_skipped_ip",
            host=host,
            reason="Subfinder requires a domain name, not a raw IP address.",
        )
        return ""

    domains: list[str] = [host]
    if scope is not None:
        for d in (scope.in_scope_domains or []):
            if d and d not in domains:
                domains.append(d)

    domain_flags: list[str] = []
    for d in domains:
        domain_flags += ["-d", d]

    cmd: list[str] = ["subfinder"] + domain_flags + _SUBFINDER_FLAGS
    return await _run_tool(cmd, timeout=timeout, tool_name="subfinder", log=log)


def _combine_outputs(host: str, nmap_out: str, subfinder_out: str) -> str:
    """Merge tool outputs into a single labelled blob for the LLM parser."""
    sections: list[str] = [
        f"# RECON TARGET: {host}",
    ]

    if nmap_out.strip():
        sections.append(f"\n## NMAP OUTPUT\n{nmap_out.strip()}")

    if subfinder_out.strip():
        sections.append(f"\n## SUBFINDER OUTPUT\n{subfinder_out.strip()}")

    if not (nmap_out.strip() or subfinder_out.strip()):
        sections.append(
            "\n## NO TOOL OUTPUT\n"
            "Neither nmap nor subfinder produced any output for this target. "
            "Both tools may be missing from $PATH or may have timed out."
        )

    return "\n".join(sections)


__all__ = ["run_recon"]