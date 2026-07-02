"""
src/agents/crawler_executor.py
==============================

**Crawler Executor** — crawls the target URL using Playwright and
writes a request log + HTML dump to ``state["raw_crawler_output"]``
for the downstream ``crawler_parser`` node.

Output format
-------------
The output is a labelled plain-text blob containing two sections:

1. **REQUEST LOG** — one line per request intercepted by Playwright:
   ``METHOD URL STATUS``  (e.g. ``GET https://example.com/ 200``)

2. **HTML DUMP** — the final rendered HTML of the top-level page
   (``page.content()`` after JS execution).

``crawler_parser`` explicitly handles both formats in its system
prompt, so no further encoding is required.

Tool choice
-----------
Playwright is used because ``crawler_parser`` is written for Playwright
output and handles JS-rendered pages — httpx/requests would miss
dynamically injected links and forms.

Install on Linux
----------------
.. code-block:: bash

    pip install playwright          # Python package (usually already installed)
    playwright install chromium     # downloads the browser binary (~170 MB)
    # Optional system deps on headless servers:
    playwright install-deps chromium

LangGraph contract
------------------
::

    async def run_crawler(state: AppState) -> dict[str, Any]:

- Reads:  ``state["target"]``      (:class:`~src.shared.schemas.Target`)
          ``state["scope"]``       (:class:`~src.shared.schemas.ScopeConfig`)
          ``state["session_id"]``  (str)
- Writes: ``{"raw_crawler_output": <str>}``

Never raises — returns ``""`` on any failure so ``crawler_parser``
emits a clean error instead of crashing the graph.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlparse

from src.shared.config import settings
from src.shared.logging import get_logger
from src.shared.state import AppState

# Maximum number of URLs the crawler will visit per session.
# Keeps runtime predictable; recon is not exhaustive enumeration.
_MAX_PAGES: int = 30

# Maximum bytes captured from page.content() per page.
_MAX_HTML_BYTES: int = 500_000


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------


async def run_crawler(state: AppState) -> dict[str, Any]:
    """LangGraph node: crawl target with Playwright, populate raw_crawler_output."""
    log = get_logger("crawler_executor")
    session_id: str = state.get("session_id") or "_default"

    target = state.get("target")
    if target is None:
        log.warning("crawler_executor_skipped", reason="target is None",
                    session_id=session_id)
        return {"raw_crawler_output": ""}

    start_url: str = str(target.url)
    scope = state.get("scope")
    allowed_domains: set[str] = _build_allowed_domains(start_url, scope)
    timeout_ms: int = settings.PLAYWRIGHT_NAV_TIMEOUT_MS
    max_depth: int = settings.PLAYWRIGHT_CRAWL_DEPTH
    session_timeout: float = settings.EXECUTION_TIMEOUT_SECONDS * max(2, max_depth)

    log.info(
        "crawler_executor_starting",
        session_id=session_id,
        start_url=start_url,
        allowed_domains=list(allowed_domains),
        max_pages=_MAX_PAGES,
        max_depth=max_depth,
        timeout_ms=timeout_ms,
    )

    try:
        raw_output = await asyncio.wait_for(
            _crawl(start_url, allowed_domains, timeout_ms, max_depth, log),
            timeout=session_timeout,
        )
    except asyncio.TimeoutError:
        log.warning(
            "crawler_executor_session_timeout",
            session_id=session_id,
            session_timeout_seconds=session_timeout,
        )
        raw_output = ""
    except Exception as exc:  # noqa: BLE001
        log.error(
            "crawler_executor_unexpected_error",
            session_id=session_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raw_output = ""

    log.info(
        "crawler_executor_complete",
        session_id=session_id,
        output_bytes=len(raw_output),
    )
    return {"raw_crawler_output": raw_output}


# ---------------------------------------------------------------------------
# Internal: Playwright crawl
# ---------------------------------------------------------------------------


async def _crawl(
    start_url: str,
    allowed_domains: set[str],
    timeout_ms: int,
    max_depth: int,
    log: Any,
) -> str:
    """BFS crawl up to ``_MAX_PAGES`` pages within ``allowed_domains``."""
    try:
        from playwright.async_api import async_playwright, BrowserContext, Page
    except ImportError:
        log.warning(
            "crawler_tool_not_found",
            tool="playwright",
            hint="Run: pip install playwright && playwright install chromium",
        )
        return ""

    request_log: list[str] = []
    html_sections: list[str] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context: BrowserContext = await browser.new_context(
            ignore_https_errors=True,
            java_script_enabled=True,
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )

        # Intercept all requests to build the request log.
        async def _on_request(req: Any) -> None:
            request_log.append(f"{req.method} {req.url}")

        async def _on_response(resp: Any) -> None:
            # Find the matching request entry and append status code.
            target_line = f"{resp.request.method} {resp.url}"
            for i in range(len(request_log) - 1, -1, -1):
                if request_log[i] == target_line:
                    request_log[i] = f"{target_line} {resp.status}"
                    break

        context.on("request", _on_request)
        context.on("response", _on_response)

        visited: set[str] = set()
        # BFS queue: (url, depth)
        queue: list[tuple[str, int]] = [(start_url, 0)]

        while queue and len(visited) < _MAX_PAGES:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            if not _is_allowed(url, allowed_domains):
                continue

            visited.add(url)

            page: Page = await context.new_page()
            try:
                await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                html: str = await page.content()

                # Truncate large pages.
                if len(html) > _MAX_HTML_BYTES:
                    html = html[:_MAX_HTML_BYTES] + "\n<!-- TRUNCATED -->"

                html_sections.append(f"<!-- PAGE: {url} -->\n{html}")

                if depth < max_depth:
                    links = await page.eval_on_selector_all(
                        "a[href]",
                        "els => els.map(e => e.href)",
                    )
                    for link in links:
                        link_str: str = str(link).split("#")[0]  # strip fragments
                        if link_str and link_str not in visited:
                            queue.append((link_str, depth + 1))

                log.debug("crawler_page_visited", url=url, depth=depth)

            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "crawler_page_error",
                    url=url,
                    error=str(exc)[:200],
                    error_type=type(exc).__name__,
                )
            finally:
                await page.close()

        await context.close()
        await browser.close()

    return _format_output(start_url, request_log, html_sections)


# ---------------------------------------------------------------------------
# Internal: helpers
# ---------------------------------------------------------------------------


def _build_allowed_domains(start_url: str, scope: Any) -> set[str]:
    """Return the set of domains the crawler is allowed to visit."""
    domains: set[str] = set()
    parsed = urlparse(start_url)
    if parsed.hostname:
        domains.add(parsed.hostname)
    if scope is not None:
        for d in (scope.in_scope_domains or []):
            if d:
                domains.add(d)
    return domains


def _is_allowed(url: str, allowed_domains: set[str]) -> bool:
    """Return True iff the URL's host is within allowed_domains."""
    if not url.startswith(("http://", "https://")):
        return False
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return any(host == d or host.endswith(f".{d}") for d in allowed_domains)


def _format_output(
    start_url: str,
    request_log: list[str],
    html_sections: list[str],
) -> str:
    """Combine request log + HTML dumps into the labelled blob
    that ``crawler_parser`` expects."""
    parts: list[str] = [f"# CRAWLER TARGET: {start_url}"]

    if request_log:
        parts.append("\n## REQUEST LOG\n" + "\n".join(request_log))
    else:
        parts.append("\n## REQUEST LOG\n(no requests captured)")

    if html_sections:
        parts.append("\n## HTML DUMPS\n" + "\n\n".join(html_sections))
    else:
        parts.append("\n## HTML DUMPS\n(no pages captured)")

    return "\n".join(parts)


__all__ = ["run_crawler"]