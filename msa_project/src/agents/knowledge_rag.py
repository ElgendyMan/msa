"""
src/agents/knowledge_rag.py
===========================

Node 13 of the 16-node LangGraph framework: the **Knowledge RAG** —
the "Librarian" that retrieves relevant penetration-testing methodologies
from a Qdrant vector database.

This node uses a three-stage RAG (Retrieval-Augmented Generation)
pipeline:

1. **Query generation**: Gemini Flash examines the detected technologies
   and parameters from the crawler/recon data and generates a highly
   specific search query (e.g. "GraphQL introspection bypass",
   "PHP deserialization magic methods").

2. **Embedding + retrieval**: The query is embedded using BGE-M3
   (``BAAI/bge-m3``) via :class:`HuggingFaceBgeEmbeddings`, and the
   resulting vector is used to search a Qdrant collection containing
   pentest methodology documents, bug-bounty writeups, and OWASP
   guides.

3. **Filtering + mapping**: Results below the similarity threshold are
   discarded; surviving results are mapped into
   :class:`~src.shared.schemas.RAGDocument` objects and packaged into
   a :class:`~src.shared.schemas.RAGContext`.

Graceful degradation
--------------------
The Knowledge RAG node is **advisory** — its absence should never
block the pipeline. If ANY of the following fail, the node logs a
warning and returns ``{"rag_context": None}``:

- ``qdrant-client`` not installed.
- ``langchain-community`` (or ``langchain-huggingface``) not installed.
- Qdrant server unreachable at ``settings.QDRANT_URL``.
- Collection ``settings.QDRANT_COLLECTION`` does not exist.
- BGE-M3 model weights cannot be loaded.
- Gemini Flash query-generation LLM call fails.
- Qdrant search returns zero results (this is NOT an error — the
  context is simply empty).

The Orchestrator and downstream nodes (Hypothesis Analyzer, Payload
Generator, etc.) check for ``rag_context is None`` and proceed without
methodology references. This is the framework's standard pattern for
optional infrastructure.

LangGraph contract
------------------
::

    async def retrieve_knowledge(state: AppState) -> dict:

- Reads: ``state["crawler_data"]`` (optional),
         ``state["recon_data"]`` (optional),
         ``state["target"]`` (optional, for context).
- Writes: returns ``{"rag_context": <RAGContext | None>}``.

Raises
------
This node does NOT raise exceptions to the caller. All failures
(including LLM errors) are caught and result in
``{"rag_context": None}``. This is by design — the RAG node is
advisory and must never crash the pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.shared.config import settings
from src.shared.llm import gemini_flash
from src.shared.logging import get_logger
from src.shared.schemas import (
    CrawlerResult,
    RAGContext,
    RAGDocument,
    ReconResult,
    Target,
    VulnerabilityCategory,
)
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum characters of context (technologies, parameters) to include
#: in the LLM query-generation prompt. Truncating keeps the prompt
#: within Flash's context window.
_MAX_CONTEXT_CHARS: int = 2000

#: Module-level cache for the BGE-M3 embedder. Loading the model is
#: expensive (downloads ~2GB of weights on first use), so we cache it
#: across calls. ``None`` means "not yet loaded or load failed".
_embedder: Any = None
_embedder_load_attempted: bool = False

#: Module-level cache for the Qdrant client. Creating a new client per
#: call would re-establish the HTTP connection each time.
_qdrant_client: Any = None
_qdrant_connect_attempted: bool = False


# ---------------------------------------------------------------------------
# Private wrapper model for Gemini Flash structured output
# ---------------------------------------------------------------------------


class _SearchQuery(BaseModel):
    """Private wrapper model for the LLM-generated search query.

    Gemini Flash's ``with_structured_output()`` enforces this schema
    at the API level. The LLM returns a JSON object with a single
    ``query`` field containing the search string.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        description="A single, highly specific search query to find "
        "relevant penetration testing methodologies or bug bounty "
        "writeups. Examples: 'GraphQL introspection bypass', "
        "'PHP deserialization magic methods', 'nginx alias traversal'.",
    )


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT: str = """\
You are a Penetration Testing Research Librarian. Your job is to \
generate a single, highly specific search query that will be used to \
search a vector database of penetration testing methodologies, bug \
bounty writeups, and OWASP guides.

QUERY GENERATION RULES:
1. Base the query on the DETECTED TECHNOLOGIES and PARAMETERS from \
the current engagement. The more specific, the better.
2. Prefer technique-specific queries over generic ones. \
"GraphQL introspection bypass" is better than "GraphQL vulnerability". \
"PHP deserialization magic methods __wakeup" is better than "PHP deserialization".
3. If a specific vulnerability category is being investigated, include \
it in the query (e.g. "SQL injection UNION based error Oracle").
4. If WAF signatures were detected, include evasion context (e.g. \
"Cloudflare WAF bypass SQL injection").
5. Keep the query to 3-10 words. It should be a search string, not a \
sentence.
6. Do NOT include the target URL or hostname in the query.
7. If no technologies or parameters are available, generate a generic \
query based on the attack surface (e.g. "web application \
penetration testing methodology").

Output ONLY the JSON object matching the _SearchQuery schema. The \
"query" field should contain the search string.
"""


_USER_PROMPT_TEMPLATE: str = """\
Generate a search query based on the current engagement context.

=== TARGET ===
URL: {target_url}

=== DETECTED TECHNOLOGIES ===
{technologies}

=== PARAMETERS DISCOVERED ===
{parameters}

=== URLS DISCOVERED (showing first 10) ===
{urls}

=== WAF SIGNATURE ===
{waf_signature}

=== INSTRUCTIONS ===
Generate a single, highly specific search query (3-10 words) to find \
relevant pentest methodologies in the vector database.

Return the _SearchQuery JSON object now.
"""


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


async def retrieve_knowledge(state: AppState) -> dict[str, Any]:
    """LangGraph Node 13: retrieve relevant pentest methodologies from
    the Qdrant vector database.

    Parameters
    ----------
    state:
        The current :class:`~src.shared.state.AppState`. Reads
        ``crawler_data``, ``recon_data``, and ``target`` for context.
        All are optional — the node works with whatever is available.

    Returns
    -------
    dict
        ``{"rag_context": <RAGContext | None>}`` — if retrieval
        succeeds, a :class:`RAGContext` with the query, documents,
        and scores. If any failure occurs (Qdrant unreachable,
        embeddings unavailable, LLM error), ``None``.
    """
    log = get_logger("knowledge_rag")

    target: Target | None = state.get("target")
    target_url: str = str(target.url) if target is not None else "(unknown)"
    log = log.bind(target_url=target_url)

    log.info("rag_retrieval_started")

    # ---------------------------------------------------------------
    # 1. Generate search query via Gemini Flash.
    # ---------------------------------------------------------------
    search_query: str | None = None

    try:
        search_query = await _generate_search_query(state, log)
    except Exception as exc:
        log.warning(
            "rag_query_generation_failed",
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
        )
        # Without a query, we can't search. Graceful degradation.
        return {"rag_context": None}

    if not search_query or not search_query.strip():
        log.warning("rag_query_empty")
        return {"rag_context": None}

    log.info("rag_query_generated", search_query=search_query)

    # ---------------------------------------------------------------
    # 2. Embed the query and search Qdrant.
    # ---------------------------------------------------------------
    try:
        query_vector: list[float] = _embed_query(search_query, log)
        raw_results: list[Any] = _search_qdrant(query_vector, log)
    except Exception as exc:
        log.warning(
            "rag_retrieval_failed",
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
            search_query=search_query,
        )
        return {"rag_context": None}

    # ---------------------------------------------------------------
    # 3. Filter by similarity threshold and map to RAGDocument.
    # ---------------------------------------------------------------
    documents: list[RAGDocument] = []
    scores: list[float] = []

    for result in raw_results:
        score: float = float(result.score) if hasattr(result, "score") else 0.0

        # Filter by similarity threshold.
        if score < settings.RAG_SIMILARITY_THRESHOLD:
            continue

        payload: dict[str, Any] = (
            result.payload if hasattr(result, "payload") else {}
        )

        doc: RAGDocument | None = _map_qdrant_payload(payload, log)
        if doc is not None:
            documents.append(doc)
            scores.append(score)

    # ---------------------------------------------------------------
    # 4. Build and return the RAGContext.
    # ---------------------------------------------------------------
    rag_context: RAGContext = RAGContext(
        query=search_query,
        retrieved_documents=documents,
        similarity_scores=scores,
        embedder_model=settings.EMBEDDER_MODEL_NAME,
    )

    log.info(
        "rag_retrieval_complete",
        search_query=search_query,
        documents_retrieved=len(documents),
        raw_results_count=len(raw_results),
        filtered_out=len(raw_results) - len(documents),
        top_score=max(scores) if scores else 0.0,
        avg_score=(
            round(sum(scores) / len(scores), 4) if scores else 0.0
        ),
    )

    return {"rag_context": rag_context}


# ---------------------------------------------------------------------------
# Internal: LLM query generation
# ---------------------------------------------------------------------------


async def _generate_search_query(state: AppState, log: Any) -> str:
    """Generate a search query using Gemini Flash.

    The LLM examines the detected technologies, parameters, URLs, and
    WAF signature to produce a highly specific search query.

    Raises
    ------
    Exception
        If the LLM call fails for any reason. The caller catches this
        and returns ``{"rag_context": None}``.
    """
    structured_llm = gemini_flash.with_structured_output(_SearchQuery)

    user_prompt: str = _build_query_prompt(state)

    result: _SearchQuery = await structured_llm.ainvoke(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    return result.query


def _build_query_prompt(state: AppState) -> str:
    """Build the user prompt for query generation.

    Extracts technologies, parameters, URLs, and WAF signature from
    the state and formats them for the LLM. Each section is truncated
    to stay within token limits.
    """
    # --- Target ---
    target: Target | None = state.get("target")
    target_url: str = str(target.url) if target is not None else "(unknown)"

    # --- Technologies ---
    recon_data: ReconResult | None = state.get("recon_data")
    if recon_data is not None and recon_data.technologies_detected:
        technologies: str = ", ".join(recon_data.technologies_detected)
    else:
        technologies = "(none detected)"

    # --- Parameters ---
    crawler_data: CrawlerResult | None = state.get("crawler_data")
    if crawler_data is not None and crawler_data.parameters:
        param_lines: list[str] = []
        for p in crawler_data.parameters[:20]:
            param_lines.append(
                f"  {p.name} ({p.location.value}"
                + (f", type={p.param_type}" if p.param_type else "")
                + ")"
            )
        parameters: str = "\n".join(param_lines)
    else:
        parameters = "(none discovered)"

    # --- URLs ---
    if crawler_data is not None and crawler_data.urls:
        url_lines: list[str] = [
            f"  {u}" for u in crawler_data.urls[:10]
        ]
        urls: str = "\n".join(url_lines)
    else:
        urls = "(none discovered)"

    # --- WAF signature ---
    if recon_data is not None:
        waf_sig: str = (
            recon_data.waf_signature.value
            if recon_data.waf_signature
            else "none"
        )
    else:
        waf_sig = "unknown"

    return _USER_PROMPT_TEMPLATE.format(
        target_url=target_url,
        technologies=technologies[:_MAX_CONTEXT_CHARS],
        parameters=parameters[:_MAX_CONTEXT_CHARS],
        urls=urls[:_MAX_CONTEXT_CHARS],
        waf_signature=waf_sig,
    )


# ---------------------------------------------------------------------------
# Internal: BGE-M3 embedding
# ---------------------------------------------------------------------------


def _get_embedder(log: Any) -> Any:
    """Get (or lazily load) the BGE-M3 embedder.

    The embedder is cached at module level to avoid re-loading the
    ~2GB model weights on every call. If the first load attempt fails
    (missing package, model download failure), subsequent calls skip
    the retry and raise immediately.

    Raises
    ------
    ImportError
        If ``langchain-community`` or ``langchain-huggingface`` is not
        installed.
    Exception
        If the model weights cannot be loaded.
    """
    global _embedder, _embedder_load_attempted

    if _embedder is not None:
        return _embedder

    if _embedder_load_attempted:
        # Previous load failed — don't retry on every call.
        raise RuntimeError(
            "BGE-M3 embedder load previously failed. "
            "Will not retry. Install langchain-community and ensure "
            "model weights are available."
        )

    _embedder_load_attempted = True

    # Try langchain-community first, then langchain-huggingface.
    try:
        try:
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        except ImportError:
            from langchain_huggingface import HuggingFaceEmbeddings as HuggingFaceBgeEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Cannot load BGE-M3 embedder: langchain-community (or "
            "langchain-huggingface) is not installed. Install with "
            "'pip install langchain-community sentence-transformers'."
        ) from exc

    log.info(
        "rag_embedder_loading",
        model_name=settings.EMBEDDER_MODEL_NAME,
    )

    _embedder = HuggingFaceBgeEmbeddings(
        model_name=settings.EMBEDDER_MODEL_NAME,
    )

    log.info("rag_embedder_loaded", model_name=settings.EMBEDDER_MODEL_NAME)

    return _embedder


def _embed_query(query: str, log: Any) -> list[float]:
    """Embed the search query using BGE-M3.

    Returns
    -------
    list[float]
        The query embedding vector (typically 1024 dimensions for BGE-M3).

    Raises
    ------
    Exception
        If the embedder cannot be loaded or the embedding fails.
    """
    embedder = _get_embedder(log)
    # HuggingFaceBgeEmbeddings.embed_query is synchronous.
    return embedder.embed_query(query)


# ---------------------------------------------------------------------------
# Internal: Qdrant search
# ---------------------------------------------------------------------------


def _get_qdrant_client(log: Any) -> Any:
    """Get (or lazily create) the Qdrant client.

    The client is cached at module level to avoid re-establishing the
    HTTP connection on every call.

    Raises
    ------
    ImportError
        If ``qdrant-client`` is not installed.
    Exception
        If the Qdrant server is unreachable.
    """
    global _qdrant_client, _qdrant_connect_attempted

    if _qdrant_client is not None:
        return _qdrant_client

    if _qdrant_connect_attempted:
        raise RuntimeError(
            "Qdrant connection previously failed. Will not retry."
        )

    _qdrant_connect_attempted = True

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError(
            "Cannot connect to Qdrant: qdrant-client is not installed. "
            "Install with 'pip install qdrant-client'."
        ) from exc

    log.info(
        "rag_qdrant_connecting",
        url=settings.QDRANT_URL,
        collection=settings.QDRANT_COLLECTION,
    )

    # Create the client. If QDRANT_API_KEY is set (Qdrant Cloud),
    # pass it; otherwise connect in local mode.
    client_kwargs: dict[str, Any] = {"url": settings.QDRANT_URL}
    api_key: str | None = settings.get_qdrant_api_key()
    if api_key:
        client_kwargs["api_key"] = api_key

    _qdrant_client = QdrantClient(**client_kwargs)

    # Verify the collection exists. This will raise if Qdrant is
    # unreachable or the collection doesn't exist.
    _qdrant_client.get_collection(collection_name=settings.QDRANT_COLLECTION)

    log.info("rag_qdrant_connected", collection=settings.QDRANT_COLLECTION)

    return _qdrant_client


def _search_qdrant(query_vector: list[float], log: Any) -> list[Any]:
    """Search the Qdrant collection with the query vector.

    Returns
    -------
    list
        A list of Qdrant ``ScoredPoint`` objects, sorted by score
        (highest first). May be empty if no results match.

    Raises
    ------
    Exception
        If Qdrant is unreachable or the search fails.
    """
    client = _get_qdrant_client(log)

    log.info(
        "rag_qdrant_searching",
        collection=settings.QDRANT_COLLECTION,
        top_k=settings.RAG_TOP_K,
        similarity_threshold=settings.RAG_SIMILARITY_THRESHOLD,
    )

    results: list[Any] = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=settings.RAG_TOP_K,
        with_payload=True,
    )

    return results


# ---------------------------------------------------------------------------
# Internal: Qdrant payload → RAGDocument mapping
# ---------------------------------------------------------------------------


def _map_qdrant_payload(
    payload: dict[str, Any], log: Any
) -> RAGDocument | None:
    """Map a Qdrant payload dict to a :class:`RAGDocument`.

    The payload should contain:
    - ``source`` (str): the source document name.
    - ``section`` (str | None): the section within the source.
    - ``content`` (str): the actual text content.
    - ``methodology_tags`` (list[str] | None): vulnerability category slugs.
    - ``source_url`` (str | None): the source URL.

    If required fields are missing or the payload is malformed, the
    function logs a warning and returns ``None`` (the result is
    silently dropped from the retrieved documents).
    """
    if not payload:
        log.warning("rag_empty_payload")
        return None

    # --- source (required) ---
    source: str = payload.get("source", "")
    if not source:
        log.warning("rag_missing_source_field", payload_keys=list(payload.keys()))
        return None

    # --- content (required) ---
    content: str = payload.get("content", "")
    if not content:
        log.warning("rag_missing_content_field", source=source)
        return None

    # --- section (optional) ---
    section: str | None = payload.get("section")
    if section is not None and not isinstance(section, str):
        section = str(section)

    # --- methodology_tags (optional, convert strings to enums) ---
    raw_tags: list[Any] = payload.get("methodology_tags", [])
    methodology_tags: list[VulnerabilityCategory] = []
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            if isinstance(tag, str):
                try:
                    methodology_tags.append(VulnerabilityCategory(tag))
                except ValueError:
                    # Unknown tag slug — skip it rather than crashing.
                    log.debug(
                        "rag_unknown_methodology_tag",
                        tag=tag,
                        source=source,
                    )
            elif isinstance(tag, VulnerabilityCategory):
                methodology_tags.append(tag)

    # --- source_url (optional, keep as string — RAGDocument validates) ---
    source_url: str | None = payload.get("source_url")
    if source_url is not None and not isinstance(source_url, str):
        source_url = str(source_url)

    try:
        return RAGDocument(
            source=source,
            section=section,
            content=content,
            methodology_tags=methodology_tags,
            source_url=source_url,  # type: ignore[arg-type]
        )
    except Exception as exc:
        log.warning(
            "rag_payload_mapping_failed",
            source=source,
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
        )
        return None


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = ["retrieve_knowledge"]
