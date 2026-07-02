"""
shared/llm.py
=============

Omni-Fallback LLM Factory (Purpose-Driven Architecture).

This module builds 4 LLM chains, each optimized for a specific purpose:

1. **deepseek_r1** (Reasoning): For Planner, Validator, Hypothesis Analyzer.
   Needs high intelligence and step-by-step logic.
   Chain: Groq(70B) → GitHub(70B) → OpenRouter(70B) → Cohere → Mistral

2. **deepseek_v3** (Payloads): For Payload Generator, Optimizer.
   Needs uncensored models and strong coding capabilities.
   Chain: OpenRouter(Qwen Coder) → OpenRouter(Dolphin) → Groq(70B) → GitHub(70B) → OpenRouter(Nemotron)

3. **gemini_flash** (Parsing): For Recon Parser, Crawler Parser.
   Needs extreme speed and reliable JSON formatting.
   Chain: Groq(8B) → Cerebras(8B) → Gemini Flash → Cloudflare(8B)

4. **gemini_pro** (Reporting): For Reporter, Memory Summarizer.
   Needs large context window and good narrative writing.
   Chain: Gemini Pro → Cohere → Mistral → GitHub(70B)

Each chain has MULTIPLE fallbacks. If the primary provider hits a rate
limit, the framework automatically falls through to the next provider.
With 8 providers across 4 chains, the framework has ~30+ LLM endpoints
available — making it extremely resilient to rate limits.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.shared.config import settings
from src.shared.exceptions import ConfigurationError
from src.shared.logging import get_logger


class _FallbackChain:
    """Wrapper that provides with_structured_output + fallbacks."""

    def __init__(self, llms: list[BaseChatModel]) -> None:
        if not llms:
            raise ValueError("_FallbackChain requires at least one LLM.")
        self._llms = llms
        self._raw = llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]

    def with_structured_output(self, schema: Any, **kw: Any) -> Any:
        structured: list[Any] = []
        for llm in self._llms:
            try:
                structured.append(llm.with_structured_output(schema, **kw))
            except Exception:
                pass
        if not structured:
            raise RuntimeError("No LLM supports with_structured_output for this schema.")
        return structured[0].with_fallbacks(structured[1:]) if len(structured) > 1 else structured[0]

    async def ainvoke(self, *a: Any, **kw: Any) -> Any:
        return await self._raw.ainvoke(*a, **kw)

    def invoke(self, *a: Any, **kw: Any) -> Any:
        return self._raw.invoke(*a, **kw)

    @property
    def model_name(self) -> str:
        for attr in ("model", "model_name"):
            v = getattr(self._llms[0], attr, None)
            if v:
                return str(v)
        return "unknown"

    @property
    def primary(self) -> BaseChatModel:
        return self._llms[0]

    @property
    def fallback_count(self) -> int:
        return max(0, len(self._llms) - 1)

    def __repr__(self) -> str:
        return f"_FallbackChain(primary={self.model_name!r}, fallbacks={self.fallback_count})"


_LOG = get_logger("llm_factory")

# Strict timeouts and retries for speed.
_STRICT_TIMEOUT = 30
_STRICT_RETRIES = 1


# ---------------------------------------------------------------------------
# Provider builders
# ---------------------------------------------------------------------------


def _build_openrouter(model: str, temperature: float) -> list[BaseChatModel]:
    keys = settings.get_openrouter_keys()
    if not keys:
        return []
    try:
        return [
            ChatOpenAI(
                model=model,
                api_key=k,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_retries=_STRICT_RETRIES,
                timeout=_STRICT_TIMEOUT,
            )
            for k in keys
        ]
    except Exception:
        return []


def _build_groq(model: str, temperature: float) -> list[BaseChatModel]:
    keys = settings.get_groq_keys()
    if not keys:
        return []
    try:
        from langchain_groq import ChatGroq
        return [
            ChatGroq(
                model=model,
                api_key=k,
                temperature=temperature,
                max_retries=_STRICT_RETRIES,
                timeout=_STRICT_TIMEOUT,
            )
            for k in keys
        ]
    except Exception:
        return []


def _build_cerebras(model: str, temperature: float) -> list[BaseChatModel]:
    keys = settings.get_cerebras_keys()
    if not keys:
        return []
    try:
        return [
            ChatOpenAI(
                model=model,
                api_key=k,
                base_url="https://api.cerebras.ai/v1",
                temperature=temperature,
                max_retries=_STRICT_RETRIES,
                timeout=_STRICT_TIMEOUT,
            )
            for k in keys
        ]
    except Exception:
        return []


def _build_github(model: str, temperature: float) -> list[BaseChatModel]:
    keys = settings.get_github_keys()
    if not keys:
        return []
    try:
        return [
            ChatOpenAI(
                model=model,
                api_key=k,
                base_url="https://models.inference.ai.azure.com",
                temperature=temperature,
                max_retries=_STRICT_RETRIES,
                timeout=_STRICT_TIMEOUT,
            )
            for k in keys
        ]
    except Exception:
        return []


def _build_mistral(model: str, temperature: float) -> list[BaseChatModel]:
    keys = settings.get_mistral_keys()
    if not keys:
        return []
    try:
        from langchain_mistralai import ChatMistralAI
        return [
            ChatMistralAI(
                model=model,
                api_key=k,
                temperature=temperature,
                max_retries=_STRICT_RETRIES,
                timeout=_STRICT_TIMEOUT,
            )
            for k in keys
        ]
    except Exception:
        return []


def _build_cohere(model: str, temperature: float) -> list[BaseChatModel]:
    keys = settings.get_cohere_keys()
    if not keys:
        return []
    try:
        from langchain_cohere import ChatCohere
        return [
            ChatCohere(
                model=model,
                cohere_api_key=k,
                temperature=temperature,
                max_retries=_STRICT_RETRIES,
                timeout=_STRICT_TIMEOUT,
            )
            for k in keys
        ]
    except Exception:
        return []


def _build_cloudflare(model: str, temperature: float) -> list[BaseChatModel]:
    account_id = settings.CLOUDFLARE_ACCOUNT_ID
    keys = settings.get_cloudflare_keys()
    if not account_id or not keys:
        return []
    try:
        return [
            ChatOpenAI(
                model=model,
                api_key=k,
                base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
                temperature=temperature,
                max_retries=_STRICT_RETRIES,
                timeout=_STRICT_TIMEOUT,
            )
            for k in keys
        ]
    except Exception:
        return []


def _build_gemini(model: str, temperature: float) -> list[BaseChatModel]:
    keys = settings.get_gemini_keys()
    if not keys:
        return []
    try:
        return [
            ChatGoogleGenerativeAI(
                model=model,
                api_key=k,
                temperature=temperature,
                max_retries=_STRICT_RETRIES,
                timeout=_STRICT_TIMEOUT,
            )
            for k in keys
        ]
    except Exception:
        return []


def _build_ollama(temperature: float) -> list[BaseChatModel]:
    try:
        from langchain_ollama import ChatOllama
        return [
            ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=temperature,
                num_ctx=8192,
            )
        ]
    except Exception:
        return []


def _assemble(
    name: str,
    temperature: float,
    tiers: list[list[BaseChatModel]],
) -> _FallbackChain:
    """Assemble a fallback chain from multiple provider tiers."""
    flat: list[BaseChatModel] = []
    for tier in tiers:
        flat.extend(tier)
    # Always add Ollama as the last-resort fallback.
    flat.extend(_build_ollama(temperature))
    if not flat:
        raise ConfigurationError(
            f"Cannot build '{name}': no LLM providers available."
        )
    _LOG.info(
        "fallback_chain_assembled",
        name=name,
        total_llms=len(flat),
        primary=getattr(flat[0], "model", "unknown"),
    )
    return _FallbackChain(flat)


# ===========================================================================
# Purpose-Driven Chains
# ===========================================================================


@lru_cache(maxsize=1)
def get_deepseek_r1() -> _FallbackChain:
    """REASONING chain (Planner, Validator, Hypothesis Analyzer).
    Needs high intelligence and step-by-step logic."""
    t = 0.1
    return _assemble("deepseek_r1", temperature=t, tiers=[
        _build_groq("llama-3.3-70b-versatile", t),
        _build_github("Llama-3.3-70B-Instruct", t),
        _build_openrouter("meta-llama/llama-3.3-70b-instruct:free", t),
        _build_cohere("command-r-plus", t),
        _build_mistral("mistral-large-latest", t),
    ])


@lru_cache(maxsize=1)
def get_deepseek_v3() -> _FallbackChain:
    """PAYLOADS chain (Generator, Optimizer).
    Needs uncensored models and strong coding capabilities."""
    t = 0.6
    return _assemble("deepseek_v3", temperature=t, tiers=[
        _build_openrouter("qwen/qwen3-coder:free", t),
        _build_openrouter("cognitivecomputations/dolphin-mistral-24b-venice-edition:free", t),
        _build_groq("llama-3.3-70b-versatile", t),
        _build_github("Llama-3.3-70B-Instruct", t),
        _build_openrouter("nvidia/nemotron-3-ultra-550b-a55b:free", t),
    ])


@lru_cache(maxsize=1)
def get_gemini_flash() -> _FallbackChain:
    """PARSING chain (Recon, Crawler, Web Filter).
    Needs extreme speed and reliable JSON formatting."""
    t = 0.1
    return _assemble("gemini_flash", temperature=t, tiers=[
        _build_groq("llama-3.1-8b-instant", t),
        _build_cerebras("llama3.1-8b", t),
        _build_gemini(settings.GEMINI_FLASH_MODEL, t),
        _build_cloudflare("@cf/meta/llama-3-8b-instruct", t),
    ])


@lru_cache(maxsize=1)
def get_gemini_pro() -> _FallbackChain:
    """REPORTING chain (Reporter, Memory Summarizer).
    Needs large context window and good narrative writing."""
    t = 0.3
    return _assemble("gemini_pro", temperature=t, tiers=[
        _build_gemini(settings.GEMINI_PRO_MODEL, t),
        _build_cohere("command-r-plus", t),
        _build_mistral("mistral-large-latest", t),
        _build_github("Llama-3.3-70B-Instruct", t),
    ])


def _instantiate_or_raise(builder: Any, display_name: str) -> Any:
    try:
        return builder()
    except Exception as exc:
        raise ConfigurationError(f"Failed to build {display_name}: {exc}") from exc


# Module-level instances (keeping original names to avoid breaking imports).
gemini_flash = _instantiate_or_raise(get_gemini_flash, "gemini_flash")
gemini_pro = _instantiate_or_raise(get_gemini_pro, "gemini_pro")
deepseek_r1 = _instantiate_or_raise(get_deepseek_r1, "deepseek_r1")
deepseek_v3 = _instantiate_or_raise(get_deepseek_v3, "deepseek_v3")

_MODEL_REGISTRY = {
    "gemini_flash": gemini_flash,
    "gemini_pro": gemini_pro,
    "deepseek_r1": deepseek_r1,
    "deepseek_v3": deepseek_v3,
}


def get_model(name: str) -> _FallbackChain:
    return _MODEL_REGISTRY[name]


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())


__all__ = [
    "gemini_flash",
    "gemini_pro",
    "deepseek_r1",
    "deepseek_v3",
    "get_gemini_flash",
    "get_gemini_pro",
    "get_deepseek_r1",
    "get_deepseek_v3",
    "get_model",
    "list_models",
    "_FallbackChain",
]
