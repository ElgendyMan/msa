"""
shared/config.py
================

Strongly-typed environment configuration for the Zero-Budget Autonomous
Web Pentesting Framework.

Loads from a ``.env`` file (via ``pydantic-settings``) and validates
every value. The application fails loudly at *import time* if neither
``GEMINI_API_KEYS`` nor ``DEEPSEEK_API_KEYS`` contains at least one
valid key — these are the framework's hard primary dependencies.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.shared.exceptions import ConfigurationError


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH: Path = PROJECT_ROOT / ".env"
SCOPE_FILE_PATH: Path = PROJECT_ROOT / "scope.json"
KNOWLEDGE_BASE_DIR: Path = PROJECT_ROOT / "data" / "knowledge_base"
QDRANT_PERSIST_DIR: Path = PROJECT_ROOT / "data" / "qdrant"
REPORTS_DIR: Path = PROJECT_ROOT / "data" / "reports"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_PATH),
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=True,
        extra="ignore",
        frozen=True,
    )

    # ---- PRIMARY API KEYS ----
    GEMINI_API_KEYS: SecretStr = Field(default=SecretStr(""))
    DEEPSEEK_API_KEYS: SecretStr = Field(default=SecretStr(""))

    # ---- FALLBACK TIER 2: Groq ----
    GROQ_API_KEYS: SecretStr = Field(default=SecretStr(""))

    # ---- ADDITIONAL FALLBACK PROVIDERS ----
    CEREBRAS_API_KEYS: SecretStr = Field(default=SecretStr(""))
    GITHUB_API_KEYS: SecretStr = Field(default=SecretStr(""))
    OPENROUTER_API_KEYS: SecretStr = Field(default=SecretStr(""))
    MISTRAL_API_KEYS: SecretStr = Field(default=SecretStr(""))
    COHERE_API_KEYS: SecretStr = Field(default=SecretStr(""))
    CLOUDFLARE_API_KEYS: SecretStr = Field(default=SecretStr(""))
    CLOUDFLARE_ACCOUNT_ID: str = Field(default="")

    # ---- FALLBACK TIER 3: Ollama ----
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="dolphin-llama3")

    # ---- LLM tuning ----
    GEMINI_FLASH_MODEL: str = Field(default="gemini-2.5-flash")
    GEMINI_FLASH_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    GEMINI_PRO_MODEL: str = Field(default="gemini-2.5-pro")
    GEMINI_PRO_TEMPERATURE: float = Field(default=0.4, ge=0.0, le=2.0)
    DEEPSEEK_R1_MODEL: str = Field(default="deepseek-reasoner")
    DEEPSEEK_R1_TEMPERATURE: float = Field(default=0.6, ge=0.0, le=2.0)
    DEEPSEEK_V3_MODEL: str = Field(default="deepseek-chat")
    DEEPSEEK_V3_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    DEEPSEEK_BASE_URL: str = Field(default="https://api.deepseek.com/v1")
    LLM_MAX_RETRIES: int = Field(default=3, ge=0, le=10)
    LLM_REQUEST_TIMEOUT_SECONDS: float = Field(default=60.0, ge=1.0, le=600.0)
    GROQ_FLASH_MODEL: str = Field(default="llama-3.1-8b-instant")
    GROQ_PRO_MODEL: str = Field(default="llama-3.3-70b-versatile")

    # ---- Execution sandbox ----
    EXECUTION_TIMEOUT_SECONDS: float = Field(default=30.0, ge=1.0, le=300.0)
    EXECUTION_MAX_RETRIES: int = Field(default=2, ge=0, le=5)
    EXECUTION_MAX_CONCURRENT: int = Field(default=5, ge=1, le=50)
    EXECUTION_RATE_LIMIT_RPS: int = Field(default=10, ge=1, le=200)
    PLAYWRIGHT_NAV_TIMEOUT_MS: int = Field(default=15_000, ge=1_000, le=120_000)
    PLAYWRIGHT_CRAWL_DEPTH: int = Field(default=3, ge=0, le=10)

    # ---- RAG / Qdrant ----
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_COLLECTION: str = Field(default="pentest_methodology")
    QDRANT_API_KEY: SecretStr | None = Field(default=None)
    RAG_TOP_K: int = Field(default=5, ge=1, le=50)
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.55, ge=0.0, le=1.0)
    EMBEDDER_MODEL_NAME: str = Field(default="BAAI/bge-m3")

    # ---- Validation policy ----
    VALIDATION_CONFIDENCE_THRESHOLD: float = Field(default=0.6, ge=0.0, le=1.0)
    HYPOTHESIS_CONFIDENCE_THRESHOLD: float = Field(default=0.4, ge=0.0, le=1.0)

    # ---- Operational ----
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    SESSION_MAX_RETRIES: int = Field(default=3, ge=0, le=20)

    @field_validator("DEEPSEEK_BASE_URL")
    @classmethod
    def _normalize_base_url(cls, v: str) -> str:
        if not v:
            raise ValueError("DEEPSEEK_BASE_URL must not be empty.")
        return v.rstrip("/")

    @model_validator(mode="after")
    def _at_least_one_primary_key(self) -> "Settings":
        gemini_keys: list[str] = self.get_gemini_api_keys()
        deepseek_keys: list[str] = self.get_deepseek_api_keys()
        if not gemini_keys and not deepseek_keys:
            raise ValueError(
                "No primary API keys found. At least one of "
                "GEMINI_API_KEYS or DEEPSEEK_API_KEYS must contain "
                "at least one non-empty, comma-separated key."
            )
        return self

    def _parse_secret_keys(self, field: SecretStr) -> list[str]:
        if field is None:
            return []
        raw: str = field.get_secret_value()
        if not raw or not raw.strip():
            return []
        keys: list[str] = []
        for part in raw.split(","):
            key: str = part.strip()
            if key:
                keys.append(key)
        return keys

    def get_gemini_api_keys(self) -> list[str]:
        return self._parse_secret_keys(self.GEMINI_API_KEYS)

    def get_deepseek_api_keys(self) -> list[str]:
        return self._parse_secret_keys(self.DEEPSEEK_API_KEYS)

    def get_groq_api_keys(self) -> list[str]:
        return self._parse_secret_keys(self.GROQ_API_KEYS)

    # ---- short-name aliases used by src.shared.llm ----
    def get_gemini_keys(self) -> list[str]:
        return self.get_gemini_api_keys()

    def get_groq_keys(self) -> list[str]:
        return self.get_groq_api_keys()

    def get_cerebras_keys(self) -> list[str]:
        return self._parse_secret_keys(self.CEREBRAS_API_KEYS)

    def get_github_keys(self) -> list[str]:
        return self._parse_secret_keys(self.GITHUB_API_KEYS)

    def get_openrouter_keys(self) -> list[str]:
        return self._parse_secret_keys(self.OPENROUTER_API_KEYS)

    def get_mistral_keys(self) -> list[str]:
        return self._parse_secret_keys(self.MISTRAL_API_KEYS)

    def get_cohere_keys(self) -> list[str]:
        return self._parse_secret_keys(self.COHERE_API_KEYS)

    def get_cloudflare_keys(self) -> list[str]:
        return self._parse_secret_keys(self.CLOUDFLARE_API_KEYS)

    # ---- backward-compatible single-key accessors ----
    def get_gemini_api_key(self) -> str:
        keys: list[str] = self.get_gemini_api_keys()
        return keys[0] if keys else ""

    def get_deepseek_api_key(self) -> str:
        keys: list[str] = self.get_deepseek_api_keys()
        return keys[0] if keys else ""

    def get_qdrant_api_key(self) -> str | None:
        if self.QDRANT_API_KEY is None:
            return None
        return self.QDRANT_API_KEY.get_secret_value()


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except Exception as exc:  # noqa: BLE001
        raise ConfigurationError(
            "Failed to load framework configuration. At least one "
            "primary API key (GEMINI_API_KEYS or DEEPSEEK_API_KEYS) "
            "must contain at least one non-empty key.",
            details={
                "env_file": str(DEFAULT_ENV_PATH),
                "project_root": str(PROJECT_ROOT),
                "cause": str(exc),
            },
        ) from exc


settings: Settings = _load_settings()


__all__ = [
    "Settings", "settings", "PROJECT_ROOT", "DEFAULT_ENV_PATH",
    "SCOPE_FILE_PATH", "KNOWLEDGE_BASE_DIR", "QDRANT_PERSIST_DIR", "REPORTS_DIR",
]