"""Configuration management for OmniGuard AI Gateway & Guardrails Platform."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for OmniGuard Gateway."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # API Keys & Credentials
    openai_api_key: str | None = Field(default=None, description="OpenAI API Key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API Key")
    gemini_api_key: str | None = Field(default=None, description="Google Gemini API Key")
    google_api_key: str | None = Field(default=None, description="Google API Key alias")
    groq_api_key: str | None = Field(default=None, description="Groq API Key")
    gateway_api_key: str = Field(default="sk-omniguard-default-key", description="Master API Key for Gateway proxy access")
    admin_api_key: str = Field(default="sk-omniguard-admin-secret", description="Admin key for management endpoints")

    # Semantic Cache Settings
    cache_enabled: bool = Field(default=True, description="Enable two-tier exact & semantic cache")
    semantic_cache_threshold: float = Field(
        default=0.88,
        ge=0.5,
        le=1.0,
        description="Cosine similarity threshold for vector semantic cache hit",
    )
    cache_ttl_seconds: int = Field(default=3600, ge=60, description="Cache entry Time-To-Live in seconds")
    cache_max_entries: int = Field(default=1000, ge=10, description="Max entries before LRU eviction")

    # Guardrails Configuration
    prompt_injection_enabled: bool = Field(default=True, description="Scan prompts for injection & jailbreaks")
    injection_action: Literal["block", "flag", "quarantine"] = Field(
        default="block",
        description="Action when injection is detected",
    )
    pii_anonymization_enabled: bool = Field(default=True, description="Anonymize PII and secret tokens")
    pii_action: Literal["anonymize", "block", "flag"] = Field(
        default="anonymize",
        description="Action when PII is detected in input",
    )
    toxicity_filter_enabled: bool = Field(default=True, description="Evaluate toxicity and harmful intent")
    output_guard_enabled: bool = Field(default=True, description="Verify model output safety & hallucination bounds")

    # Rate Limiting Configuration
    rate_limit_enabled: bool = Field(default=True, description="Enforce Token Bucket rate limiting per tenant")
    default_rpm: int = Field(default=60, ge=1, description="Default Requests Per Minute limit")
    default_tpm: int = Field(default=60000, ge=100, description="Default Tokens Per Minute limit")

    # Provider & Routing Configuration
    default_provider: Literal["mock", "openai", "anthropic", "gemini", "groq"] = Field(
        default="mock",
        description="Default active LLM provider",
    )
    routing_strategy: Literal["failover_chain", "cost_optimized", "latency_optimized"] = Field(
        default="failover_chain",
        description="Dynamic routing and load-balancing strategy",
    )
    offline_mode: bool = Field(
        default=True,
        description="When True, routes all queries through deterministic offline mock engine",
    )

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="API server bind host")
    port: int = Field(default=8000, ge=1024, le=65535, description="API server bind port")
    log_level: str = Field(default="INFO", description="Logging verbosity level")

    def get_effective_gemini_key(self) -> str | None:
        """Resolves Gemini API key checking GEMINI_API_KEY and GOOGLE_API_KEY."""
        return self.gemini_api_key or self.google_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def is_provider_configured(self, provider: str) -> bool:
        """Checks if the required credentials for a given provider are configured."""
        p = provider.lower()
        if p == "mock":
            return True
        if p == "openai":
            return bool(self.openai_api_key or os.getenv("OPENAI_API_KEY"))
        if p == "anthropic":
            return bool(self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))
        if p in ("gemini", "google"):
            return bool(self.get_effective_gemini_key())
        if p == "groq":
            return bool(self.groq_api_key or os.getenv("GROQ_API_KEY"))
        return False


# Singleton config getter
_config_instance: Settings | None = None


def get_config() -> Settings:
    """Returns singleton instance of application settings."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Settings()
    return _config_instance
