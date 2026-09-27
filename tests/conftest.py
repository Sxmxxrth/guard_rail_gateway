"""Pytest fixtures and configuration for OmniGuard test suite."""

import pytest

from guard_rail_gateway.cache.semantic_cache import SemanticCache
from guard_rail_gateway.config import Settings
from guard_rail_gateway.guardrails.pipeline import GuardrailPipeline
from guard_rail_gateway.ratelimit.manager import RateLimitManager
from guard_rail_gateway.router.resilient_router import ResilientRouter
from guard_rail_gateway.telemetry import TelemetryManager


@pytest.fixture
def mock_config():
    """Provides a deterministic test configuration."""
    return Settings(
        default_provider="mock",
        offline_mode=True,
        cache_enabled=True,
        semantic_cache_threshold=0.85,
        default_rpm=60,
        default_tpm=60000,
    )


@pytest.fixture
def semantic_cache():
    """Provides a fresh semantic cache instance."""
    return SemanticCache(
        similarity_threshold=0.85,
        max_entries=100,
        default_ttl_seconds=300,
    )


@pytest.fixture
def guardrail_pipeline(mock_config):
    """Provides a fresh guardrail pipeline instance."""
    return GuardrailPipeline(config=mock_config)


@pytest.fixture
def rate_limiter(mock_config):
    """Provides a rate limit manager."""
    return RateLimitManager.from_config(mock_config)


@pytest.fixture
def router(mock_config):
    """Provides a resilient router configured for offline mock."""
    return ResilientRouter(config=mock_config)


@pytest.fixture
def telemetry():
    """Provides a fresh telemetry manager."""
    return TelemetryManager()
