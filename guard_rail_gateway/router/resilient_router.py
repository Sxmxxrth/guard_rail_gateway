"""Intelligent multi-provider routing engine with automatic failover and circuit breaker protection."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from guard_rail_gateway.config import Settings, get_config
from guard_rail_gateway.models import ChatCompletionRequest, ChatCompletionResponse
from guard_rail_gateway.router.circuit_breaker import CircuitBreaker
from guard_rail_gateway.router.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    GeminiProvider,
    GroqProvider,
    MockLLMProvider,
    OpenAIProvider,
)

logger = logging.getLogger("guard_rail_gateway.router")


class ResilientRouter:
    """Coordinates upstream LLM requests with dynamic routing strategies and failover recovery."""

    def __init__(self, config: Settings | None = None):
        self.config = config or get_config()

        # Initialize Providers
        self.mock_provider = MockLLMProvider()
        self.providers: dict[str, BaseLLMProvider] = {"mock": self.mock_provider}
        self.breakers: dict[str, CircuitBreaker] = {
            "mock": CircuitBreaker(name="mock", failure_threshold=10, recovery_timeout_seconds=5.0)
        }

        # Initialize configured real providers
        if self.config.openai_api_key:
            self.providers["openai"] = OpenAIProvider(api_key=self.config.openai_api_key)
            self.breakers["openai"] = CircuitBreaker(name="openai")

        if self.config.groq_api_key:
            self.providers["groq"] = GroqProvider(api_key=self.config.groq_api_key)
            self.breakers["groq"] = CircuitBreaker(name="groq")

        gemini_key = self.config.get_effective_gemini_key()
        if gemini_key:
            self.providers["gemini"] = GeminiProvider(api_key=gemini_key)
            self.breakers["gemini"] = CircuitBreaker(name="gemini")

        if self.config.anthropic_api_key:
            self.providers["anthropic"] = AnthropicProvider(api_key=self.config.anthropic_api_key)
            self.breakers["anthropic"] = CircuitBreaker(name="anthropic")

        # Exponentially Weighted Moving Average (EWMA) latency per provider (ms)
        self.latency_ewma: dict[str, float] = {p: 100.0 for p in self.providers}

    def _get_provider_chain(self) -> list[str]:
        """Calculates prioritized provider ordering based on active routing strategy."""
        if self.config.offline_mode or self.config.default_provider == "mock":
            return ["mock"]

        strategy = self.config.routing_strategy
        active_providers = [p for p in self.providers if p != "mock"]

        if not active_providers:
            return ["mock"]

        if strategy == "cost_optimized":
            # Sort by output cost ascending
            sorted_providers = sorted(active_providers, key=lambda p: self.providers[p].cost_per_1k_output)
            return sorted_providers + ["mock"]

        elif strategy == "latency_optimized":
            # Sort by running EWMA latency ascending
            sorted_providers = sorted(active_providers, key=lambda p: self.latency_ewma.get(p, 999.0))
            return sorted_providers + ["mock"]

        else:  # failover_chain
            primary = self.config.default_provider
            chain = [primary] if primary in active_providers else []
            for p in ["groq", "gemini", "openai", "anthropic"]:
                if p in active_providers and p not in chain:
                    chain.append(p)
            chain.append("mock")
            return chain

    def _update_latency(self, provider_name: str, elapsed_ms: float) -> None:
        """Updates the running EWMA latency score for a provider."""
        current = self.latency_ewma.get(provider_name, elapsed_ms)
        alpha = 0.2  # Smoothing factor
        self.latency_ewma[provider_name] = round((alpha * elapsed_ms) + ((1.0 - alpha) * current), 2)

    async def route_and_complete_async(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[ChatCompletionResponse, str, bool]:
        """Routes request across provider chain with circuit breaker and failover recovery.
        
        Returns:
            Tuple of (response: ChatCompletionResponse, provider_used: str, failover_occurred: bool)
        """
        chain = self._get_provider_chain()
        failover_occurred = False

        for idx, provider_name in enumerate(chain):
            provider = self.providers.get(provider_name, self.mock_provider)
            breaker = self.breakers.get(provider_name, self.breakers["mock"])

            if not breaker.can_execute():
                logger.warning(f"Circuit breaker OPEN for provider '{provider_name}'; skipping to fallback.")
                failover_occurred = True
                continue

            start_t = time.perf_counter()
            try:
                response = await provider.complete_async(request)
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                breaker.record_success()
                self._update_latency(provider_name, elapsed_ms)

                return response, provider_name, (idx > 0 or failover_occurred)

            except Exception as e:  # noqa: BLE001
                breaker.record_failure()
                logger.warning(f"Provider '{provider_name}' failed with error ({e}); attempting next fallback.")
                failover_occurred = True
                continue

        # Ultimate fallback: deterministic mock
        response = await self.mock_provider.complete_async(request)
        return response, "mock", True

    async def route_and_stream_async(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """Streams completion chunks from the highest-priority healthy provider."""
        chain = self._get_provider_chain()

        for provider_name in chain:
            provider = self.providers.get(provider_name, self.mock_provider)
            breaker = self.breakers.get(provider_name, self.breakers["mock"])

            if not breaker.can_execute():
                continue

            try:
                async for chunk in provider.stream_async(request):
                    yield chunk
                breaker.record_success()
                return
            except Exception as e:  # noqa: BLE001
                breaker.record_failure()
                logger.warning(f"Streaming provider '{provider_name}' failed ({e}); failing over.")
                continue

        # Ultimate fallback streaming
        async for chunk in self.mock_provider.stream_async(request):
            yield chunk
