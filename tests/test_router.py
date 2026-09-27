"""Tests for Circuit Breaker and Resilient Multi-Provider Router."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from guard_rail_gateway.config import Settings
from guard_rail_gateway.models import ChatCompletionRequest, ChatMessage
from guard_rail_gateway.router.circuit_breaker import CircuitBreaker, CircuitState
from guard_rail_gateway.router.providers import BaseLLMProvider, MockLLMProvider
from guard_rail_gateway.router.resilient_router import ResilientRouter

# =============================================================================
# Circuit Breaker State Machine Tests
# =============================================================================

def test_circuit_breaker_initial_state():
    breaker = CircuitBreaker(name="test_provider", failure_threshold=3, recovery_timeout_seconds=1.0)
    assert breaker.state == CircuitState.CLOSED
    assert breaker.can_execute() is True
    assert breaker.failure_count == 0


def test_circuit_breaker_trips_to_open():
    breaker = CircuitBreaker(name="test_provider", failure_threshold=3, recovery_timeout_seconds=0.2)

    # 1st failure: still closed
    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.can_execute() is True

    # 2nd failure: still closed
    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.can_execute() is True

    # 3rd failure: trips to OPEN
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.can_execute() is False


def test_circuit_breaker_half_open_recovery():
    breaker = CircuitBreaker(name="test_provider", failure_threshold=2, recovery_timeout_seconds=0.1)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.can_execute() is False

    # Wait for recovery timeout
    time.sleep(0.12)
    assert breaker.can_execute() is True
    assert breaker.state == CircuitState.HALF_OPEN

    # Canary success -> recovers to CLOSED
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_circuit_breaker_reset():
    breaker = CircuitBreaker(name="test_provider", failure_threshold=1)
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    breaker.reset()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.can_execute() is True


# =============================================================================
# MockLLMProvider Tests
# =============================================================================

@pytest.mark.asyncio
async def test_mock_provider_complete():
    provider = MockLLMProvider()
    request = ChatCompletionRequest(
        model="mock-gpt-4o",
        messages=[ChatMessage(role="user", content="How do I reset my account password?")],
    )

    response = await provider.complete_async(request)
    assert response.id.startswith("chatcmpl-mock-")
    assert len(response.choices) == 1
    assert "password" in response.choices[0].message.content.lower()
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0


@pytest.mark.asyncio
async def test_mock_provider_streaming():
    provider = MockLLMProvider()
    request = ChatCompletionRequest(
        model="mock-gpt-4o",
        messages=[ChatMessage(role="user", content="Hello OmniGuard!")],
        stream=True,
    )

    chunks: list[str] = []
    async for chunk in provider.stream_async(request):
        chunks.append(chunk)

    assert len(chunks) >= 3
    assert chunks[-1] == "data: [DONE]\n\n"
    assert any("data: {" in c for c in chunks)


def test_provider_cost_estimation():
    provider = MockLLMProvider()
    cost = provider.estimate_cost(prompt_tokens=1000, completion_tokens=500)
    assert cost == 0.0

    class PaidProvider(BaseLLMProvider):
        async def complete_async(self, req): pass
        async def stream_async(self, req): yield ""

    paid = PaidProvider(name="paid", cost_per_1k_input=0.01, cost_per_1k_output=0.03)
    paid_cost = paid.estimate_cost(1000, 1000)
    assert paid_cost == 0.04


# =============================================================================
# ResilientRouter Tests
# =============================================================================

@pytest.mark.asyncio
async def test_resilient_router_default_mock():
    config = Settings(offline_mode=True)
    router = ResilientRouter(config=config)

    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="What is an AI gateway?")],
    )
    response, provider_used, failover = await router.route_and_complete_async(request)

    assert provider_used == "mock"
    assert failover is False
    assert "OmniGuard AI Gateway" in response.choices[0].message.content


@pytest.mark.asyncio
async def test_resilient_router_failover_on_provider_error():
    config = Settings(offline_mode=False, default_provider="openai")
    router = ResilientRouter(config=config)

    # Inject a failing primary provider
    flaky_provider = AsyncMock(spec=BaseLLMProvider)
    flaky_provider.complete_async.side_effect = RuntimeError("Upstream 503 Provider Unavailable")
    flaky_provider.cost_per_1k_output = 0.002

    router.providers["openai"] = flaky_provider
    router.breakers["openai"] = CircuitBreaker(name="openai", failure_threshold=2)

    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello, is anyone there?")],
    )
    _, provider_used, failover = await router.route_and_complete_async(request)

    # Flaky provider failed -> failover to mock
    assert provider_used == "mock"
    assert failover is True
    assert router.breakers["openai"].failure_count == 1


@pytest.mark.asyncio
async def test_resilient_router_skip_open_circuit():
    config = Settings(offline_mode=False, default_provider="openai")
    router = ResilientRouter(config=config)

    broken_provider = AsyncMock(spec=BaseLLMProvider)
    broken_provider.cost_per_1k_output = 0.001
    router.providers["openai"] = broken_provider
    breaker = CircuitBreaker(name="openai", failure_threshold=1)
    breaker.record_failure()  # Immediately trip to OPEN
    router.breakers["openai"] = breaker

    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Testing circuit breaker bypass")],
    )
    _, provider_used, failover = await router.route_and_complete_async(request)

    assert provider_used == "mock"
    assert failover is True
    # broken provider was never called because breaker was OPEN
    broken_provider.complete_async.assert_not_called()


@pytest.mark.asyncio
async def test_resilient_router_streaming():
    config = Settings(offline_mode=True)
    router = ResilientRouter(config=config)

    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello streaming!")],
        stream=True,
    )
    chunks = []
    async for chunk in router.route_and_stream_async(request):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert chunks[-1] == "data: [DONE]\n\n"


# =============================================================================
# Real Provider Client Tests with Mocked HTTP
# =============================================================================

from unittest.mock import MagicMock, patch

import httpx

from guard_rail_gateway.router.providers import (
    AnthropicProvider,
    GeminiProvider,
    GroqProvider,
    OpenAIProvider,
)


@pytest.mark.asyncio
async def test_openai_provider_mock_http():
    provider = OpenAIProvider(api_key="sk-test-key")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "id": "chatcmpl-test-openai",
        "object": "chat.completion",
        "created": 123456789,
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "OpenAI answer"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hello OpenAI")])
        res = await provider.complete_async(req)
        assert res.id == "chatcmpl-test-openai"
        assert res.choices[0].message.content == "OpenAI answer"


@pytest.mark.asyncio
async def test_groq_provider_mock_http():
    provider = GroqProvider(api_key="gsk-test-key")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "id": "chatcmpl-test-groq",
        "object": "chat.completion",
        "created": 123456789,
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Groq fast reply"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hello Groq")])
        res = await provider.complete_async(req)
        assert res.id == "chatcmpl-test-groq"
        assert res.choices[0].message.content == "Groq fast reply"


@pytest.mark.asyncio
async def test_anthropic_provider_mock_http():
    provider = AnthropicProvider(api_key="sk-ant-test-key")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "id": "msg_claude_123",
        "model": "claude-3-5-sonnet",
        "content": [{"type": "text", "text": "Claude thoughtful reply"}],
        "usage": {"input_tokens": 20, "output_tokens": 30},
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hello Claude")])
        res = await provider.complete_async(req)
        assert res.choices[0].message.content == "Claude thoughtful reply"
        assert res.usage.prompt_tokens == 20
        assert res.usage.completion_tokens == 30

        # Also test stream_async
        stream_chunks = []
        async for chunk in provider.stream_async(req):
            stream_chunks.append(chunk)
        assert len(stream_chunks) >= 2
        assert "Claude thoughtful reply" in stream_chunks[0]


@pytest.mark.asyncio
async def test_gemini_provider_mock_http():
    provider = GeminiProvider(api_key="AIzaSyTestKey")
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Gemini lightning response"}]
                }
            }
        ]
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hello Gemini")])
        res = await provider.complete_async(req)
        assert res.choices[0].message.content == "Gemini lightning response"

        # Also test stream_async
        stream_chunks = []
        async for chunk in provider.stream_async(req):
            stream_chunks.append(chunk)
        assert len(stream_chunks) >= 2


def test_routing_strategies_chains():
    config = Settings(
        offline_mode=False,
        default_provider="groq",
        openai_api_key="sk-test",
        groq_api_key="gsk-test",
        anthropic_api_key="sk-ant",
        gemini_api_key="aiza-test",
        routing_strategy="cost_optimized",
    )
    router = ResilientRouter(config=config)
    cost_chain = router._get_provider_chain()
    # Cheapest output provider (Groq: 0.0008, Gemini: 0.00105) should be early
    assert cost_chain[0] == "groq"
    assert cost_chain[-1] == "mock"

    # Latency optimized
    router.config.routing_strategy = "latency_optimized"
    router.latency_ewma["anthropic"] = 15.0
    router.latency_ewma["openai"] = 80.0
    latency_chain = router._get_provider_chain()
    assert latency_chain[0] == "anthropic"
    assert latency_chain[-1] == "mock"

