"""Multi-provider LLM integrations (OpenAI, Anthropic, Gemini, Groq, Mock)."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import httpx

from guard_rail_gateway.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)


class BaseLLMProvider(ABC):
    """Abstract interface defining the execution contract for LLM backends."""

    def __init__(self, name: str, cost_per_1k_input: float = 0.005, cost_per_1k_output: float = 0.015):
        self.name = name
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

    @abstractmethod
    async def complete_async(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Asynchronously executes chat completion against upstream model."""

    @abstractmethod
    async def stream_async(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Asynchronously streams response token chunks in SSE data format."""

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimates request cost in USD based on token counts."""
        input_cost = (prompt_tokens / 1000.0) * self.cost_per_1k_input
        output_cost = (completion_tokens / 1000.0) * self.cost_per_1k_output
        return round(input_cost + output_cost, 6)


class MockLLMProvider(BaseLLMProvider):
    """100% deterministic offline LLM provider ensuring live interview demos never fail."""

    def __init__(self, name: str = "mock"):
        super().__init__(name=name, cost_per_1k_input=0.0, cost_per_1k_output=0.0)

    def _generate_mock_reply(self, prompt: str) -> str:
        """Synthesizes high-relevance deterministic mock text tailored to prompt keywords."""
        p = prompt.lower()
        if "password" in p or "reset" in p:
            return (
                "To reset your account password, navigate to the Security Settings tab in your profile. "
                "Click on 'Change Password', verify your identity via the confirmation link sent to your email, "
                "and set a new compliant password containing at least 12 characters."
            )
        elif "api" in p or "endpoint" in p or "gateway" in p:
            return (
                "OmniGuard AI Gateway proxies incoming requests to upstream LLMs while applying two-tier "
                "semantic caching (exact hash + TF-IDF cosine similarity) and real-time guardrail filtering. "
                "It reduces latency by up to 95% and prevents prompt injections."
            )
        elif "python" in p or "code" in p:
            return (
                "```python\ndef calculate_semantic_similarity(vec_a: list[float], vec_b: list[float]) -> float:\n"
                "    \"\"\"Calculates dot-product cosine similarity between normalized vectors.\"\"\"\n"
                "    return sum(a * b for a, b in zip(vec_a, vec_b))\n```\n"
                "This function executes in sub-millisecond time and scales efficiently across memory-bounded indices."
            )
        elif "hello" in p or "hi" in p:
            return "Hello! I am OmniGuard AI Assistant, protected by multi-layer guardrails and semantic caching. How may I assist you today?"
        else:
            return (
                f"OmniGuard Gateway successfully routed your inquiry regarding '{prompt[:50]}'. "
                "The response was generated through the resilient proxy layer with active PII sanitization "
                "and token bucket rate limit validation."
            )

    async def complete_async(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generates deterministic mock completion."""
        user_prompt = request.get_latest_user_prompt()
        content = self._generate_mock_reply(user_prompt)

        p_tokens = max(1, len(user_prompt.split()))
        c_tokens = max(1, len(content.split()))

        return ChatCompletionResponse(
            id=f"chatcmpl-mock-{uuid.uuid4().hex[:10]}",
            model=request.model or "mock-gpt-4o",
            created=int(time.time()),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=p_tokens + c_tokens,
            ),
        )

    async def stream_async(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Streams mock response word-by-word with realistic token intervals."""
        user_prompt = request.get_latest_user_prompt()
        content = self._generate_mock_reply(user_prompt)
        words = content.split(" ")
        cmpl_id = f"chatcmpl-mock-{uuid.uuid4().hex[:10]}"

        for i, word in enumerate(words):
            chunk_token = word if i == 0 else f" {word}"
            chunk_data = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_token},
                        "finish_reason": None if i < len(words) - 1 else "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            await asyncio.sleep(0.015)

        yield "data: [DONE]\n\n"


class OpenAIProvider(BaseLLMProvider):
    """Upstream OpenAI Chat Completions API provider."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        super().__init__(name="openai", cost_per_1k_input=0.005, cost_per_1k_output=0.015)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def complete_async(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = request.model_dump(exclude={"bypass_cache"})

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return ChatCompletionResponse.model_validate(data)

    async def stream_async(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = request.model_dump(exclude={"bypass_cache"})
        payload["stream"] = True

        async with (
            httpx.AsyncClient(timeout=30.0) as client,
            client.stream("POST", url, json=payload, headers=headers) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield f"{line}\n\n"


class GroqProvider(BaseLLMProvider):
    """Ultra-low latency Groq LPU provider using OpenAI-compatible interface."""

    def __init__(self, api_key: str):
        super().__init__(name="groq", cost_per_1k_input=0.0005, cost_per_1k_output=0.0008)
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"

    async def complete_async(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Use Groq model default if needed
        model = request.model if "llama" in request.model.lower() or "mixtral" in request.model.lower() else "llama-3.3-70b-versatile"
        payload = request.model_dump(exclude={"bypass_cache"})
        payload["model"] = model

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return ChatCompletionResponse.model_validate(data)

    async def stream_async(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        model = request.model if "llama" in request.model.lower() or "mixtral" in request.model.lower() else "llama-3.3-70b-versatile"
        payload = request.model_dump(exclude={"bypass_cache"})
        payload["model"] = model
        payload["stream"] = True

        async with (
            httpx.AsyncClient(timeout=20.0) as client,
            client.stream("POST", url, json=payload, headers=headers) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield f"{line}\n\n"


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: str):
        super().__init__(name="anthropic", cost_per_1k_input=0.003, cost_per_1k_output=0.015)
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"

    async def complete_async(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        # Convert OpenAI messages to Anthropic format
        system_text = request.get_system_prompt()
        messages_payload = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": messages_payload,
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature,
        }
        if system_text:
            payload["system"] = system_text

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [{}])[0].get("text", "")

            return ChatCompletionResponse(
                id=data.get("id", f"chatcmpl-claude-{uuid.uuid4().hex[:8]}"),
                model=data.get("model", "claude-3-5-sonnet"),
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=content),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=data.get("usage", {}).get("input_tokens", 0),
                    completion_tokens=data.get("usage", {}).get("output_tokens", 0),
                    total_tokens=data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0),
                ),
            )

    async def stream_async(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        # Fallback to completion wrapper for streaming if needed
        res = await self.complete_async(request)
        chunk_data = {
            "id": res.id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": res.model,
            "choices": [{"index": 0, "delta": {"content": res.choices[0].message.content}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"
        yield "data: [DONE]\n\n"


class GeminiProvider(BaseLLMProvider):
    """Google Gemini REST API provider."""

    def __init__(self, api_key: str):
        super().__init__(name="gemini", cost_per_1k_input=0.00035, cost_per_1k_output=0.00105)
        self.api_key = api_key

    async def complete_async(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        user_prompt = request.get_latest_user_prompt()
        payload = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": request.temperature},
        }

        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]

            return ChatCompletionResponse(
                id=f"chatcmpl-gemini-{uuid.uuid4().hex[:10]}",
                model="gemini-1.5-flash",
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=content),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=len(user_prompt.split()),
                    completion_tokens=len(content.split()),
                    total_tokens=len(user_prompt.split()) + len(content.split()),
                ),
            )

    async def stream_async(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        res = await self.complete_async(request)
        chunk_data = {
            "id": res.id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": res.model,
            "choices": [{"index": 0, "delta": {"content": res.choices[0].message.content}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"
        yield "data: [DONE]\n\n"
