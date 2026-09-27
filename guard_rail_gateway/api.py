"""FastAPI reverse proxy exposing OpenAI-compatible endpoints with guardrails and caching."""

from __future__ import annotations

import json
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from guard_rail_gateway.cache.semantic_cache import SemanticCache
from guard_rail_gateway.config import get_config
from guard_rail_gateway.guardrails.pipeline import GuardrailPipeline
from guard_rail_gateway.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    GatewayTelemetry,
    UsageInfo,
)
from guard_rail_gateway.ratelimit.manager import RateLimitManager
from guard_rail_gateway.router.resilient_router import ResilientRouter
from guard_rail_gateway.telemetry import get_telemetry

# Initialize FastAPI Application
app = FastAPI(
    title="OmniGuard AI Gateway",
    description="Enterprise OpenAI-Compatible LLM Gateway with Semantic Caching & Multi-Layer Guardrails",
    version="0.1.0",
)

# CORS middleware for Web SDKs and Streamlit UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Gateway Subsystems
_config = get_config()
_semantic_cache = SemanticCache(
    similarity_threshold=_config.semantic_cache_threshold,
    max_entries=_config.cache_max_entries,
    default_ttl_seconds=_config.cache_ttl_seconds,
)
_guardrail_pipeline = GuardrailPipeline(config=_config)
_rate_limiter = RateLimitManager.from_config(_config)
_router = ResilientRouter(config=_config)
_telemetry = get_telemetry()


def get_client_id(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> str:
    """Extracts client identity from Authorization header or X-API-Key."""
    if isinstance(x_api_key, str) and x_api_key.strip():
        return x_api_key.strip()
    if isinstance(authorization, str) and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        return token[:16] if token else "anonymous_client"
    return "anonymous_client"


# =============================================================================
# OpenAI-Compatible Endpoints
# =============================================================================

@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["Chat Completions"],
    summary="OpenAI-compatible chat completions proxy with guardrails and cache",
)
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    client_id: str = Depends(get_client_id),
):
    """Proxies OpenAI chat completions, evaluating security guardrails and two-tier semantic cache."""
    start_time = time.perf_counter()
    user_prompt = request.get_latest_user_prompt()
    p_tokens_est = max(1, len(user_prompt.split()))

    # 1. Rate Limiting Check (Token Bucket RPM & TPM)
    rl_status = _rate_limiter.check_rate_limit(client_id, estimated_tokens=p_tokens_est + 100)
    if not rl_status.is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for client '{client_id}'. Try again in {rl_status.retry_after_seconds}s.",
            headers={"Retry-After": str(int(rl_status.retry_after_seconds) + 1)},
        )

    # 2. Multi-Layer Guardrail Pipeline (Injection, Toxicity, PII)
    guard_context = {"system_prompt": request.get_system_prompt(), "client_id": client_id}
    input_verdict = _guardrail_pipeline.evaluate_input(user_prompt, context=guard_context)

    # Log security findings if any
    for finding in input_verdict.findings:
        _telemetry.log_incident(
            finding,
            client_id=client_id,
            endpoint="/v1/chat/completions",
            latency_ms=input_verdict.latency_ms,
        )

    if not input_verdict.is_safe:
        _telemetry.record_request(
            latency_ms=input_verdict.latency_ms,
            was_blocked=True,
            was_sanitized=False,
            cache_hit=False,
        )
        # Blocked by security guardrails
        block_msg = (
            f"I cannot fulfill this request because it violates safety policy: "
            f"{input_verdict.findings[0].description}."
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-blocked-{uuid.uuid4().hex[:8]}",
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=block_msg),
                    finish_reason="guardrail_block",
                )
            ],
            usage=UsageInfo(prompt_tokens=p_tokens_est, completion_tokens=len(block_msg.split()), total_tokens=p_tokens_est + len(block_msg.split())),
            omniguard_telemetry=GatewayTelemetry(
                threats_detected=len(input_verdict.findings),
                guardrails_applied=["PromptInjectionGuardrail", "ToxicityHarmGuardrail", "PIISanitizerGuardrail"],
                total_latency_ms=input_verdict.latency_ms,
            ),
        )

    # Use sanitized prompt for downstream processing
    safe_prompt = input_verdict.sanitized_text or user_prompt
    pii_token_map = input_verdict.pii_token_map

    # 3. Two-Tier Semantic Cache Lookup
    if not request.bypass_cache and _config.cache_enabled:
        cache_match = _semantic_cache.lookup(safe_prompt, model=request.model)
        if cache_match.is_hit and cache_match.entry is not None:
            # Reversibly restore any PII placeholders
            restored_content = _guardrail_pipeline.evaluate_output(
                cache_match.entry.response_content,
                pii_token_map=pii_token_map,
                context=guard_context,
            ).sanitized_text or cache_match.entry.response_content

            total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            _telemetry.record_request(
                latency_ms=total_elapsed_ms,
                was_blocked=False,
                was_sanitized=bool(pii_token_map),
                cache_hit=True,
            )

            # Check if streaming requested
            if request.stream:
                async def stream_cached():
                    chunk_id = f"chatcmpl-cache-{uuid.uuid4().hex[:8]}"
                    words = restored_content.split(" ")
                    for i, w in enumerate(words):
                        delta = w if i == 0 else f" {w}"
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None if i < len(words) - 1 else "stop"}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(stream_cached(), media_type="text/event-stream")

            c_tokens = max(1, len(restored_content.split()))
            return ChatCompletionResponse(
                id=f"chatcmpl-cache-{uuid.uuid4().hex[:8]}",
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=restored_content),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=p_tokens_est,
                    completion_tokens=c_tokens,
                    total_tokens=p_tokens_est + c_tokens,
                ),
                omniguard_telemetry=GatewayTelemetry(
                    cache_hit=True,
                    cache_match_type=cache_match.match_type,
                    cache_similarity=cache_match.similarity_score,
                    pii_entities_sanitized=len(pii_token_map),
                    provider_used="semantic_cache",
                    total_latency_ms=round(total_elapsed_ms, 2),
                    estimated_cost_saved_usd=cache_match.entry.estimated_cost_saved_usd,
                ),
            )

    # 4. Prepare sanitized request for upstream model
    clean_messages = []
    for msg in request.messages:
        if msg.role == "user" and msg.content == user_prompt:
            clean_messages.append(ChatMessage(role=msg.role, content=safe_prompt))
        else:
            clean_messages.append(msg)

    sanitized_request = request.model_copy(update={"messages": clean_messages})

    # 5. Handle Streaming Request
    if request.stream:
        async def stream_upstream():
            async for chunk in _router.route_and_stream_async(sanitized_request):
                yield chunk

        _telemetry.record_request(
            latency_ms=30.0,
            was_blocked=False,
            was_sanitized=bool(pii_token_map),
            cache_hit=False,
        )
        return StreamingResponse(stream_upstream(), media_type="text/event-stream")

    # 6. Non-Streaming Upstream Route & Complete
    upstream_start = time.perf_counter()
    upstream_resp, provider_name, failover_occurred = await _router.route_and_complete_async(sanitized_request)
    upstream_ms = (time.perf_counter() - upstream_start) * 1000.0

    raw_response_text = upstream_resp.choices[0].message.content

    # 7. Output Guardrail & Reversible PII Restoration
    out_verdict = _guardrail_pipeline.evaluate_output(
        raw_response_text,
        pii_token_map=pii_token_map,
        context=guard_context,
    )
    final_content = out_verdict.sanitized_text or raw_response_text

    # 8. Store in Semantic Cache
    if _config.cache_enabled:
        _semantic_cache.store(
            prompt=safe_prompt,
            response_content=raw_response_text,
            model=request.model,
            tokens_saved=upstream_resp.usage.total_tokens,
            cost_saved_usd=_semantic_cache.cost_per_1k * (upstream_resp.usage.total_tokens / 1000.0),
        )

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0
    _telemetry.record_request(
        latency_ms=total_latency_ms,
        was_blocked=False,
        was_sanitized=bool(pii_token_map),
        cache_hit=False,
    )

    upstream_resp.choices[0].message.content = final_content
    upstream_resp.omniguard_telemetry = GatewayTelemetry(
        cache_hit=False,
        cache_match_type="none",
        pii_entities_sanitized=len(pii_token_map),
        provider_used=provider_name,
        failover_occurred=failover_occurred,
        total_latency_ms=round(total_latency_ms, 2),
        upstream_latency_ms=round(upstream_ms, 2),
    )

    return upstream_resp


@app.get("/v1/models", tags=["Models"])
async def list_models():
    """Returns available models supported by the gateway."""
    models_list = [
        {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
        {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
        {"id": "claude-3-5-sonnet", "object": "model", "owned_by": "anthropic"},
        {"id": "gemini-1.5-flash", "object": "model", "owned_by": "google"},
        {"id": "llama-3.3-70b-versatile", "object": "model", "owned_by": "groq"},
        {"id": "mock-gpt-4o", "object": "model", "owned_by": "omniguard"},
    ]
    return {"object": "list", "data": models_list}


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint providing status of gateway subsystems."""
    return {
        "status": "healthy",
        "service": "OmniGuard AI Gateway",
        "version": "0.1.0",
        "active_provider": _config.default_provider,
        "routing_strategy": _config.routing_strategy,
        "cache_enabled": _config.cache_enabled,
        "cached_entries": len(_semantic_cache.exact_cache),
        "guardrails_active": ["PromptInjectionGuardrail", "ToxicityHarmGuardrail", "PIISanitizerGuardrail", "OutputSafetyGuardrail"],
    }


@app.get("/metrics", tags=["System"])
async def get_metrics():
    """Returns operational telemetry, cache performance, and threat statistics."""
    telemetry_summary = _telemetry.get_summary()
    cache_summary = _semantic_cache.get_metrics()
    return {
        "telemetry": telemetry_summary,
        "cache": cache_summary,
    }


@app.get("/v1/incidents", tags=["Security"])
async def get_incidents(limit: int = 50):
    """Returns audit log of intercepted security incidents and sanitized PII events."""
    incidents = _telemetry.get_recent_incidents(limit=limit)
    return {"object": "list", "count": len(incidents), "data": [inc.model_dump() for inc in incidents]}


@app.post("/v1/cache/clear", tags=["Cache"])
async def clear_cache():
    """Flushes exact and vector semantic caches."""
    _semantic_cache.clear()
    return {"status": "success", "message": "Semantic cache successfully purged."}
