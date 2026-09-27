"""Core Pydantic data models for OmniGuard AI Gateway, Guardrails, and Cache."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# OpenAI-Compatible Chat Models
# =============================================================================

class ChatMessage(BaseModel):
    """Represents a single message in a conversational thread."""
    model_config = ConfigDict(validate_assignment=True)

    role: Literal["system", "user", "assistant", "function", "tool"] = Field(
        ...,
        description="Role of the message sender",
    )
    content: str = Field(..., description="Text content of the message")
    name: str | None = Field(default=None, description="Optional name identifier for participant")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible request body for chat completion endpoint."""
    model_config = ConfigDict(validate_assignment=True, extra="allow")

    model: str = Field(default="gpt-4o", description="Target model ID or alias")
    messages: list[ChatMessage] = Field(..., min_length=1, description="List of conversation messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=10)
    stream: bool = Field(default=False, description="Whether to stream response tokens via SSE")
    max_tokens: int | None = Field(default=None, ge=1)
    user: str | None = Field(default=None, description="Client user or tenant identifier")
    bypass_cache: bool = Field(default=False, description="When True, forces upstream query bypass")

    def get_latest_user_prompt(self) -> str:
        """Extracts the most recent user prompt in the conversation."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return self.messages[-1].content if self.messages else ""

    def get_system_prompt(self) -> str:
        """Extracts system instructions if present."""
        for msg in self.messages:
            if msg.role == "system":
                return msg.content
        return ""


class ChatCompletionChoice(BaseModel):
    """A single choice option returned by chat completion."""
    index: int = Field(default=0)
    message: ChatMessage
    finish_reason: Literal["stop", "length", "guardrail_block", "content_filter"] = "stop"


class UsageInfo(BaseModel):
    """Token usage telemetry."""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class GatewayTelemetry(BaseModel):
    """OmniGuard gateway execution telemetry attached to responses."""
    request_id: str = Field(default_factory=lambda: f"req-{uuid.uuid4().hex[:10]}")
    cache_hit: bool = False
    cache_match_type: Literal["none", "exact_hash", "vector_semantic"] = "none"
    cache_similarity: float = 0.0
    guardrails_applied: list[str] = Field(default_factory=list)
    threats_detected: int = 0
    pii_entities_sanitized: int = 0
    provider_used: str = "mock"
    failover_occurred: bool = False
    total_latency_ms: float = 0.0
    upstream_latency_ms: float = 0.0
    cost_usd: float = 0.0
    estimated_cost_saved_usd: float = 0.0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response object."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "gpt-4o"
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    system_fingerprint: str | None = "fp_omniguard_v1"
    omniguard_telemetry: GatewayTelemetry | None = None


# =============================================================================
# Guardrail & Threat Models
# =============================================================================

class GuardrailStatus(str, Enum):
    """Execution status resulting from guardrail evaluation."""
    PASSED = "passed"
    BLOCKED = "blocked"
    FLAGGED = "flagged"
    SANITIZED = "sanitized"


class ThreatSeverity(str, Enum):
    """Severity classification for intercepted security findings."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatCategory(str, Enum):
    """Standardized threat categories monitored by OmniGuard."""
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    PII_LEAK = "pii_leak"
    SECRET_KEY = "secret_key"
    TOXICITY = "toxicity"
    HARMFUL_CONTENT = "harmful_content"
    SYSTEM_LEAK = "system_prompt_leak"
    HALLUCINATION = "hallucination"


class SecurityFinding(BaseModel):
    """Details of a specific security rule violation or warning."""
    category: ThreatCategory
    severity: ThreatSeverity
    rule_name: str
    description: str
    matched_text: str | None = None
    start_idx: int | None = None
    end_idx: int | None = None
    mitigation_action: str = "blocked"


class GuardrailVerdict(BaseModel):
    """Aggregated verdict returned by the guardrail pipeline."""
    status: GuardrailStatus = GuardrailStatus.PASSED
    is_safe: bool = True
    original_text: str = ""
    sanitized_text: str | None = None
    findings: list[SecurityFinding] = Field(default_factory=list)
    pii_token_map: dict[str, str] = Field(
        default_factory=dict,
        description="Maps placeholder token -> original PII value for reversible restoration",
    )
    latency_ms: float = 0.0


# =============================================================================
# Semantic Cache Models
# =============================================================================

class CacheEntry(BaseModel):
    """Cached response item stored in the semantic cache."""
    key: str = Field(..., description="Unique entry identifier / prompt hash")
    prompt: str = Field(..., description="Original user prompt text")
    prompt_hash: str = Field(..., description="SHA-256 normalized hash")
    response_content: str = Field(..., description="Cached assistant response")
    model: str = "gpt-4o"
    created_at: float = Field(default_factory=time.time)
    expires_at: float
    access_count: int = 0
    last_accessed: float = Field(default_factory=time.time)
    tokens_saved: int = 0
    estimated_cost_saved_usd: float = 0.0
    vector: list[float] | None = None

    def is_expired(self) -> bool:
        """Checks if cache entry has exceeded its TTL."""
        return time.time() > self.expires_at


class CacheMatchResult(BaseModel):
    """Result of searching the cache for a given query."""
    is_hit: bool = False
    match_type: Literal["none", "exact_hash", "vector_semantic"] = "none"
    similarity_score: float = 0.0
    entry: CacheEntry | None = None
    lookup_latency_ms: float = 0.0


# =============================================================================
# Rate Limiting Models
# =============================================================================

class RateLimitStatus(BaseModel):
    """Rate limit quota status for a specific tenant or API key."""
    is_allowed: bool = True
    current_rpm: int = 0
    limit_rpm: int = 60
    current_tpm: int = 0
    limit_tpm: int = 60000
    retry_after_seconds: float = 0.0


# =============================================================================
# Security Incident Audit Model
# =============================================================================

class SecurityIncident(BaseModel):
    """Audit log item representing an intercepted threat or sanitized PII event."""
    incident_id: str = Field(default_factory=lambda: f"inc-{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    client_id: str = "client-default"
    endpoint: str = "/v1/chat/completions"
    category: ThreatCategory
    severity: ThreatSeverity
    trigger_rule: str
    action_taken: str
    sample_snippet: str
    latency_overhead_ms: float = 0.0
