"""Tenant-aware rate limit manager enforcing dual RPM and TPM quotas."""

from __future__ import annotations

import threading

from guard_rail_gateway.config import Settings, get_config
from guard_rail_gateway.models import RateLimitStatus
from guard_rail_gateway.ratelimit.token_bucket import TokenBucket


class RateLimitManager:
    """Manages independent Token Buckets for Requests Per Minute (RPM) and Tokens Per Minute (TPM)."""

    def __init__(
        self,
        default_rpm: int = 60,
        default_tpm: int = 60000,
        enabled: bool = True,
    ):
        self.default_rpm = default_rpm
        self.default_tpm = default_tpm
        self.enabled = enabled

        # client_id -> (request_bucket, token_bucket)
        self._buckets: dict[str, tuple[TokenBucket, TokenBucket]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: Settings | None = None) -> RateLimitManager:
        cfg = config or get_config()
        return cls(
            default_rpm=cfg.default_rpm,
            default_tpm=cfg.default_tpm,
            enabled=cfg.rate_limit_enabled,
        )

    def _get_or_create_buckets(self, client_id: str) -> tuple[TokenBucket, TokenBucket]:
        """Retrieves or initializes rate limiter buckets for a client identifier."""
        with self._lock:
            if client_id not in self._buckets:
                # Fill rate: capacity / 60 seconds
                rpm_fill_rate = float(self.default_rpm) / 60.0
                tpm_fill_rate = float(self.default_tpm) / 60.0

                req_bucket = TokenBucket(capacity=float(self.default_rpm), fill_rate=rpm_fill_rate)
                tok_bucket = TokenBucket(capacity=float(self.default_tpm), fill_rate=tpm_fill_rate)
                self._buckets[client_id] = (req_bucket, tok_bucket)

            return self._buckets[client_id]

    def check_rate_limit(
        self,
        client_id: str = "default_client",
        estimated_tokens: int = 50,
    ) -> RateLimitStatus:
        """Evaluates whether an incoming request violates RPM or TPM limits.
        
        Args:
            client_id: Tenant or API key identifier.
            estimated_tokens: Expected token consumption for prompt + completion.
            
        Returns:
            RateLimitStatus object indicating allowed/throttled status.
        """
        if not self.enabled:
            return RateLimitStatus(
                is_allowed=True,
                current_rpm=0,
                limit_rpm=self.default_rpm,
                current_tpm=0,
                limit_tpm=self.default_tpm,
                retry_after_seconds=0.0,
            )

        req_bucket, tok_bucket = self._get_or_create_buckets(client_id)

        # 1. Attempt to consume 1 request token
        req_allowed, req_retry = req_bucket.consume(1.0)
        if not req_allowed:
            return RateLimitStatus(
                is_allowed=False,
                current_rpm=int(req_bucket.get_available_tokens()),
                limit_rpm=self.default_rpm,
                current_tpm=int(tok_bucket.get_available_tokens()),
                limit_tpm=self.default_tpm,
                retry_after_seconds=req_retry,
            )

        # 2. Attempt to consume estimated token count
        tok_allowed, tok_retry = tok_bucket.consume(float(estimated_tokens))
        if not tok_allowed:
            # Refund request token since token quota was exceeded
            req_bucket.consume(-1.0)
            return RateLimitStatus(
                is_allowed=False,
                current_rpm=int(req_bucket.get_available_tokens()),
                limit_rpm=self.default_rpm,
                current_tpm=int(tok_bucket.get_available_tokens()),
                limit_tpm=self.default_tpm,
                retry_after_seconds=tok_retry,
            )

        return RateLimitStatus(
            is_allowed=True,
            current_rpm=int(req_bucket.get_available_tokens()),
            limit_rpm=self.default_rpm,
            current_tpm=int(tok_bucket.get_available_tokens()),
            limit_tpm=self.default_tpm,
            retry_after_seconds=0.0,
        )

    def reset_client(self, client_id: str) -> None:
        """Resets rate limit quotas for a specific client."""
        with self._lock:
            if client_id in self._buckets:
                req_b, tok_b = self._buckets[client_id]
                req_b.reset()
                tok_b.reset()

    def clear(self) -> None:
        """Clears all client bucket records."""
        with self._lock:
            self._buckets.clear()
