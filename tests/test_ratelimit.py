"""Tests for OmniGuard Token Bucket and Dual RPM/TPM Rate Limiter."""

from __future__ import annotations

import concurrent.futures
import time

from guard_rail_gateway.config import Settings
from guard_rail_gateway.ratelimit.manager import RateLimitManager
from guard_rail_gateway.ratelimit.token_bucket import TokenBucket

# =============================================================================
# Token Bucket Algorithm Tests
# =============================================================================

def test_token_bucket_initial_capacity():
    bucket = TokenBucket(capacity=10.0, fill_rate=1.0)
    assert bucket.get_available_tokens() == 10.0

    allowed, retry_after = bucket.consume(3.0)
    assert allowed is True
    assert retry_after == 0.0
    assert 6.9 <= bucket.get_available_tokens() <= 7.1


def test_token_bucket_exhaustion():
    bucket = TokenBucket(capacity=5.0, fill_rate=1.0)
    allowed, retry_after = bucket.consume(5.0)
    assert allowed is True
    assert retry_after == 0.0

    # Over-consumption attempt
    allowed_second, retry_after_second = bucket.consume(2.0)
    assert allowed_second is False
    assert retry_after_second > 0.0


def test_token_bucket_replenish():
    # Fill rate 10 tokens/sec
    bucket = TokenBucket(capacity=5.0, fill_rate=10.0)
    allowed, _ = bucket.consume(5.0)
    assert allowed is True

    # Sleep 0.2s -> adds ~2 tokens
    time.sleep(0.22)
    assert bucket.get_available_tokens() >= 1.5

    allowed_after, _ = bucket.consume(1.0)
    assert allowed_after is True


def test_token_bucket_reset():
    bucket = TokenBucket(capacity=10.0, fill_rate=1.0)
    bucket.consume(10.0)
    assert bucket.get_available_tokens() < 1.0

    bucket.reset()
    assert bucket.get_available_tokens() == 10.0


def test_token_bucket_concurrency():
    bucket = TokenBucket(capacity=50.0, fill_rate=0.0)
    consumed_count = 0

    def worker():
        nonlocal consumed_count
        allowed, _ = bucket.consume(1.0)
        if allowed:
            consumed_count += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker) for _ in range(70)]
        concurrent.futures.wait(futures)

    # Exactly 50 calls should succeed, remaining 20 should fail
    assert consumed_count == 50
    assert bucket.get_available_tokens() < 1.0


# =============================================================================
# Dual RPM & TPM RateLimitManager Tests
# =============================================================================

def test_rate_limit_manager_normal_flow():
    mgr = RateLimitManager(default_rpm=10, default_tpm=1000)
    status = mgr.check_rate_limit("client_a", estimated_tokens=100)

    assert status.is_allowed is True
    assert status.retry_after_seconds == 0.0
    assert status.limit_rpm == 10
    assert status.limit_tpm == 1000


def test_rate_limit_manager_rpm_exhaustion():
    mgr = RateLimitManager(default_rpm=3, default_tpm=10000)

    # First 3 should pass
    for _ in range(3):
        res = mgr.check_rate_limit("tenant_1", estimated_tokens=10)
        assert res.is_allowed is True

    # 4th should be throttled by RPM
    res_4 = mgr.check_rate_limit("tenant_1", estimated_tokens=10)
    assert res_4.is_allowed is False
    assert res_4.retry_after_seconds > 0.0


def test_rate_limit_manager_tpm_exhaustion_with_refund():
    mgr = RateLimitManager(default_rpm=100, default_tpm=500)

    # Consume 400 tokens -> succeeds
    res_1 = mgr.check_rate_limit("tenant_2", estimated_tokens=400)
    assert res_1.is_allowed is True

    # Request 300 tokens -> exceeds 500 TPM quota
    res_2 = mgr.check_rate_limit("tenant_2", estimated_tokens=300)
    assert res_2.is_allowed is False
    assert res_2.retry_after_seconds > 0.0


def test_rate_limit_manager_tenant_isolation():
    mgr = RateLimitManager(default_rpm=2, default_tpm=1000)

    # Exhaust tenant A
    mgr.check_rate_limit("tenant_a", estimated_tokens=10)
    mgr.check_rate_limit("tenant_a", estimated_tokens=10)
    assert mgr.check_rate_limit("tenant_a", estimated_tokens=10).is_allowed is False

    # Tenant B should still be unaffected
    assert mgr.check_rate_limit("tenant_b", estimated_tokens=10).is_allowed is True


def test_rate_limit_manager_reset_and_clear():
    mgr = RateLimitManager(default_rpm=2, default_tpm=1000)
    mgr.check_rate_limit("tenant_c", estimated_tokens=10)
    mgr.check_rate_limit("tenant_c", estimated_tokens=10)
    assert mgr.check_rate_limit("tenant_c", estimated_tokens=10).is_allowed is False

    mgr.reset_client("tenant_c")
    assert mgr.check_rate_limit("tenant_c", estimated_tokens=10).is_allowed is True

    mgr.clear()
    assert len(mgr._buckets) == 0


def test_rate_limit_manager_disabled():
    mgr = RateLimitManager(default_rpm=1, default_tpm=1, enabled=False)
    for _ in range(10):
        status = mgr.check_rate_limit("tenant_d", estimated_tokens=5000)
        assert status.is_allowed is True
        assert status.retry_after_seconds == 0.0


def test_rate_limit_manager_from_config():
    config = Settings(default_rpm=50, default_tpm=50000, rate_limit_enabled=True)
    mgr = RateLimitManager.from_config(config)
    assert mgr.default_rpm == 50
    assert mgr.default_tpm == 50000
    assert mgr.enabled is True
