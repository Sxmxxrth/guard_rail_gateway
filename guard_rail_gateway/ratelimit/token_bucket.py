"""Token Bucket rate limiting algorithm for high-throughput API gateway throttling."""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """Thread-safe Token Bucket rate limiter.
    
    Tokens are replenished continuously at a fixed fill rate up to max capacity.
    Allows bursts up to capacity while enforcing long-term rate limits.
    """

    def __init__(self, capacity: float, fill_rate: float):
        """
        Args:
            capacity: Maximum burst capacity (e.g. 60 requests).
            fill_rate: Replenishment rate in tokens per second (e.g. 1.0 token/sec for 60 RPM).
        """
        self.capacity = float(capacity)
        self.fill_rate = float(fill_rate)
        self.tokens = float(capacity)
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def _replenish(self, now: float) -> None:
        """Adds regenerated tokens based on elapsed wall-clock time."""
        delta = now - self.last_update
        if delta > 0:
            self.tokens = min(self.capacity, self.tokens + (delta * self.fill_rate))
            self.last_update = now

    def consume(self, count: float = 1.0) -> tuple[bool, float]:
        """Attempts to consume tokens from the bucket.
        
        Returns:
            Tuple of (is_allowed: bool, retry_after_seconds: float)
        """
        with self._lock:
            now = time.monotonic()
            self._replenish(now)

            if self.tokens >= count:
                self.tokens -= count
                return True, 0.0

            # Calculate time needed to accumulate the missing tokens
            deficit = count - self.tokens
            retry_after = deficit / self.fill_rate if self.fill_rate > 0 else 1.0
            return False, round(retry_after, 2)

    def get_available_tokens(self) -> float:
        """Returns the current available token balance."""
        with self._lock:
            self._replenish(time.monotonic())
            return round(self.tokens, 2)

    def reset(self) -> None:
        """Resets bucket back to full capacity."""
        with self._lock:
            self.tokens = self.capacity
            self.last_update = time.monotonic()
