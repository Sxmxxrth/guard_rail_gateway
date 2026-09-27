"""Circuit Breaker pattern preventing cascading upstream provider failures."""

from __future__ import annotations

import threading
import time
from enum import Enum


class CircuitState(str, Enum):
    """Lifecycle states of a circuit breaker."""
    CLOSED = "closed"        # Healthy, requests allowed
    OPEN = "open"            # Unhealthy, requests fail-fast
    HALF_OPEN = "half_open"  # Canary testing for recovery


class CircuitBreaker:
    """Thread-safe circuit breaker monitoring provider error rates and recovery intervals."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.last_state_change = time.monotonic()
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        """Determines if a request should be dispatched to this provider."""
        with self._lock:
            now = time.monotonic()
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if (now - self.last_state_change) > self.recovery_timeout:
                    # Transition to half-open for canary request
                    self.state = CircuitState.HALF_OPEN
                    self.last_state_change = now
                    return True
                return False

            return self.state == CircuitState.HALF_OPEN

    def record_success(self) -> None:
        """Records a successful request, resetting failures and closing half-open circuits."""
        with self._lock:
            self.failure_count = 0
            if self.state != CircuitState.CLOSED:
                self.state = CircuitState.CLOSED
                self.last_state_change = time.monotonic()

    def record_failure(self) -> None:
        """Records an execution error, tripping the circuit if failure threshold is reached."""
        with self._lock:
            now = time.monotonic()
            self.failure_count += 1
            self.last_failure_time = now

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.last_state_change = now

    def reset(self) -> None:
        """Manually forces the circuit breaker back to CLOSED state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_state_change = time.monotonic()
