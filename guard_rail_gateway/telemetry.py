"""Observability, security incident logging, and OpenTelemetry tracking."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from guard_rail_gateway.models import (
    SecurityFinding,
    SecurityIncident,
    ThreatCategory,
)


class TelemetryManager:
    """Thread-safe telemetry aggregator and security incident audit store."""

    def __init__(self, max_incident_history: int = 500):
        self.max_history = max_incident_history
        self._incidents: deque[SecurityIncident] = deque(maxlen=max_incident_history)
        self._lock = threading.Lock()

        # Telemetry Aggregation Counters
        self.total_requests: int = 0
        self.blocked_requests: int = 0
        self.sanitized_requests: int = 0
        self.cache_hits: int = 0
        self.total_latency_ms: float = 0.0
        self.category_counts: dict[str, int] = {cat.value: 0 for cat in ThreatCategory}
        self.start_timestamp: float = time.time()

    def record_request(
        self,
        latency_ms: float,
        was_blocked: bool = False,
        was_sanitized: bool = False,
        cache_hit: bool = False,
    ) -> None:
        """Records metrics for an individual proxy request."""
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            if was_blocked:
                self.blocked_requests += 1
            if was_sanitized:
                self.sanitized_requests += 1
            if cache_hit:
                self.cache_hits += 1

    def log_incident(
        self,
        finding: SecurityFinding,
        client_id: str = "client-default",
        endpoint: str = "/v1/chat/completions",
        latency_ms: float = 0.0,
    ) -> SecurityIncident:
        """Appends a detected security event to the audit log."""
        with self._lock:
            cat_key = finding.category.value
            self.category_counts[cat_key] = self.category_counts.get(cat_key, 0) + 1

            incident = SecurityIncident(
                client_id=client_id,
                endpoint=endpoint,
                category=finding.category,
                severity=finding.severity,
                trigger_rule=finding.rule_name,
                action_taken=finding.mitigation_action,
                sample_snippet=finding.matched_text or finding.description[:50],
                latency_overhead_ms=round(latency_ms, 2),
            )
            self._incidents.appendleft(incident)
            return incident

    def get_recent_incidents(self, limit: int = 50) -> list[SecurityIncident]:
        """Returns the most recent security incidents."""
        with self._lock:
            return list(self._incidents)[:limit]

    def get_summary(self) -> dict[str, Any]:
        """Returns high-level telemetry and security posture summary."""
        with self._lock:
            total = max(1, self.total_requests)
            avg_latency = self.total_latency_ms / total
            cache_rate = (self.cache_hits / total) * 100.0

            return {
                "uptime_seconds": round(time.time() - self.start_timestamp, 1),
                "total_requests": self.total_requests,
                "blocked_requests": self.blocked_requests,
                "sanitized_requests": self.sanitized_requests,
                "cache_hits": self.cache_hits,
                "cache_hit_ratio_percent": round(cache_rate, 1),
                "avg_latency_ms": round(avg_latency, 2),
                "threat_category_breakdown": dict(self.category_counts),
                "total_incidents_recorded": len(self._incidents),
            }

    def clear(self) -> None:
        """Clears all telemetry counters and incident history."""
        with self._lock:
            self._incidents.clear()
            self.total_requests = 0
            self.blocked_requests = 0
            self.sanitized_requests = 0
            self.cache_hits = 0
            self.total_latency_ms = 0.0
            self.category_counts = {cat.value: 0 for cat in ThreatCategory}


# Global singleton instance
_telemetry_instance: TelemetryManager | None = None


def get_telemetry() -> TelemetryManager:
    """Returns singleton telemetry manager."""
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = TelemetryManager()
    return _telemetry_instance
