"""Abstract base class and contract for all OmniGuard guardrails."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from guard_rail_gateway.models import GuardrailVerdict


class BaseGuardrail(ABC):
    """Abstract interface defining the execution contract for individual guardrail modules."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def evaluate(self, text: str, context: dict[str, Any] | None = None) -> GuardrailVerdict:
        """Evaluates input or output text against guardrail rules.
        
        Args:
            text: Text to evaluate (user prompt or model response).
            context: Optional contextual parameters (e.g. conversation history, system prompt).
            
        Returns:
            GuardrailVerdict indicating safe/blocked status, sanitized text, and findings.
        """

    def run(self, text: str, context: dict[str, Any] | None = None) -> GuardrailVerdict:
        """Executes the guardrail with latency benchmarking and enable/disable checks."""
        if not self.enabled:
            return GuardrailVerdict(
                is_safe=True,
                original_text=text,
                sanitized_text=text,
                latency_ms=0.0,
            )

        start = time.perf_counter()
        verdict = self.evaluate(text, context)
        verdict.latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return verdict
