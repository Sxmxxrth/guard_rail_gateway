"""Output validation guardrail verifying response integrity and preventing system leaks."""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal

from guard_rail_gateway.guardrails.base import BaseGuardrail
from guard_rail_gateway.models import (
    GuardrailStatus,
    GuardrailVerdict,
    SecurityFinding,
    ThreatCategory,
    ThreatSeverity,
)


class OutputSafetyGuardrail(BaseGuardrail):
    """Inspects upstream LLM responses to prevent system prompt leakage and privacy violations."""

    LEAK_SIGNATURES: ClassVar[list[tuple[str, str, ThreatSeverity]]] = [
        (
            r"\b(my\s+system\s+prompt\s+is|here\s+are\s+my\s+initial\s+instructions|the\s+system\s+instructions\s+given\s+to\s+me)\b",
            "system_prompt_leak_explicit",
            ThreatSeverity.HIGH,
        ),
        (
            r"<\|im_start\|>system|<\|system\|>",
            "system_token_leak",
            ThreatSeverity.HIGH,
        ),
        (
            r"\b(CONFIDENTIAL_INTERNAL_SYSTEM_SECRET|ROOT_ACCESS_GRANTED|DEBUG_BYPASS_MODE)\b",
            "internal_secret_leak",
            ThreatSeverity.CRITICAL,
        ),
    ]

    def __init__(
        self,
        name: str = "OutputSafetyGuardrail",
        action: Literal["block", "flag", "sanitize"] = "sanitize",
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.action = action

    def evaluate(self, text: str, context: dict[str, Any] | None = None) -> GuardrailVerdict:
        """Evaluates model completion against leakage and sensitive output rules."""
        findings: list[SecurityFinding] = []
        sanitized_text = text

        # 1. Check if output verbatim repeats a secret system prompt provided in context
        system_prompt = (context or {}).get("system_prompt", "")
        if system_prompt and len(system_prompt) > 30 and system_prompt in text:
            findings.append(
                SecurityFinding(
                    category=ThreatCategory.SYSTEM_LEAK,
                    severity=ThreatSeverity.CRITICAL,
                    rule_name="verbatim_system_prompt_exposure",
                    description="Upstream LLM output verbatim leaks the hidden system prompt",
                    matched_text=system_prompt[:50] + "...",
                    mitigation_action=self.action,
                )
            )
            sanitized_text = sanitized_text.replace(
                system_prompt,
                "[REDACTED BY OMNIGUARD: System instructions protected]",
            )

        # 2. Check leakage signatures
        for pattern, rule_name, severity in self.LEAK_SIGNATURES:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                findings.append(
                    SecurityFinding(
                        category=ThreatCategory.SYSTEM_LEAK,
                        severity=severity,
                        rule_name=rule_name,
                        description=f"Potential system prompt leakage pattern: '{match.group(0)}'",
                        matched_text=match.group(0),
                        start_idx=match.start(),
                        end_idx=match.end(),
                        mitigation_action=self.action,
                    )
                )

        if not findings:
            return GuardrailVerdict(
                status=GuardrailStatus.PASSED,
                is_safe=True,
                original_text=text,
                sanitized_text=text,
                findings=[],
            )

        if self.action == "block":
            return GuardrailVerdict(
                status=GuardrailStatus.BLOCKED,
                is_safe=False,
                original_text=text,
                sanitized_text=None,
                findings=findings,
            )
        elif self.action == "flag":
            return GuardrailVerdict(
                status=GuardrailStatus.FLAGGED,
                is_safe=True,
                original_text=text,
                sanitized_text=text,
                findings=findings,
            )
        else:  # sanitize
            return GuardrailVerdict(
                status=GuardrailStatus.SANITIZED,
                is_safe=True,
                original_text=text,
                sanitized_text=sanitized_text,
                findings=findings,
            )
