"""Toxicity, harassment, and harmful content guardrail."""

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


class ToxicityHarmGuardrail(BaseGuardrail):
    """Monitors and intercepts disallowed harm categories: cyberattacks, violence, hate speech."""

    HARM_RULES: ClassVar[list[tuple[str, str, ThreatSeverity]]] = [
        # Cyberattack & Malware Vectors
        (
            r"\b(write|create|code|generate)\s+(a\s+)?(ransomware|keylogger|trojan|rootkit|ddos\s+botnet|zero-day\s+exploit)\b",
            "cyberattack_malware_generation",
            ThreatSeverity.CRITICAL,
        ),
        (
            r"\bhow\s+to\s+(hack|infiltrate|breach|exploit|defy)\s+(into\s+)?(database|server|firewall|bank|infrastructure)\b",
            "cyberattack_unauthorized_intrusion",
            ThreatSeverity.HIGH,
        ),
        (
            r"\bbypass\s+(waf|firewall|antivirus|edr)\s+detection\s+payload\b",
            "cyberattack_evasion",
            ThreatSeverity.HIGH,
        ),
        # Violence & Weaponry
        (
            r"\bhow\s+to\s+(make|synthesize|build|construct)\s+(a\s+)?(bomb|explosive|biological\s+weapon|chemical\s+weapon)\b",
            "harm_dangerous_weapons",
            ThreatSeverity.CRITICAL,
        ),
        # Self-Harm & Suicide
        (
            r"\b(ways|instructions|methods)\s+to\s+(commit\s+suicide|kill\s+myself|end\s+my\s+life|self-harm)\b",
            "harm_self_harm",
            ThreatSeverity.CRITICAL,
        ),
        # Harassment & Slurs
        (
            r"\b(kill\s+all|eradicate\s+all)\s+(people|members|races)\b",
            "harm_hate_speech",
            ThreatSeverity.CRITICAL,
        ),
    ]

    def __init__(
        self,
        name: str = "ToxicityHarmGuardrail",
        action: Literal["block", "flag"] = "block",
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.action = action

    def evaluate(self, text: str, context: dict[str, Any] | None = None) -> GuardrailVerdict:
        """Scans text against harmful and prohibited intent categories."""
        findings: list[SecurityFinding] = []

        for pattern, rule_name, severity in self.HARM_RULES:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                findings.append(
                    SecurityFinding(
                        category=ThreatCategory.HARMFUL_CONTENT,
                        severity=severity,
                        rule_name=rule_name,
                        description=f"Disallowed harm category detected: {rule_name.replace('_', ' ')}",
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
        else:
            return GuardrailVerdict(
                status=GuardrailStatus.FLAGGED,
                is_safe=True,
                original_text=text,
                sanitized_text=text,
                findings=findings,
            )
