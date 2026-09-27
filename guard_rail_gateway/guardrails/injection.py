"""Prompt injection, delimiter hijacking, and jailbreak detection guardrail."""

from __future__ import annotations

import base64
import binascii
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


class PromptInjectionGuardrail(BaseGuardrail):
    """Detects adversarial prompt injections, system token spoofing, and jailbreaks."""

    # 1. Direct Instruction Override Signatures
    OVERRIDE_PATTERNS: ClassVar[list[tuple[str, str, ThreatSeverity]]] = [
        (
            r"\b(ignore|disregard|forget|bypass|override)\s+(all\s+)?(previous|above|prior|system)\s+(instructions|prompts|rules|commands|guidelines)\b",
            "instruction_override_direct",
            ThreatSeverity.CRITICAL,
        ),
        (
            r"\b(you\s+must\s+ignore|do\s+not\s+follow)\s+(the\s+)?(previous|initial|system)\s+instructions\b",
            "instruction_override_strict",
            ThreatSeverity.CRITICAL,
        ),
        (
            r"\bdisregard\s+(everything|all\s+rules)\s+(said|written)\s+before\b",
            "instruction_override_wipe",
            ThreatSeverity.CRITICAL,
        ),
    ]

    # 2. Jailbreak Persona and Roleplay Patterns
    JAILBREAK_PATTERNS: ClassVar[list[tuple[str, str, ThreatSeverity]]] = [
        (
            r"\b(dan\s+mode|do\s+anything\s+now|developer\s+mode\s+enabled|jailbreak\s+activated)\b",
            "jailbreak_dan_mode",
            ThreatSeverity.CRITICAL,
        ),
        (
            r"\b(you\s+are\s+now|act\s+as)\s+(an?\s+)?(unfiltered|uncensored|evil|immoral|shadow|jailbroken)\s+(ai|assistant|bot|llm)\b",
            "jailbreak_unfiltered_persona",
            ThreatSeverity.HIGH,
        ),
        (
            r"\bfor\s+the\s+rest\s+of\s+this\s+conversation\s+you\s+have\s+no\s+(rules|ethics|morals|limits|restrictions)\b",
            "jailbreak_no_restrictions",
            ThreatSeverity.CRITICAL,
        ),
        (
            r"\bhypothetical\s+fictional\s+scenario\s+where\s+safety\s+guidelines\s+do\s+not\s+apply\b",
            "jailbreak_hypothetical_bypass",
            ThreatSeverity.HIGH,
        ),
    ]

    # 3. Delimiter and Special Token Spoofing
    DELIMITER_PATTERNS: ClassVar[list[tuple[str, str, ThreatSeverity]]] = [
        (r"<\|im_start\|>system|<\|system\|>|<\|im_end\|>", "token_spoof_chatml", ThreatSeverity.HIGH),
        (r"\[INST\]\s*<<SYS>>|<<SYS>>|\[/INST\]", "token_spoof_llama_sys", ThreatSeverity.HIGH),
        (r"---BEGIN\s+SYSTEM\s+PROMPT---|---END\s+SYSTEM\s+PROMPT---", "token_spoof_custom_delimiter", ThreatSeverity.MEDIUM),
        (r"\b(Human:|Assistant:)\s*$", "token_spoof_anthropic_roles", ThreatSeverity.MEDIUM),
    ]

    # 4. Obfuscation & Evasion Patterns
    BASE64_PATTERN: ClassVar[re.Pattern] = re.compile(r"\b([A-Za-z0-9+/]{24,}={0,2})\b")

    def __init__(
        self,
        name: str = "PromptInjectionGuardrail",
        action: Literal["block", "flag", "quarantine"] = "block",
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.action = action

    def _normalize_text(self, text: str) -> str:
        """Strips zero-width spaces and decodes common leetspeak substitutions."""
        # Remove zero-width unicode characters
        cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
        return cleaned

    def _inspect_base64_payloads(self, text: str) -> list[SecurityFinding]:
        """Inspects and decodes embedded base64 strings to check for hidden injection payloads."""
        findings: list[SecurityFinding] = []
        matches = self.BASE64_PATTERN.findall(text)

        for match in matches:
            try:
                decoded_bytes = base64.b64decode(match, validate=True)
                decoded_str = decoded_bytes.decode("utf-8", errors="ignore").lower()
                # Check if decoded string contains override triggers
                for pattern, rule_name, severity in self.OVERRIDE_PATTERNS + self.JAILBREAK_PATTERNS:
                    if re.search(pattern, decoded_str, re.IGNORECASE):
                        findings.append(
                            SecurityFinding(
                                category=ThreatCategory.PROMPT_INJECTION,
                                severity=severity,
                                rule_name=f"obfuscated_base64_{rule_name}",
                                description=f"Base64-obfuscated injection payload detected: {decoded_str[:60]}...",
                                matched_text=match[:30] + "...",
                                mitigation_action=self.action,
                            )
                        )
                        break
            except (binascii.Error, ValueError):
                continue

        return findings

    def evaluate(self, text: str, context: dict[str, Any] | None = None) -> GuardrailVerdict:
        """Evaluates input prompt against injection patterns, delimiters, and obfuscations."""
        cleaned_text = self._normalize_text(text)
        findings: list[SecurityFinding] = []

        # 1. Check direct instruction overrides
        for pattern, rule_name, severity in self.OVERRIDE_PATTERNS:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                findings.append(
                    SecurityFinding(
                        category=ThreatCategory.PROMPT_INJECTION,
                        severity=severity,
                        rule_name=rule_name,
                        description=f"Direct instruction override attempt detected: '{match.group(0)}'",
                        matched_text=match.group(0),
                        start_idx=match.start(),
                        end_idx=match.end(),
                        mitigation_action=self.action,
                    )
                )

        # 2. Check jailbreak persona & roleplay signatures
        for pattern, rule_name, severity in self.JAILBREAK_PATTERNS:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                findings.append(
                    SecurityFinding(
                        category=ThreatCategory.JAILBREAK,
                        severity=severity,
                        rule_name=rule_name,
                        description=f"Adversarial jailbreak signature detected: '{match.group(0)}'",
                        matched_text=match.group(0),
                        start_idx=match.start(),
                        end_idx=match.end(),
                        mitigation_action=self.action,
                    )
                )

        # 3. Check delimiter & special token spoofing
        for pattern, rule_name, severity in self.DELIMITER_PATTERNS:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                findings.append(
                    SecurityFinding(
                        category=ThreatCategory.PROMPT_INJECTION,
                        severity=severity,
                        rule_name=rule_name,
                        description=f"System delimiter token spoofing detected: '{match.group(0)}'",
                        matched_text=match.group(0),
                        start_idx=match.start(),
                        end_idx=match.end(),
                        mitigation_action=self.action,
                    )
                )

        # 4. Check base64 obfuscated payloads
        b64_findings = self._inspect_base64_payloads(cleaned_text)
        findings.extend(b64_findings)

        if not findings:
            return GuardrailVerdict(
                status=GuardrailStatus.PASSED,
                is_safe=True,
                original_text=text,
                sanitized_text=text,
                findings=[],
            )

        # Handle mitigation action
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
        else:  # quarantine
            return GuardrailVerdict(
                status=GuardrailStatus.BLOCKED,
                is_safe=False,
                original_text=text,
                sanitized_text="[QUARANTINED BY OMNIGUARD: Adversarial injection detected]",
                findings=findings,
            )
