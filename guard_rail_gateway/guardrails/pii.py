"""Bidirectional PII and Secret Token Anonymizer with reversible tokenization."""

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


def luhn_checksum_valid(card_number: str) -> bool:
    """Validates credit card numbers using the standard Luhn algorithm (mod 10)."""
    digits = [int(c) for c in card_number if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += (doubled - 9) if doubled > 9 else doubled
        else:
            checksum += digit

    return checksum % 10 == 0


class PIISanitizerGuardrail(BaseGuardrail):
    """Detects, redacts, and reversibly tokenizes PII and secret access tokens."""

    # Regex patterns for sensitive entities
    PATTERNS: ClassVar[dict[str, tuple[str, ThreatCategory, ThreatSeverity]]] = {
        "email_address": (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            ThreatCategory.PII_LEAK,
            ThreatSeverity.MEDIUM,
        ),
        "social_security_number": (
            r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
            ThreatCategory.PII_LEAK,
            ThreatSeverity.HIGH,
        ),
        "phone_number": (
            r"(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
            ThreatCategory.PII_LEAK,
            ThreatSeverity.LOW,
        ),
        "credit_card": (
            r"\b(?:\d[ -]*?){13,19}\b",
            ThreatCategory.PII_LEAK,
            ThreatSeverity.CRITICAL,
        ),
        "openai_api_key": (
            r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b",
            ThreatCategory.SECRET_KEY,
            ThreatSeverity.CRITICAL,
        ),
        "aws_access_key": (
            r"\bAKIA[0-9A-Z]{16}\b",
            ThreatCategory.SECRET_KEY,
            ThreatSeverity.CRITICAL,
        ),
        "github_token": (
            r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b",
            ThreatCategory.SECRET_KEY,
            ThreatSeverity.CRITICAL,
        ),
        "bearer_jwt": (
            r"Bearer\s+eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+",
            ThreatCategory.SECRET_KEY,
            ThreatSeverity.CRITICAL,
        ),
    }

    def __init__(
        self,
        name: str = "PIISanitizerGuardrail",
        action: Literal["anonymize", "block", "flag"] = "anonymize",
        enabled: bool = True,
    ):
        super().__init__(name=name, enabled=enabled)
        self.action = action

    def evaluate(self, text: str, context: dict[str, Any] | None = None) -> GuardrailVerdict:
        """Identifies PII/secrets, attaches findings, and produces sanitized text with token mapping."""
        findings: list[SecurityFinding] = []
        token_map: dict[str, str] = {}
        sanitized_text = text
        counter: dict[str, int] = {}

        # 1. Evaluate each pattern
        for rule_name, (regex_str, category, severity) in self.PATTERNS.items():
            matches = list(re.finditer(regex_str, sanitized_text))
            # Process in reverse order to preserve string indices when substituting
            for match in reversed(matches):
                val = match.group(0)

                # Special check for credit card: Luhn validation
                if rule_name == "credit_card":
                    clean_digits = re.sub(r"\D", "", val)
                    if not luhn_checksum_valid(clean_digits):
                        continue  # Skip false positive random numbers

                counter[rule_name] = counter.get(rule_name, 0) + 1
                token_placeholder = f"{{{{PII_{rule_name.upper()}_{counter[rule_name]}}}}}"

                findings.append(
                    SecurityFinding(
                        category=category,
                        severity=severity,
                        rule_name=rule_name,
                        description=f"Sensitive {rule_name.replace('_', ' ')} detected and sanitized",
                        matched_text=val[:4] + "***" + val[-4:] if len(val) > 8 else "***",
                        start_idx=match.start(),
                        end_idx=match.end(),
                        mitigation_action=self.action,
                    )
                )

                if self.action == "anonymize":
                    # Record mapping for reverse de-anonymization
                    token_map[token_placeholder] = val
                    start, end = match.start(), match.end()
                    sanitized_text = sanitized_text[:start] + token_placeholder + sanitized_text[end:]

        if not findings:
            return GuardrailVerdict(
                status=GuardrailStatus.PASSED,
                is_safe=True,
                original_text=text,
                sanitized_text=text,
                findings=[],
                pii_token_map={},
            )

        if self.action == "block":
            return GuardrailVerdict(
                status=GuardrailStatus.BLOCKED,
                is_safe=False,
                original_text=text,
                sanitized_text=None,
                findings=findings,
                pii_token_map={},
            )
        elif self.action == "flag":
            return GuardrailVerdict(
                status=GuardrailStatus.FLAGGED,
                is_safe=True,
                original_text=text,
                sanitized_text=text,
                findings=findings,
                pii_token_map={},
            )
        else:  # anonymize
            return GuardrailVerdict(
                status=GuardrailStatus.SANITIZED,
                is_safe=True,
                original_text=text,
                sanitized_text=sanitized_text,
                findings=findings,
                pii_token_map=token_map,
            )

    @staticmethod
    def restore_pii(text: str, token_map: dict[str, str]) -> str:
        """Reversibly restores original PII values into a model's completion output."""
        restored = text
        for placeholder, original_val in token_map.items():
            restored = restored.replace(placeholder, original_val)
        return restored
