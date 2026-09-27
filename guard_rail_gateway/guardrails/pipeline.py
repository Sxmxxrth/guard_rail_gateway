"""Composable multi-stage guardrail pipeline orchestrator."""

from __future__ import annotations

import time
from typing import Any

from guard_rail_gateway.config import Settings, get_config
from guard_rail_gateway.guardrails.injection import PromptInjectionGuardrail
from guard_rail_gateway.guardrails.output_guard import OutputSafetyGuardrail
from guard_rail_gateway.guardrails.pii import PIISanitizerGuardrail
from guard_rail_gateway.guardrails.toxicity import ToxicityHarmGuardrail
from guard_rail_gateway.models import (
    GuardrailStatus,
    GuardrailVerdict,
    SecurityFinding,
)


class GuardrailPipeline:
    """Orchestrates input and output guardrail evaluation stages with reversible PII de-anonymization."""

    def __init__(self, config: Settings | None = None):
        self.config = config or get_config()

        # Initialize Input Guardrails
        self.injection_guard = PromptInjectionGuardrail(
            action=self.config.injection_action,
            enabled=self.config.prompt_injection_enabled,
        )
        self.toxicity_guard = ToxicityHarmGuardrail(
            action="block",
            enabled=self.config.toxicity_filter_enabled,
        )
        self.pii_guard = PIISanitizerGuardrail(
            action=self.config.pii_action,
            enabled=self.config.pii_anonymization_enabled,
        )

        # Initialize Output Guardrail
        self.output_guard = OutputSafetyGuardrail(
            action="sanitize",
            enabled=self.config.output_guard_enabled,
        )

    def evaluate_input(self, text: str, context: dict[str, Any] | None = None) -> GuardrailVerdict:
        """Evaluates incoming user prompt across injection, toxicity, and PII guardrails.
        
        Execution Order:
        1. InjectionGuard: If prompt injection detected -> block immediately.
        2. ToxicityGuard: If harmful intent detected -> block immediately.
        3. PIIGuard: If sensitive data detected -> anonymize with reversible tokens.
        """
        start_time = time.perf_counter()
        current_text = text
        all_findings: list[SecurityFinding] = []
        combined_token_map: dict[str, str] = {}
        pipeline_status = GuardrailStatus.PASSED

        # 1. Injection Guard
        inj_verdict = self.injection_guard.run(current_text, context)
        all_findings.extend(inj_verdict.findings)
        if not inj_verdict.is_safe:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GuardrailVerdict(
                status=inj_verdict.status,
                is_safe=False,
                original_text=text,
                sanitized_text=None,
                findings=all_findings,
                latency_ms=round(elapsed_ms, 3),
            )

        # 2. Toxicity & Harm Guard
        tox_verdict = self.toxicity_guard.run(current_text, context)
        all_findings.extend(tox_verdict.findings)
        if not tox_verdict.is_safe:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GuardrailVerdict(
                status=tox_verdict.status,
                is_safe=False,
                original_text=text,
                sanitized_text=None,
                findings=all_findings,
                latency_ms=round(elapsed_ms, 3),
            )

        # 3. PII & Secret Anonymizer
        pii_verdict = self.pii_guard.run(current_text, context)
        all_findings.extend(pii_verdict.findings)
        if not pii_verdict.is_safe:
            # PII was configured to block instead of anonymize
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GuardrailVerdict(
                status=pii_verdict.status,
                is_safe=False,
                original_text=text,
                sanitized_text=None,
                findings=all_findings,
                latency_ms=round(elapsed_ms, 3),
            )

        if pii_verdict.sanitized_text is not None:
            current_text = pii_verdict.sanitized_text
            combined_token_map.update(pii_verdict.pii_token_map)
            if pii_verdict.status == GuardrailStatus.SANITIZED:
                pipeline_status = GuardrailStatus.SANITIZED

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return GuardrailVerdict(
            status=pipeline_status,
            is_safe=True,
            original_text=text,
            sanitized_text=current_text,
            findings=all_findings,
            pii_token_map=combined_token_map,
            latency_ms=round(elapsed_ms, 3),
        )

    def evaluate_output(
        self,
        text: str,
        pii_token_map: dict[str, str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> GuardrailVerdict:
        """Evaluates model output for leaks, then reversibly de-anonymizes PII tokens."""
        start_time = time.perf_counter()

        # 1. Output safety evaluation
        out_verdict = self.output_guard.run(text, context)
        eval_text = out_verdict.sanitized_text if out_verdict.sanitized_text is not None else text

        # 2. Reversibly restore PII tokens back to original values
        if pii_token_map:
            eval_text = PIISanitizerGuardrail.restore_pii(eval_text, pii_token_map)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return GuardrailVerdict(
            status=out_verdict.status,
            is_safe=out_verdict.is_safe,
            original_text=text,
            sanitized_text=eval_text,
            findings=out_verdict.findings,
            latency_ms=round(elapsed_ms, 3),
        )
