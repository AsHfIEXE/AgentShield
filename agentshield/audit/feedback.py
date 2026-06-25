"""
Misclassification feedback loop store.
Captures human overrides to dynamically inject corrected examples in classifier prompts.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agentshield.models import ClassificationVerdict, ToolCallRequest


class MisclassificationStore:
    """Stores logs of corrected classifications for model feedback and few-shot calibration."""

    def __init__(self):
        self.store: list[dict[str, Any]] = []

    def record_override(
        self,
        request: ToolCallRequest,
        original_verdict: ClassificationVerdict,
        human_decision: bool,
    ) -> None:
        """Record a human operator override decision.

        Args:
            request: The tool request call.
            original_verdict: The classification verdict that was overridden.
            human_decision: True if operator allowed/approved (was blocked but shouldn't be),
                           False if operator denied/blocked (was allowed but shouldn't be).
        """
        import datetime
        case = {
            "tool_name": request.tool_name,
            "tool_args": request.tool_args,
            "original_verdict": {
                "risk_level": original_verdict.risk_level.value,
                "attack_type": original_verdict.attack_type.value,
                "confidence_score": original_verdict.confidence_score,
                "reasoning": original_verdict.reasoning,
                "recommended_action": original_verdict.recommended_action.value,
            },
            "correct_action": "ALLOW" if human_decision else "BLOCK",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self.store.append(case)

    def as_few_shot_examples(self, max_cases: int = 5) -> list[str]:
        """Convert recorded corrections into XML-formatted few-shot examples for LLM context."""
        recent_cases = self.store[-max_cases:] if len(self.store) > max_cases else self.store
        examples = []

        for i, case in enumerate(recent_cases):
            # Map action back to risk level
            cor_action = case["correct_action"]
            risk = "LOW" if cor_action == "ALLOW" else "HIGH"
            attack = "NONE" if cor_action == "ALLOW" else "ASI02_TOOL_MISUSE"

            ex = f"""[FEEDBACK EXAMPLE {i+1} — operator corrected]
Task: "Task context"
History: Completed some actions.
Tool call: {case['tool_name']}(**{json.dumps(case['tool_args'])})
{{"risk_level":"{risk}","attack_type":"{attack}","confidence_score":1.00,"reasoning":"Operator overridden correction.","recommended_action":"{cor_action}"}}"""
            examples.append(ex)

        return examples

    def save(self, path: str) -> None:
        """Save overrides store to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.store, f, indent=2)

    def load(self, path: str) -> None:
        """Load overrides store from a JSON file."""
        if os.path.exists(path):
            with open(path, "r") as f:
                self.store = json.load(f)
