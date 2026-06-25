"""
Policy rule models and evaluation parser.
Supports nested AND/OR conditions and diverse logical operators.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from agentshield.models import (
    ClassificationVerdict,
    PolicyAction,
    SessionContext,
    ToolCallRequest,
    RiskLevel,
)


class PolicyRule(BaseModel):
    """A single policy rule containing a priority, conditions, and action decision."""

    id: str
    priority: int = 0
    condition: dict[str, Any] = Field(default_factory=dict)
    action: PolicyAction


class PolicyRuleSet(BaseModel):
    """A list of policy rules categorized under a policy ID."""

    version: str = "1.0"
    policy_id: str
    rules: list[PolicyRule] = Field(default_factory=list)


class RuleEvaluator:
    """Parses and evaluates rules against classification context."""

    def build_context(
        self,
        verdict: ClassificationVerdict,
        tool_call: ToolCallRequest,
        session: SessionContext,
    ) -> dict[str, Any]:
        """Flatten session metadata, tool metadata, and verdict into a single context dict."""
        tool_name = tool_call.tool_name.lower()
        original_task = session.original_task.lower()

        # Simple heuristic to determine if tool was part of original task intent
        tool_in_original_scope = True
        if session.allowed_tools:
            tool_in_original_scope = tool_call.tool_name in session.allowed_tools
        elif verdict.risk_level == RiskLevel.LOW:
            tool_in_original_scope = True
        else:
            # Check keywords
            if "email" in tool_name and "email" not in original_task and "mail" not in original_task:
                tool_in_original_scope = False
            elif "delete" in tool_name and not any(kw in original_task for kw in ["delete", "remove", "build", "compile", "clean", "organize"]):
                tool_in_original_scope = False
            elif "execute" in tool_name and not any(kw in original_task for kw in ["execute", "run", "organize", "process", "build", "compile", "script", "compute", "calculate", "test"]):
                tool_in_original_scope = False

        # Categorize tool
        from agentshield.mock_tools import get_tool_category
        tool_category = get_tool_category(tool_call.tool_name)

        # Check if there are network arguments
        has_network_args = False
        args_str = str(tool_call.tool_args).lower()
        if "http" in args_str or "www." in args_str or ".com" in args_str or ".io" in args_str:
            has_network_args = True

        return {
            "risk_level": verdict.risk_level.value,
            "attack_type": verdict.attack_type.value,
            "confidence_score": verdict.confidence_score,
            "tool_name": tool_call.tool_name,
            "tool_category": tool_category,
            "tool_in_original_scope": tool_in_original_scope,
            "action_count": session.action_count_for(tool_call.tool_name),
            "session_tool_count": len(session.action_history),
            "has_network_args": has_network_args,
        }

    def evaluate_condition(self, condition: dict[str, Any], context: dict[str, Any]) -> bool:
        """Evaluate a single rule condition block (supporting nested or/and operators)."""
        if not condition:
            return True

        # Handle nested logic blocks
        if "and" in condition:
            return all(self.evaluate_condition(c, context) for c in condition["and"])

        if "or" in condition:
            return any(self.evaluate_condition(c, context) for c in condition["or"])

        # Base condition: field, operator, value
        field = condition.get("field")
        operator = condition.get("operator")
        expected_val = condition.get("value")

        if field not in context:
            return False

        actual_val = context[field]

        # String norm for operators
        op = operator.lower() if isinstance(operator, str) else ""

        if op == "eq":
            return actual_val == expected_val
        if op == "neq":
            return actual_val != expected_val
        if op == "gt":
            return actual_val > expected_val
        if op == "lt":
            return actual_val < expected_val
        if op == "gte":
            return actual_val >= expected_val
        if op == "lte":
            return actual_val <= expected_val
        if op == "in":
            return actual_val in expected_val if isinstance(expected_val, list) else actual_val == expected_val
        if op == "not_in":
            return actual_val not in expected_val if isinstance(expected_val, list) else actual_val != expected_val
        if op == "contains":
            return expected_val in actual_val if isinstance(actual_val, (str, list)) else False
        if op == "matches":
            if isinstance(actual_val, str) and isinstance(expected_val, str):
                return bool(re.search(expected_val, actual_val))
            return False

        return False
