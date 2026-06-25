"""
Policy enforcement engine.
Loads rules from YAML configurations and determines action decisions.
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

from agentshield.models import (
    ClassificationVerdict,
    PolicyAction,
    PolicyResult,
    SessionContext,
    ToolCallRequest,
)
from agentshield.policy.human_approval import HumanApprovalQueue
from agentshield.policy.rules import PolicyRule, PolicyRuleSet, RuleEvaluator

DEFAULT_POLICY_YAML = """version: "1.0"
policy_id: "agentshield-default"
rules:
  - id: "block-high-network"
    priority: 100
    condition:
      and:
        - field: "risk_level"
          operator: "eq"
          value: "HIGH"
        - field: "tool_category"
          operator: "in"
          value: ["network", "email", "database", "code"]
    action: "BLOCK"
  - id: "escalate-mcp-anomaly"
    priority: 95
    condition:
      field: "attack_type"
      operator: "eq"
      value: "ASI07_INTER_AGENT_COMMS"
    action: "ESCALATE"
  - id: "escalate-uncertain-high"
    priority: 90
    condition:
      and:
        - field: "risk_level"
          operator: "eq"
          value: "HIGH"
        - field: "confidence_score"
          operator: "lt"
          value: 0.8
    action: "ESCALATE"
  - id: "block-out-of-scope"
    priority: 80
    condition:
      field: "tool_in_original_scope"
      operator: "eq"
      value: false
    action: "BLOCK"
  - id: "flag-repeated-tool"
    priority: 70
    condition:
      field: "action_count"
      operator: "gt"
      value: 10
    action: "ESCALATE"
  - id: "log-medium"
    priority: 50
    condition:
      field: "risk_level"
      operator: "eq"
      value: "MEDIUM"
    action: "LOG_AND_ALLOW"
  - id: "allow-low"
    priority: 10
    condition:
      field: "risk_level"
      operator: "eq"
      value: "LOW"
    action: "ALLOW"
"""


class PolicyEngine:
    """Configurable rule engine mapping classification results to action gates."""

    def __init__(self, policy_path: Optional[str] = None, policy_yaml: Optional[str] = None):
        self.evaluator = RuleEvaluator()
        self.human_approval_queue = HumanApprovalQueue()
        self.ruleset: Optional[PolicyRuleSet] = None

        if policy_yaml:
            self.load_yaml(policy_yaml)
        elif policy_path and os.path.exists(policy_path):
            with open(policy_path, "r") as f:
                self.load_yaml(f.read())
        else:
            self.load_yaml(DEFAULT_POLICY_YAML)

    def load_yaml(self, yaml_str: str) -> None:
        """Parse rule configurations from a YAML string."""
        data = yaml.safe_load(yaml_str)
        rules = []
        for r in data.get("rules", []):
            rules.append(
                PolicyRule(
                    id=r["id"],
                    priority=r.get("priority", 0),
                    condition=r.get("condition", {}),
                    action=PolicyAction(r["action"].upper()),
                )
            )
        self.ruleset = PolicyRuleSet(
            version=str(data.get("version", "1.0")),
            policy_id=data.get("policy_id", "custom"),
            rules=rules,
        )

    def to_yaml(self) -> str:
        """Serialize current rule set back to YAML string format."""
        if not self.ruleset:
            return ""

        rules_list = []
        for r in self.ruleset.rules:
            rules_list.append({
                "id": r.id,
                "priority": r.priority,
                "condition": r.condition,
                "action": r.action.value,
            })

        data = {
            "version": self.ruleset.version,
            "policy_id": self.ruleset.policy_id,
            "rules": rules_list,
        }
        return yaml.dump(data, sort_keys=False)

    def evaluate(
        self,
        verdict: ClassificationVerdict,
        tool_call: ToolCallRequest,
        session: SessionContext,
    ) -> PolicyResult:
        """Evaluate rules in priority order (descending). First match wins."""
        if not self.ruleset or not self.ruleset.rules:
            # Safe default fallback
            return PolicyResult(action=PolicyAction.BLOCK, matched_rule_id="fallback-safety")

        context = self.evaluator.build_context(verdict, tool_call, session)

        # Sort rules: highest priority first
        sorted_rules = sorted(self.ruleset.rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            if self.evaluator.evaluate_condition(rule.condition, context):
                # Map recommended action if the rule action says so,
                # but typically rule action governs
                return PolicyResult(
                    action=rule.action,
                    matched_rule_id=rule.id,
                    matched_rule_condition=str(rule.condition),
                )

        # Fallback to BLOCK if nothing matched (fail-closed posture)
        return PolicyResult(action=PolicyAction.BLOCK, matched_rule_id="default-block-unmatched")
