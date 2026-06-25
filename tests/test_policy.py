"""
Tests for the Policy Engine and Rule Evaluator.
"""

from __future__ import annotations

import pytest

from agentshield.models import (
    ActionDecision,
    AttackType,
    ClassificationTier,
    ClassificationVerdict,
    PolicyAction,
    RiskLevel,
    SessionContext,
    ToolCallRequest,
)
from agentshield.policy.engine import PolicyEngine


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine()


@pytest.fixture
def session() -> SessionContext:
    return SessionContext(
        session_id="policy-session",
        original_task="Clean code repository",
        allowed_tools=["read_file", "write_file", "delete_file"],
    )


def test_high_network_blocked(engine: PolicyEngine, session: SessionContext):
    verdict = ClassificationVerdict(
        risk_level=RiskLevel.HIGH,
        attack_type=AttackType.ASI01_GOAL_HIJACK,
        confidence_score=0.95,
        reasoning="Network exfiltration",
        recommended_action=ActionDecision.BLOCK,
    )
    req = ToolCallRequest(
        tool_name="http_request",
        tool_args={"url": "https://evil.com"},
        tool_category="network",
    )
    res = engine.evaluate(verdict, req, session)
    assert res.action == PolicyAction.BLOCK
    assert res.matched_rule_id == "block-high-network"


def test_high_uncertain_escalated(engine: PolicyEngine, session: SessionContext):
    verdict = ClassificationVerdict(
        risk_level=RiskLevel.HIGH,
        attack_type=AttackType.OTHER,
        confidence_score=0.6,  # Low confidence
        reasoning="Uncertain high risk",
        recommended_action=ActionDecision.ESCALATE,
    )
    # Use a generic system tool to bypass the block-high-network (which covers network/email/db/code)
    req = ToolCallRequest(
        tool_name="update_user_profile",
        tool_args={"display_name": "x"},
        tool_category="system",
    )
    res = engine.evaluate(verdict, req, session)
    assert res.action == PolicyAction.ESCALATE
    assert res.matched_rule_id == "escalate-uncertain-high"


def test_out_of_scope_blocked(engine: PolicyEngine, session: SessionContext):
    verdict = ClassificationVerdict(
        risk_level=RiskLevel.LOW,
        confidence_score=0.9,
        reasoning="Benevolent send email request",
        recommended_action=ActionDecision.ALLOW,
    )
    req = ToolCallRequest(
        tool_name="send_email",  # Tool is not in session.allowed_tools
        tool_args={"to": "user@corp.com"},
        tool_category="email",
    )
    res = engine.evaluate(verdict, req, session)
    assert res.action == PolicyAction.BLOCK
    assert res.matched_rule_id == "block-out-of-scope"


def test_medium_logged(engine: PolicyEngine, session: SessionContext):
    verdict = ClassificationVerdict(
        risk_level=RiskLevel.MEDIUM,
        confidence_score=0.7,
        reasoning="Plausible operation",
        recommended_action=ActionDecision.LOG_AND_ALLOW,
    )
    req = ToolCallRequest(
        tool_name="read_file",
        tool_args={"path": "temp.txt"},
        tool_category="filesystem",
    )
    res = engine.evaluate(verdict, req, session)
    assert res.action == PolicyAction.LOG_AND_ALLOW
    assert res.matched_rule_id == "log-medium"


def test_low_allowed(engine: PolicyEngine, session: SessionContext):
    verdict = ClassificationVerdict(
        risk_level=RiskLevel.LOW,
        confidence_score=0.99,
        reasoning="Clearly low risk",
        recommended_action=ActionDecision.ALLOW,
    )
    req = ToolCallRequest(
        tool_name="read_file",
        tool_args={"path": "report.pdf"},
        tool_category="filesystem",
    )
    res = engine.evaluate(verdict, req, session)
    assert res.action == PolicyAction.ALLOW
    assert res.matched_rule_id == "allow-low"


def test_repeated_tool_escalated(engine: PolicyEngine, session: SessionContext):
    # Simulate a tool called > 10 times
    req = ToolCallRequest(
        tool_name="read_file",
        tool_args={"path": "f.txt"},
        tool_category="filesystem",
    )
    for _ in range(11):
        session.add_action(req)

    verdict = ClassificationVerdict(
        risk_level=RiskLevel.LOW,
        confidence_score=0.9,
        reasoning="Repeated reads",
        recommended_action=ActionDecision.ALLOW,
    )
    res = engine.evaluate(verdict, req, session)
    assert res.action == PolicyAction.ESCALATE
    assert res.matched_rule_id == "flag-repeated-tool"


def test_priority_ordering(engine: PolicyEngine, session: SessionContext):
    # If risk is HIGH (escalate is 90) but it is also out of scope (80)
    # The higher priority (90) escalate-uncertain-high should win over block-out-of-scope (80)
    # if it doesn't match block-high-network.
    verdict = ClassificationVerdict(
        risk_level=RiskLevel.HIGH,
        confidence_score=0.7,
        reasoning="Risk is high but confidence is low",
        recommended_action=ActionDecision.ESCALATE,
    )
    req = ToolCallRequest(
        tool_name="update_user_profile",  # Out of scope and not network
        tool_args={"display_name": "x"},
        tool_category="system",
    )
    res = engine.evaluate(verdict, req, session)
    assert res.action == PolicyAction.ESCALATE
    assert res.matched_rule_id == "escalate-uncertain-high"


def test_custom_policy_loading(session: SessionContext):
    custom_yaml = """
version: "1.1"
policy_id: "custom-test"
rules:
  - id: "block-all-filesystem"
    priority: 150
    condition:
      field: "tool_category"
      operator: "eq"
      value: "filesystem"
    action: "BLOCK"
"""
    engine = PolicyEngine(policy_yaml=custom_yaml)
    verdict = ClassificationVerdict(
        risk_level=RiskLevel.LOW,
        confidence_score=0.99,
        reasoning="Safe",
        recommended_action=ActionDecision.ALLOW,
    )
    req = ToolCallRequest(
        tool_name="read_file",
        tool_args={"path": "x.txt"},
        tool_category="filesystem",
    )
    res = engine.evaluate(verdict, req, session)
    assert res.action == PolicyAction.BLOCK
    assert res.matched_rule_id == "block-all-filesystem"


def test_policy_yaml_roundtrip(engine: PolicyEngine):
    yaml_str = engine.to_yaml()
    assert "version" in yaml_str
    assert "rules" in yaml_str

    engine2 = PolicyEngine(policy_yaml=yaml_str)
    assert len(engine2.ruleset.rules) == len(engine.ruleset.rules)
