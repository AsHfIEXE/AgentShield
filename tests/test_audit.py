"""
Tests for the Audit Logger and Feedback system.
"""

from __future__ import annotations

import json
import os
import pytest

from agentshield.audit.feedback import MisclassificationStore
from agentshield.audit.logger import AuditLog
from agentshield.models import (
    ActionDecision,
    ClassificationTier,
    ClassificationVerdict,
    PolicyAction,
    PolicyResult,
    RiskLevel,
    ToolCallRequest,
)


@pytest.fixture
def sample_data() -> tuple[ToolCallRequest, ClassificationVerdict, PolicyResult]:
    req = ToolCallRequest(
        tool_name="read_file",
        tool_args={"path": "report.pdf"},
        tool_category="filesystem",
        session_id="test-session",
    )
    verdict = ClassificationVerdict(
        risk_level=RiskLevel.LOW,
        confidence_score=0.95,
        reasoning="Benign access",
        recommended_action=ActionDecision.ALLOW,
        tier_used=ClassificationTier.TIER1_DETERMINISTIC,
    )
    result = PolicyResult(
        action=PolicyAction.ALLOW,
        matched_rule_id="allow-low",
    )
    return req, verdict, result


def test_log_entry_created(sample_data: tuple[ToolCallRequest, ClassificationVerdict, PolicyResult]):
    req, verdict, result = sample_data
    logger = AuditLog(session_id="session-123")
    entry = logger.log(req, verdict, result)

    assert len(logger.entries) == 1
    assert logger.entries[0].entry_id == entry.entry_id
    assert logger.entries[0].tool_call.tool_name == "read_file"
    assert logger.entries[0].policy_result.action == PolicyAction.ALLOW


def test_stats_counting(sample_data: tuple[ToolCallRequest, ClassificationVerdict, PolicyResult]):
    req, verdict, result = sample_data
    logger = AuditLog()

    # Log ALLOW
    logger.log(req, verdict, result)

    # Log BLOCK
    result_block = PolicyResult(action=PolicyAction.BLOCK, matched_rule_id="block-high")
    logger.log(req, verdict, result_block)

    # Log ESCALATE
    result_esc = PolicyResult(action=PolicyAction.ESCALATE, matched_rule_id="escalate")
    logger.log(req, verdict, result_esc)

    stats = logger.stats
    assert stats["total"] == 3
    assert stats["allowed"] == 1
    assert stats["blocked"] == 1
    assert stats["escalated"] == 1


def test_human_override_recorded(sample_data: tuple[ToolCallRequest, ClassificationVerdict, PolicyResult]):
    req, verdict, result = sample_data
    logger = AuditLog()
    entry = logger.log(req, verdict, result)

    logger.record_human_override(req.tool_call_id, True)
    assert logger.entries[0].human_override is True
    assert logger.entries[0].human_override_timestamp is not None
    assert logger.stats["human_overrides_allow"] == 1


def test_as_dataframe(sample_data: tuple[ToolCallRequest, ClassificationVerdict, PolicyResult]):
    req, verdict, result = sample_data
    logger = AuditLog()
    logger.log(req, verdict, result)

    df = logger.as_dataframe()
    assert not df.empty
    assert list(df.columns) == [
        "timestamp",
        "tool_name",
        "args_summary",
        "risk_level",
        "confidence",
        "attack_type",
        "verdict",
        "reasoning",
        "human_override",
    ]
    assert df.iloc[0]["tool_name"] == "read_file"


def test_session_summary(sample_data: tuple[ToolCallRequest, ClassificationVerdict, PolicyResult]):
    req, verdict, result = sample_data
    logger = AuditLog()
    logger.log(req, verdict, result)

    summary = logger.get_session_summary()
    assert "read_file" in summary
    assert "ALLOW" in summary
    assert "Benign access" in summary


def test_feedback_store_records_override(sample_data: tuple[ToolCallRequest, ClassificationVerdict, PolicyResult]):
    req, verdict, _ = sample_data
    store = MisclassificationStore()

    store.record_override(req, verdict, True)
    assert len(store.store) == 1
    assert store.store[0]["tool_name"] == "read_file"
    assert store.store[0]["correct_action"] == "ALLOW"


def test_feedback_few_shot_generation(sample_data: tuple[ToolCallRequest, ClassificationVerdict, PolicyResult]):
    req, verdict, _ = sample_data
    store = MisclassificationStore()
    store.record_override(req, verdict, True)

    examples = store.as_few_shot_examples()
    assert len(examples) == 1
    assert "read_file" in examples[0]
    assert "ALLOW" in examples[0]


def test_json_export_import(sample_data: tuple[ToolCallRequest, ClassificationVerdict, PolicyResult]):
    req, verdict, result = sample_data
    logger = AuditLog(session_id="session-1")
    logger.log(req, verdict, result)

    temp_path = "/home/ashfiexe/agentshield/tests/temp_audit.json"
    try:
        logger.to_json(temp_path)
        assert os.path.exists(temp_path)

        logger2 = AuditLog(session_id="session-2")
        logger2.from_json(temp_path)

        assert len(logger2.entries) == 1
        assert logger2.entries[0].tool_call.tool_name == "read_file"
        assert logger2.entries[0].policy_result.action == PolicyAction.ALLOW
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_cef_and_json_event_generation(sample_data: tuple[ToolCallRequest, ClassificationVerdict, PolicyResult]):
    req, verdict, result = sample_data
    logger = AuditLog(session_id="session-siem")
    entry = logger.log(req, verdict, result)

    cef = logger.to_cef(entry)
    assert "CEF:0|AgentShield|Gateway|" in cef
    assert "session=session-siem" in cef
    assert "tool=read_file" in cef
    assert "action=ALLOW" in cef

    json_evt = json.loads(logger.to_json_event(entry))
    assert json_evt["session_id"] == "session-siem"
    assert json_evt["tool_call"]["tool_name"] == "read_file"
    assert json_evt["policy_result"]["action"] == "ALLOW"

