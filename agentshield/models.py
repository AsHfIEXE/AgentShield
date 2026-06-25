"""
Core data models shared across all AgentShield components.

All inter-component communication uses these Pydantic models for
strict type safety and JSON serialization.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AttackType(str, Enum):
    NONE = "NONE"
    ASI01_GOAL_HIJACK = "ASI01_GOAL_HIJACK"
    ASI02_TOOL_MISUSE = "ASI02_TOOL_MISUSE"
    ASI03_IDENTITY_ABUSE = "ASI03_IDENTITY_ABUSE"
    ASI05_CODE_EXECUTION = "ASI05_CODE_EXECUTION"
    ASI06_MEMORY_POISONING = "ASI06_MEMORY_POISONING"
    ASI07_INTER_AGENT_COMMS = "ASI07_INTER_AGENT_COMMS"
    OTHER = "OTHER"


class ActionDecision(str, Enum):
    ALLOW = "ALLOW"
    LOG_AND_ALLOW = "LOG_AND_ALLOW"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"


class ClassificationTier(str, Enum):
    TIER1_DETERMINISTIC = "TIER1_DETERMINISTIC"
    TIER2_LIGHTWEIGHT_LLM = "TIER2_LIGHTWEIGHT_LLM"
    TIER3_FULL_LLM = "TIER3_FULL_LLM"
    SEQUENCE_DETECTION = "SEQUENCE_DETECTION"


# ── Request / Response Models ─────────────────────────────────────────────────


class ToolCallRequest(BaseModel):
    """Represents a single tool invocation intercepted before execution."""

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_category: Optional[str] = None  # network, filesystem, email, code, database
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = ""
    source_framework: str = "generic"  # langchain, autogen, crewai, generic
    tool_call_id: str = ""

    @model_validator(mode="before")
    @classmethod
    def populate_ids(cls, data: Any) -> Any:
        if isinstance(data, dict):
            t_id = data.get("tool_call_id")
            c_id = data.get("call_id")
            if t_id and not c_id:
                data["call_id"] = t_id
            elif c_id and not t_id:
                data["tool_call_id"] = c_id
        return data

    def args_summary(self, max_len: int = 120) -> str:
        """Truncated string representation of arguments for display."""
        s = str(self.tool_args)
        return s[:max_len] + "…" if len(s) > max_len else s


class ClassificationVerdict(BaseModel):
    """Output from the intent classification engine."""

    risk_level: RiskLevel
    attack_type: AttackType = AttackType.NONE
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    recommended_action: ActionDecision
    tier_used: ClassificationTier = ClassificationTier.TIER1_DETERMINISTIC
    raw_llm_response: Optional[str] = None


class PolicyAction(str, Enum):
    ALLOW = "ALLOW"
    LOG_AND_ALLOW = "LOG_AND_ALLOW"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"


class PolicyResult(BaseModel):
    """Output from the policy enforcement gate."""

    action: PolicyAction
    matched_rule_id: Optional[str] = None
    matched_rule_condition: Optional[str] = None


class AuditEntry(BaseModel):
    """A single row in the audit log."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = ""
    tool_call: ToolCallRequest
    verdict: ClassificationVerdict
    policy_result: PolicyResult
    human_override: Optional[bool] = None  # True=approved, False=denied, None=no override
    human_override_timestamp: Optional[datetime] = None


# ── Session Context ───────────────────────────────────────────────────────────


class SessionContext(BaseModel):
    """Tracks the full session state for context-aware classification."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_task: str = ""
    allowed_paths: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    action_history: list[ToolCallRequest] = Field(default_factory=list)
    registered_tools_at_start: list[str] = Field(default_factory=list)

    def compressed_history(self, max_recent: int = 10) -> str:
        """Return a compact representation of action history for the classifier.

        Keeps last `max_recent` calls verbatim; summarizes older ones.
        """
        if not self.action_history:
            return "No prior actions in this session."

        parts: list[str] = []

        # Summarize older actions
        older = self.action_history[:-max_recent] if len(self.action_history) > max_recent else []
        if older:
            tool_counts: dict[str, int] = {}
            for call in older:
                tool_counts[call.tool_name] = tool_counts.get(call.tool_name, 0) + 1
            summary_items = [f"{name}(×{count})" for name, count in tool_counts.items()]
            parts.append(f"Earlier: {', '.join(summary_items)}")

        # Recent actions verbatim
        recent = self.action_history[-max_recent:]
        for call in recent:
            parts.append(f"  → {call.tool_name}({call.args_summary(80)})")

        return "\n".join(parts)

    def add_action(self, call: ToolCallRequest) -> None:
        self.action_history.append(call)

    def action_count_for(self, tool_name: str) -> int:
        return sum(1 for c in self.action_history if c.tool_name == tool_name)


# ── Scenario Definitions (for test suite & demo) ─────────────────────────────


class AttackScenario(BaseModel):
    """Defines a test scenario for the classification pipeline."""

    scenario_id: str
    description: str
    original_task: str
    action_history: list[ToolCallRequest] = Field(default_factory=list)
    tool_call: ToolCallRequest
    expected_risk: RiskLevel
    expected_action: ActionDecision
    attack_category: AttackType = AttackType.NONE
    owasp_reference: Optional[str] = None  # e.g. "goal_hijack.basic_001"
