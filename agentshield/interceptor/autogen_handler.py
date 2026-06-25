"""
AutoGen v0.4+ compatible intervention handler.
Intercepts FunctionCall messages and applies AgentShield protections before execution.
"""

from __future__ import annotations

import asyncio
from typing import Any

# Check if AutoGen is installed, otherwise define dummy placeholders
try:
    from autogen_core import DefaultInterventionHandler, FunctionCall
    AUTOGEN_AVAILABLE = True
except ImportError:
    # Dummy classes to prevent import crashes
    class DefaultInterventionHandler:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    class FunctionCall:  # type: ignore
        pass
    AUTOGEN_AVAILABLE = False

from agentshield.interceptor.base import BaseInterceptor
from agentshield.models import (
    PolicyAction,
    SessionContext,
    ToolCallRequest as ASToolCallRequest,
)


class AgentShieldHandler(DefaultInterventionHandler):
    """AutoGen Intervention Handler.

    Intercepts outgoing FunctionCall messages and executes AgentShield validation.
    Raises ValueError to abort the message chain if blocked or denied.
    """

    def __init__(self, classifier, policy, audit, session: SessionContext):
        if not AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen Core is not installed. "
                "Please run `pip install autogen-core` to use this handler."
            )
        super().__init__()
        self.interceptor = BaseInterceptor(classifier, policy, audit)
        self.policy = policy
        self.session = session
        self.audit = audit

    async def on_send(self, message: Any, *, sender: Any, recipient: Any) -> Any:
        """Intercept FunctionCall message type, validate risk and policy."""
        if not isinstance(message, FunctionCall):
            return message  # Pass non-tool messages through directly

        # Convert AutoGen FunctionCall to AgentShield Request
        # In AutoGen v0.4, FunctionCall typically has name and arguments attributes
        tool_name = getattr(message, "name", "unknown_tool")
        tool_args = getattr(message, "arguments", {})

        as_request = ASToolCallRequest(
            tool_name=tool_name,
            tool_args=tool_args,
            session_id=self.session.session_id,
            source_framework="autogen",
        )

        policy_result, verdict = await self.interceptor.intercept(as_request, self.session)

        if policy_result.action == PolicyAction.BLOCK:
            raise ValueError(
                f"[AgentShield] Blocked tool execution: {verdict.reasoning}"
            )

        if policy_result.action == PolicyAction.ESCALATE:
            approved = await self.policy.human_approval_queue.wait(as_request, verdict)
            if not approved:
                raise ValueError(
                    f"[AgentShield] Execution denied by human reviewer: {verdict.reasoning}"
                )
            # Log override and add to session
            self.session.add_action(as_request)
            self.audit.record_human_override(as_request.call_id, True)

        # Allow / Log and Allow
        return message
