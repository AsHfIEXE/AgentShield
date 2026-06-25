"""
LangChain-compatible middleware wrapper.
Intercepts tool calls in LangChain 1.x before execution using AgentMiddleware.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

# Check if LangChain is installed, otherwise define dummy placeholders
try:
    from langchain.agents.middleware import AgentMiddleware
    from langchain.agents.middleware.types import ToolCallRequest, ToolMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Dummy classes to prevent import-time crashes when LangChain is not installed
    class AgentMiddleware:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    class ToolCallRequest:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    class ToolMessage:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    LANGCHAIN_AVAILABLE = False

from agentshield.interceptor.base import BaseInterceptor
from agentshield.models import (
    PolicyAction,
    SessionContext,
    ToolCallRequest as ASToolCallRequest,
)


class AgentShieldMiddleware(AgentMiddleware):
    """LangChain AgentMiddleware filter.

    Intercepts every ToolCallRequest before execution, pausing for classification
    and policy evaluation, returning a direct ToolMessage block response on violation.
    """

    def __init__(self, classifier, policy_engine, audit_log, session: SessionContext):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. "
                "Please run `pip install langchain langchain-core` to use this middleware."
            )
        super().__init__()
        self.interceptor = BaseInterceptor(classifier, policy_engine, audit_log)
        self.policy = policy_engine
        self.session = session
        self.audit = audit_log

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[..., Any]
    ) -> Any:
        """Asynchronous tool call wrapper."""
        # Convert LangChain request to AgentShield request model
        as_request = ASToolCallRequest(
            tool_name=request.tool_name,
            tool_args=request.tool_args,
            tool_call_id=request.tool_call_id,
            session_id=self.session.session_id,
            source_framework="langchain",
        )

        policy_result, verdict = await self.interceptor.intercept(as_request, self.session)

        if policy_result.action == PolicyAction.BLOCK:
            return ToolMessage(
                content=f"[AgentShield] Blocked: {verdict.reasoning}",
                tool_call_id=request.tool_call_id,
            )

        if policy_result.action == PolicyAction.ESCALATE:
            approved = await self.policy.human_approval_queue.wait(as_request, verdict)
            if not approved:
                return ToolMessage(
                    content=f"[AgentShield] Denied by human reviewer: {verdict.reasoning}",
                    tool_call_id=request.tool_call_id,
                )
            # If approved, add action to session and log override
            self.session.add_action(as_request)
            self.audit.record_human_override(as_request.call_id, True)

        # Allow / Log and Allow
        return await handler(request)

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[..., Any]
    ) -> Any:
        """Synchronous wrapper delegating to async implementation."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.awrap_tool_call(request, handler))
