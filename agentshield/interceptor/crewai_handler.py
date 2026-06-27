"""
CrewAI-compatible interceptor.
Wraps CrewAI tool execution by subclassing BaseTool and injecting the
AgentShield interception layer before the _run() method.

Usage:
    from agentshield.interceptor.crewai_handler import shield_crewai_tool

    # Wrap any existing CrewAI tool:
    raw_tool = MyCrewAITool()
    safe_tool = shield_crewai_tool(raw_tool, classifier, policy, audit, session)

    # Then pass safe_tool to your CrewAI Agent as normal.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Type

try:
    from crewai.tools import BaseTool
    from pydantic import BaseModel
    CREWAI_AVAILABLE = True
except ImportError:
    class BaseTool:  # type: ignore
        pass
    class BaseModel:  # type: ignore
        pass
    CREWAI_AVAILABLE = False

from agentshield.interceptor.base import BaseInterceptor
from agentshield.models import (
    PolicyAction,
    SessionContext,
    ToolCallRequest as ASToolCallRequest,
)


def _run_coroutine(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


class ShieldedCrewAITool(BaseTool):
    """A CrewAI BaseTool wrapper that gates all tool calls through AgentShield."""

    # We store shield components via class attributes set at construction time.
    # CrewAI uses Pydantic under the hood so we avoid instance __dict__ conflicts.
    _shield_interceptor: Any = None
    _shield_policy: Any = None
    _shield_audit: Any = None
    _shield_session: Any = None
    _wrapped_tool: Any = None

    name: str = "shielded_tool"
    description: str = "AgentShield-protected tool"
    args_schema: Optional[Type[BaseModel]] = None

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept and gate the tool call before delegating to the real tool."""
        tool_args: dict[str, Any] = dict(zip(["input"], args)) if args else {}
        tool_args.update(kwargs)

        as_request = ASToolCallRequest(
            tool_name=self.name,
            tool_args=tool_args,
            session_id=self._shield_session.session_id,
            source_framework="crewai",
        )

        policy_result, verdict = _run_coroutine(
            self._shield_interceptor.intercept(as_request, self._shield_session)
        )

        if policy_result.action == PolicyAction.BLOCK:
            return f"[AgentShield] Tool blocked: {verdict.reasoning}"

        if policy_result.action == PolicyAction.ESCALATE:
            approved = _run_coroutine(
                self._shield_policy.human_approval_queue.wait(as_request, verdict)
            )
            if not approved:
                return f"[AgentShield] Tool denied by human reviewer: {verdict.reasoning}"
            self._shield_session.add_action(as_request)
            self._shield_audit.record_human_override(as_request.call_id, True)

        # Delegate to the wrapped tool
        return self._wrapped_tool._run(*args, **kwargs)


def shield_crewai_tool(
    tool: Any,
    classifier: Any,
    policy_engine: Any,
    audit_log: Any,
    session: SessionContext,
) -> ShieldedCrewAITool:
    """Wrap any CrewAI BaseTool with AgentShield interception.

    Returns a new ShieldedCrewAITool that gates every _run() call.
    """
    if not CREWAI_AVAILABLE:
        raise ImportError(
            "crewai is not installed. Run: pip install crewai>=1.14"
        )

    interceptor = BaseInterceptor(classifier, policy_engine, audit_log)

    shielded = ShieldedCrewAITool()
    shielded.name = tool.name
    shielded.description = tool.description
    if hasattr(tool, "args_schema"):
        shielded.args_schema = tool.args_schema

    shielded._shield_interceptor = interceptor
    shielded._shield_policy = policy_engine
    shielded._shield_audit = audit_log
    shielded._shield_session = session
    shielded._wrapped_tool = tool

    return shielded
