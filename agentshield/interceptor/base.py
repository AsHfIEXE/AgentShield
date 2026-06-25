"""
Base framework-agnostic interceptor and Python decorator hooks.
Sits synchronously or asynchronously between agent triggers and tool registries.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable, Coroutine

from agentshield.models import (
    ActionDecision,
    ClassificationVerdict,
    PolicyAction,
    PolicyResult,
    SessionContext,
    ToolCallRequest,
)


class BaseInterceptor:
    """Core interceptor coordinating classification, policy execution, and audit logging."""

    def __init__(self, classifier, policy_engine, audit_log):
        self.classifier = classifier
        self.policy = policy_engine
        self.audit = audit_log

    async def intercept(
        self, request: ToolCallRequest, session: SessionContext
    ) -> tuple[PolicyResult, ClassificationVerdict]:
        """Intercept tool invocation, classify risk, evaluate policy, and log."""
        # 1. Classification
        verdict = await self.classifier.classify(request, session)

        # 2. Policy Enforcement
        policy_result = self.policy.evaluate(verdict, request, session)

        # 3. Log to audit trail
        self.audit.log(request, verdict, policy_result)

        # 4. Record to session history for context-awareness (only if allowed or logged)
        if policy_result.action in [PolicyAction.ALLOW, PolicyAction.LOG_AND_ALLOW]:
            session.add_action(request)

        return policy_result, verdict


def agentshield_intercept(
    classifier, policy, audit, session: SessionContext
) -> Callable:
    """Decorator factory to wrap any tool function with AgentShield gateway protections.

    Raises PermissionError if tool execution is blocked or denied.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            tool_args = {"args": list(args), "kwargs": kwargs}
            request = ToolCallRequest(
                tool_name=func.__name__,
                tool_args=tool_args,
                tool_category=getattr(func, "category", None),
                session_id=session.session_id,
            )

            # Intercept hook
            interceptor = BaseInterceptor(classifier, policy, audit)
            policy_result, verdict = await interceptor.intercept(request, session)

            if policy_result.action == PolicyAction.BLOCK:
                raise PermissionError(
                    f"[AgentShield] Blocked tool execution: {verdict.reasoning}"
                )

            if policy_result.action == PolicyAction.ESCALATE:
                approved = await policy.human_approval_queue.wait(request, verdict)
                if not approved:
                    raise PermissionError(
                        f"[AgentShield] Execution denied by human reviewer: {verdict.reasoning}"
                    )
                # If approved, add to session action history and proceed
                session.add_action(request)
                audit.record_human_override(request.call_id, True)

            # Allow / Log and Allow
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Sync fallback wraps async version
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator


class ShieldedToolWrapper:
    """Wraps individual tools or function registries with pre-execution safety filters."""

    def __init__(self, func: Callable, classifier, policy, audit, session: SessionContext):
        self.func = func
        self.decorated = agentshield_intercept(classifier, policy, audit, session)(func)

    def __call__(self, *args, **kwargs) -> Any:
        return self.decorated(*args, **kwargs)
