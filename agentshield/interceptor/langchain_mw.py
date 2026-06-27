"""
LangChain-compatible callback handler (LangChain 0.3+ / langchain-core 0.3+).

The old `langchain.agents.middleware.AgentMiddleware` API was removed.
Modern LangChain uses BaseCallbackHandler with on_tool_start / on_tool_end / on_tool_error.

Usage:
    from agentshield.interceptor.langchain_mw import AgentShieldCallbackHandler
    from langchain_core.callbacks import CallbackManager

    shield_cb = AgentShieldCallbackHandler(
        classifier=classifier,
        policy_engine=policy,
        audit_log=audit,
        session=session,
    )
    # Inject via callbacks kwarg on the agent executor:
    agent_executor.invoke({"input": "..."}, config={"callbacks": [shield_cb]})
    # Or attach globally:
    # agent_executor = AgentExecutor(..., callbacks=[shield_cb])
"""

from __future__ import annotations

import asyncio
from typing import Any, Union

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    class BaseCallbackHandler:  # type: ignore
        pass
    LANGCHAIN_AVAILABLE = False

from agentshield.interceptor.base import BaseInterceptor
from agentshield.models import (
    PolicyAction,
    SessionContext,
    ToolCallRequest as ASToolCallRequest,
)


class AgentShieldCallbackHandler(BaseCallbackHandler):
    """LangChain CallbackHandler that intercepts every tool call before execution.

    Works with any LangChain agent executor by hooking into on_tool_start.
    When a tool call is classified as BLOCK or ESCALATE (and denied), it raises
    ToolException to halt execution and return an error message to the agent.
    """

    raise_error: bool = True  # Required by langchain_core to surface errors properly

    def __init__(self, classifier, policy_engine, audit_log, session: SessionContext):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is not installed. "
                "Run: pip install langchain-core>=0.3 langchain>=0.3"
            )
        super().__init__()
        self.interceptor = BaseInterceptor(classifier, policy_engine, audit_log)
        self.policy = policy_engine
        self.session = session
        self.audit = audit_log
        # Stores last verdict so on_tool_end can log it
        self._last_call_id: str | None = None

    def _run_async(self, coro):
        """Run an async coroutine from a sync context."""
        try:
            loop = asyncio.get_running_loop()
            # Already in an event loop (e.g. Jupyter) — create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Intercept tool call BEFORE execution — blocks here if the call is denied."""
        tool_name = serialized.get("name", "unknown_tool")
        # Prefer structured inputs; fall back to raw string
        tool_args: dict[str, Any] = inputs or {"input": input_str}
        call_id = str(run_id) if run_id else ""

        as_request = ASToolCallRequest(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=call_id,
            session_id=self.session.session_id,
            source_framework="langchain",
        )
        self._last_call_id = as_request.call_id

        policy_result, verdict = self._run_async(
            self.interceptor.intercept(as_request, self.session)
        )

        if policy_result.action == PolicyAction.BLOCK:
            # Raise ToolException — LangChain routes this back to the agent as an error message
            try:
                from langchain_core.tools import ToolException
            except ImportError:
                raise PermissionError(
                    f"[AgentShield] Blocked: {verdict.reasoning}"
                )
            raise ToolException(f"[AgentShield] Blocked: {verdict.reasoning}")

        if policy_result.action == PolicyAction.ESCALATE:
            approved = self._run_async(
                self.policy.human_approval_queue.wait(as_request, verdict)
            )
            if not approved:
                try:
                    from langchain_core.tools import ToolException
                except ImportError:
                    raise PermissionError(
                        f"[AgentShield] Denied by human reviewer: {verdict.reasoning}"
                    )
                raise ToolException(
                    f"[AgentShield] Denied by human reviewer: {verdict.reasoning}"
                )
            self.session.add_action(as_request)
            self.audit.record_human_override(as_request.call_id, True)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called after tool succeeds — nothing to do, audit was already logged in intercept()."""
        pass

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool raises. AgentShield blocks propagate as ToolException — let them through."""
        pass
