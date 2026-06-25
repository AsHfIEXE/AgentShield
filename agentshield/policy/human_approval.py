"""
Human-in-the-loop escalation queue.
Bridges asynchronous tool interception with synchronous/threaded dashboard approvals.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import threading
from typing import Any, Optional

from agentshield.models import ClassificationVerdict, ToolCallRequest


class HumanApprovalQueue:
    """Thread-safe queue to hold tool calls requiring human operator review."""

    def __init__(self):
        self._lock = threading.Lock()
        self.pending: dict[str, dict[str, Any]] = {}

    async def wait(
        self,
        request: ToolCallRequest,
        verdict: ClassificationVerdict,
        timeout: float = 30.0,
    ) -> bool:
        """Hold the calling thread/coroutine until approval decision is received.

        Defaults to fail-closed on timeout.
        """
        loop = asyncio.get_running_loop()
        event = asyncio.Event()

        item = {
            "request": request,
            "verdict": verdict,
            "timestamp": datetime.now(timezone.utc),
            "event": event,
            "loop": loop,
            "approved": False,
        }

        with self._lock:
            self.pending[request.tool_call_id] = item

        try:
            # Wait for event to trigger with timeout
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return item["approved"]
        except asyncio.TimeoutError:
            # Timeout triggers fail-closed security posture
            with self._lock:
                if request.tool_call_id in self.pending:
                    del self.pending[request.tool_call_id]
            return False

    def resolve(self, call_id: str, approved: bool) -> None:
        """Resolve a pending request. Thread-safe."""
        with self._lock:
            item = self.pending.get(call_id)
            if not item:
                return

            item["approved"] = approved
            loop = item["loop"]
            event = item["event"]

            # Safely trigger event across thread boundaries
            loop.call_soon_threadsafe(event.set)
            del self.pending[call_id]

    def get_pending(self) -> list[dict[str, Any]]:
        """Return a copy of pending items for UI display."""
        with self._lock:
            return [
                {
                    "call_id": k,
                    "tool_name": v["request"].tool_name,
                    "tool_args": v["request"].tool_args,
                    "reasoning": v["verdict"].reasoning,
                    "timestamp": v["timestamp"],
                }
                for k, v in self.pending.items()
            ]

    def clear_resolved(self) -> None:
        """Clear out queue."""
        with self._lock:
            self.pending.clear()
