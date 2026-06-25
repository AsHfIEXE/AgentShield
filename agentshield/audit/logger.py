"""
Audit logging system.
Tracks and serializes decision actions and logs history for verification.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from typing import Any, Optional

import pandas as pd
from pydantic import TypeAdapter

from agentshield.models import (
    AuditEntry,
    ClassificationVerdict,
    PolicyAction,
    PolicyResult,
    ToolCallRequest,
)


class AuditLog:
    """Stores a log of all tool call decisions, classifications, and actions."""

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self.entries: list[AuditEntry] = []

        # Configure Syslog/SIEM logger
        self.syslog_logger = logging.getLogger(f"agentshield_siem_{session_id or 'default'}")
        self.syslog_logger.setLevel(logging.INFO)
        self.syslog_logger.handlers.clear()

        syslog_host = os.environ.get("AGENTSHEILD_SYSLOG_HOST")
        syslog_port = int(os.environ.get("AGENTSHEILD_SYSLOG_PORT", "514"))

        if syslog_host:
            try:
                handler = logging.handlers.SysLogHandler(address=(syslog_host, syslog_port))
                self.syslog_logger.addHandler(handler)
            except Exception:
                self.syslog_logger.addHandler(logging.StreamHandler(sys.stdout))
        else:
            handler = None
            if sys.platform.startswith("linux"):
                for path in ["/dev/log", "/var/run/syslog"]:
                    if os.path.exists(path):
                        try:
                            handler = logging.handlers.SysLogHandler(address=path)
                            break
                        except Exception:
                            pass
            if not handler:
                handler = logging.StreamHandler(sys.stdout)
            self.syslog_logger.addHandler(handler)

    def to_cef(self, entry: AuditEntry) -> str:
        """Format an AuditEntry in Common Event Format (CEF)."""
        severity = 3
        if entry.verdict.risk_level.value == "MEDIUM":
            severity = 5
        elif entry.verdict.risk_level.value == "HIGH":
            severity = 8
        if entry.policy_result.action.value in ["BLOCK", "ESCALATE"]:
            severity = 10

        rule_id = entry.policy_result.matched_rule_id or "unknown"
        tool_name = entry.tool_call.tool_name
        action = entry.policy_result.action.value
        
        reason = entry.verdict.reasoning.replace("=", "\\=").replace("|", "\\|")
        ext = (
            f"session={entry.session_id} "
            f"tool={tool_name} "
            f"action={action} "
            f"risk={entry.verdict.risk_level.value} "
            f"confidence={entry.verdict.confidence_score:.2f} "
            f"reason={reason}"
        )
        if entry.human_override is not None:
            override_val = "APPROVED" if entry.human_override else "DENIED"
            ext += f" override={override_val}"
            
        return f"CEF:0|AgentShield|Gateway|0.1.0|{rule_id}|Tool Call Evaluation|{severity}|{ext}"

    def to_json_event(self, entry: AuditEntry) -> str:
        """Format an AuditEntry as a flat JSON log line."""
        return json.dumps({
            "timestamp": entry.timestamp.isoformat(),
            "session_id": entry.session_id,
            "tool_call": {
                "tool_name": entry.tool_call.tool_name,
                "tool_args": entry.tool_call.tool_args,
                "tool_category": entry.tool_call.tool_category,
            },
            "verdict": {
                "risk_level": entry.verdict.risk_level.value,
                "attack_type": entry.verdict.attack_type.value,
                "confidence_score": entry.verdict.confidence_score,
                "reasoning": entry.verdict.reasoning,
            },
            "policy_result": {
                "action": entry.policy_result.action.value,
                "matched_rule_id": entry.policy_result.matched_rule_id,
            },
            "human_override": entry.human_override,
        })

    def log(
        self,
        tool_call: ToolCallRequest,
        verdict: ClassificationVerdict,
        policy_result: PolicyResult,
    ) -> AuditEntry:
        """Create and append a new audit log record."""
        entry = AuditEntry(
            session_id=self.session_id or tool_call.session_id,
            tool_call=tool_call,
            verdict=verdict,
            policy_result=policy_result,
        )
        self.entries.append(entry)

        # Emit SIEM logs
        try:
            self.syslog_logger.info(self.to_cef(entry))
            self.syslog_logger.info(self.to_json_event(entry))
        except Exception:
            pass

        return entry

    def record_human_override(self, call_id: str, approved: bool) -> None:
        """Record the operator approval/override decision for a pending entry."""
        # Find the entry matching this tool call ID
        for entry in self.entries:
            if entry.tool_call.tool_call_id == call_id:
                import datetime
                entry.human_override = approved
                entry.human_override_timestamp = datetime.datetime.now(datetime.timezone.utc)

                # Re-emit updated SIEM logs with human override info
                try:
                    self.syslog_logger.info(self.to_cef(entry))
                    self.syslog_logger.info(self.to_json_event(entry))
                except Exception:
                    pass
                break

    def get_session_summary(self) -> str:
        """Generate a compact session summary text for inclusion in classification prompts."""
        if not self.entries:
            return "No previous tool calls in this session."

        lines = []
        for entry in self.entries:
            time_str = entry.timestamp.strftime("%H:%M:%S")
            tool_name = entry.tool_call.tool_name
            args = entry.tool_call.args_summary(50)
            verdict = entry.policy_result.action.value
            reasoning = entry.verdict.reasoning
            override = ""
            if entry.human_override is not None:
                status = "APPROVED" if entry.human_override else "DENIED"
                override = f" (OVERRIDDEN TO {status})"

            lines.append(
                f"[{time_str}] {tool_name}({args}) -> {verdict}{override} [{reasoning}]"
            )

        return "\n".join(lines)

    @property
    def stats(self) -> dict[str, int]:
        """Compute aggregate statistics of all decisions."""
        stats_dict = {
            "total": 0,
            "blocked": 0,
            "allowed": 0,
            "escalated": 0,
            "logged": 0,
            "human_overrides_allow": 0,
            "human_overrides_deny": 0,
        }

        for entry in self.entries:
            stats_dict["total"] += 1
            action = entry.policy_result.action

            if action == PolicyAction.BLOCK:
                stats_dict["blocked"] += 1
            elif action == PolicyAction.ALLOW:
                stats_dict["allowed"] += 1
            elif action == PolicyAction.ESCALATE:
                stats_dict["escalated"] += 1
            elif action == PolicyAction.LOG_AND_ALLOW:
                stats_dict["logged"] += 1

            if entry.human_override is True:
                stats_dict["human_overrides_allow"] += 1
            elif entry.human_override is False:
                stats_dict["human_overrides_deny"] += 1

        return stats_dict

    def as_dataframe(self) -> pd.DataFrame:
        """Convert audit log entries to a pandas DataFrame for Streamlit rendering."""
        if not self.entries:
            return pd.DataFrame(
                columns=[
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
            )

        records = []
        for entry in self.entries:
            records.append({
                "timestamp": entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "tool_name": entry.tool_call.tool_name,
                "args_summary": entry.tool_call.args_summary(100),
                "risk_level": entry.verdict.risk_level.value,
                "confidence": round(entry.verdict.confidence_score, 2),
                "attack_type": entry.verdict.attack_type.value,
                "verdict": entry.policy_result.action.value,
                "reasoning": entry.verdict.reasoning,
                "human_override": "Approved"
                if entry.human_override is True
                else "Denied"
                if entry.human_override is False
                else "None",
            })

        return pd.DataFrame(records)

    def to_json(self, path: str) -> None:
        """Serialize log entries to a JSON file."""
        data = [json.loads(e.model_dump_json()) for e in self.entries]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def from_json(self, path: str) -> None:
        """Load log entries from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        adapter = TypeAdapter(list[AuditEntry])
        self.entries = adapter.validate_python(data)
