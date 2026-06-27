"""
SQLite-backed persistent audit storage.

Wraps AuditLog with a SQLite backend so audit entries survive restarts.
Falls back gracefully to in-memory if the DB path is not writable.

Usage:
    from agentshield.audit.storage import PersistentAuditLog

    audit = PersistentAuditLog(session_id="prod-session", db_path="./agentshield_audit.db")
    # Use identically to AuditLog — all entries are auto-persisted.
    # On restart, call audit.load_session() to replay previous entries.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

from agentshield.audit.logger import AuditLog
from agentshield.models import (
    AuditEntry,
    ClassificationVerdict,
    PolicyResult,
    ToolCallRequest,
)

_DEFAULT_DB = os.path.join(
    os.environ.get("AGENTSHIELD_DB_DIR", "."),
    "agentshield_audit.db",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_entries (
    entry_id    TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    tool_name   TEXT NOT NULL,
    tool_args   TEXT NOT NULL,
    tool_category TEXT,
    risk_level  TEXT NOT NULL,
    attack_type TEXT NOT NULL,
    confidence  REAL NOT NULL,
    reasoning   TEXT NOT NULL,
    tier_used   TEXT NOT NULL,
    action      TEXT NOT NULL,
    matched_rule TEXT,
    human_override INTEGER,
    human_override_ts TEXT,
    raw_entry   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session ON audit_entries(session_id);
CREATE INDEX IF NOT EXISTS idx_ts ON audit_entries(timestamp);
"""


class PersistentAuditLog(AuditLog):
    """AuditLog with automatic SQLite persistence for production deployments."""

    def __init__(
        self,
        session_id: str = "",
        db_path: str = _DEFAULT_DB,
        load_existing: bool = True,
    ):
        super().__init__(session_id=session_id)
        self.db_path = db_path
        self._db_lock = threading.Lock()
        self._db_available = False

        try:
            self._init_db()
            self._db_available = True
            if load_existing and session_id:
                self._load_session(session_id)
        except Exception as e:
            # Graceful fallback: warn but continue in-memory only
            import warnings
            warnings.warn(
                f"[AgentShield] SQLite storage unavailable ({e}). "
                "Falling back to in-memory audit log.",
                stacklevel=2,
            )

    def _init_db(self) -> None:
        with self._db_lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()

    def _load_session(self, session_id: str) -> None:
        """Reload prior entries for this session from the DB into memory."""
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            cur = conn.execute(
                "SELECT raw_entry FROM audit_entries WHERE session_id=? ORDER BY timestamp",
                (session_id,),
            )
            rows = cur.fetchall()
            conn.close()

        for (raw,) in rows:
            try:
                data = json.loads(raw)
                entry = AuditEntry.model_validate(data)
                self.entries.append(entry)
            except Exception:
                pass  # Skip corrupt entries silently

    def _persist(self, entry: AuditEntry) -> None:
        """Write a single entry to SQLite."""
        if not self._db_available:
            return
        try:
            raw = entry.model_dump_json()
            override_ts = (
                entry.human_override_timestamp.isoformat()
                if entry.human_override_timestamp
                else None
            )
            with self._db_lock:
                conn = sqlite3.connect(self.db_path)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO audit_entries VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                    )
                    """,
                    (
                        entry.entry_id,
                        entry.session_id,
                        entry.timestamp.isoformat(),
                        entry.tool_call.tool_name,
                        json.dumps(entry.tool_call.tool_args),
                        entry.tool_call.tool_category,
                        entry.verdict.risk_level.value,
                        entry.verdict.attack_type.value,
                        entry.verdict.confidence_score,
                        entry.verdict.reasoning,
                        entry.verdict.tier_used.value,
                        entry.policy_result.action.value,
                        entry.policy_result.matched_rule_id,
                        1 if entry.human_override is True else (
                            0 if entry.human_override is False else None
                        ),
                        override_ts,
                        raw,
                    ),
                )
                conn.commit()
                conn.close()
        except Exception:
            pass  # Never let storage errors block the interceptor

    def log(
        self,
        tool_call: ToolCallRequest,
        verdict: ClassificationVerdict,
        policy_result: PolicyResult,
    ) -> AuditEntry:
        """Override parent log() to also persist to SQLite."""
        entry = super().log(tool_call, verdict, policy_result)
        self._persist(entry)
        return entry

    def record_human_override(self, call_id: str, approved: bool) -> None:
        """Override parent to also update the SQLite record."""
        super().record_human_override(call_id, approved)
        if not self._db_available:
            return
        # Find and re-persist the updated entry
        for entry in self.entries:
            if entry.tool_call.tool_call_id == call_id:
                self._persist(entry)
                break

    @classmethod
    def export_sessions(cls, db_path: str = _DEFAULT_DB) -> list[str]:
        """Return all distinct session IDs stored in the database."""
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.execute("SELECT DISTINCT session_id FROM audit_entries ORDER BY timestamp DESC")
            sessions = [row[0] for row in cur.fetchall()]
            conn.close()
            return sessions
        except Exception:
            return []

    @classmethod
    def load_session(cls, session_id: str, db_path: str = _DEFAULT_DB) -> "PersistentAuditLog":
        """Reconstruct an AuditLog for a past session from the database."""
        log = cls(session_id=session_id, db_path=db_path, load_existing=True)
        return log
