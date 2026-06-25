"""
Tier 1 deterministic classifier.
Zero-latency regex-based pattern matching and sequence detection.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agentshield.models import (
    ActionDecision,
    AttackType,
    ClassificationTier,
    ClassificationVerdict,
    RiskLevel,
    SessionContext,
    ToolCallRequest,
)

# ── Compiled Patterns ────────────────────────────────────────────────────────

TIER1_PATTERNS = {
    "path_traversal": re.compile(r"(\.\./|/etc/passwd|/etc/shadow|~/\.ssh/|/root/)"),
    "code_injection": re.compile(
        r"(subprocess\.|os\.system|eval\(|exec\(|__import__|shutil\.)"
    ),
    "credential_exposure": re.compile(
        r"(api[_-]?key|secret[_-]?key|password|bearer\s+[a-zA-Z0-9]+|private[_-]?key|ssh[_-]?key)",
        re.IGNORECASE,
    ),
    "known_exfil_domains": re.compile(
        r"(ngrok\.io|requestbin|webhook\.site|burpcollaborator|attacker\.com)",
        re.IGNORECASE,
    ),
    "dangerous_paths": re.compile(
        r"(/etc/|/root/|~/\.ssh/|/proc/|/sys/|\.\./)"
    ),
}


class Tier1Classifier:
    """Deterministic, regex-based and heuristic classifier."""

    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize a tool based on its name."""
        from agentshield.mock_tools import get_tool_category
        return get_tool_category(tool_name)

    def _scan_value(self, val: Any) -> list[tuple[str, str]]:
        """Recursively scan arguments for pattern matches."""
        matches = []
        if isinstance(val, str):
            for pat_name, regex in TIER1_PATTERNS.items():
                match = regex.search(val)
                if match:
                    matches.append((pat_name, match.group(0)))
        elif isinstance(val, dict):
            for v in val.values():
                matches.extend(self._scan_value(v))
        elif isinstance(val, (list, tuple)):
            for v in val:
                matches.extend(self._scan_value(v))
        return matches

    def _scan_args(self, tool_args: dict[str, Any]) -> list[tuple[str, str]]:
        """Scan all values in arguments for sensitive patterns."""
        return self._scan_value(tool_args)

    def check_sequence(
        self, session: SessionContext, current_call: ToolCallRequest
    ) -> Optional[ClassificationVerdict]:
        """Check the session history for suspicious multi-step sequences."""
        history = session.action_history
        if not history:
            return None

        current_category = self._categorize_tool(current_call.tool_name)

        # Sequence Pattern 1: read sensitive path -> exfil to network
        # Find if a sensitive file was read in the past
        has_read_sensitive = False
        for call in history:
            cat = self._categorize_tool(call.tool_name)
            if cat == "filesystem":
                # Check if it scanned /etc or similar
                matches = self._scan_args(call.tool_args)
                if any(m[0] in ["path_traversal", "dangerous_paths"] for m in matches):
                    has_read_sensitive = True
                    break

        if has_read_sensitive and current_category in ["network", "email"]:
            return ClassificationVerdict(
                risk_level=RiskLevel.HIGH,
                attack_type=AttackType.ASI02_TOOL_MISUSE,
                confidence_score=0.95,
                reasoning="Suspicious sequence: sensitive filesystem read followed by outbound network communication.",
                recommended_action=ActionDecision.BLOCK,
                tier_used=ClassificationTier.SEQUENCE_DETECTION,
            )

        # Sequence Pattern 2: list_files -> compress -> network
        has_listed = any(self._categorize_tool(c.tool_name) == "filesystem" and "list" in c.tool_name.lower() for c in history)
        has_compressed = any("compress" in c.tool_name.lower() or "zip" in c.tool_name.lower() for c in history)
        if has_listed and has_compressed and current_category in ["network", "email"]:
            return ClassificationVerdict(
                risk_level=RiskLevel.HIGH,
                attack_type=AttackType.ASI02_TOOL_MISUSE,
                confidence_score=0.92,
                reasoning="Suspicious sequence: list files, compress, then attempt to send data externally.",
                recommended_action=ActionDecision.BLOCK,
                tier_used=ClassificationTier.SEQUENCE_DETECTION,
            )

        # Sequence Pattern 3: read any file -> send network/email containing file content or potential exfil args
        has_read = any(self._categorize_tool(c.tool_name) == "filesystem" and "read" in c.tool_name.lower() for c in history)
        if has_read and current_category in ["network", "email"]:
            # Check if there are large data payloads or suspicious strings in current arguments
            args_str = str(current_call.tool_args).lower()
            if len(args_str) > 100 or any(w in args_str for w in ["data", "body", "content", "payload"]):
                return ClassificationVerdict(
                    risk_level=RiskLevel.HIGH,
                    attack_type=AttackType.ASI01_GOAL_HIJACK,
                    confidence_score=0.88,
                    reasoning="Suspicious sequence: reading files followed by network request containing potential data payload.",
                    recommended_action=ActionDecision.BLOCK,
                    tier_used=ClassificationTier.SEQUENCE_DETECTION,
                )

        return None

    def classify(
        self, tool_call: ToolCallRequest, session: SessionContext
    ) -> Optional[ClassificationVerdict]:
        """Runs fast, deterministic regex checks and rule evaluations."""
        # 1. MCP registration / supply chain attack (ASI07)
        # If new tool registered mid-session
        if (
            session.registered_tools_at_start
            and tool_call.tool_name not in session.registered_tools_at_start
        ):
            return ClassificationVerdict(
                risk_level=RiskLevel.HIGH,
                attack_type=AttackType.ASI07_INTER_AGENT_COMMS,
                confidence_score=0.98,
                reasoning=f"MCP supply chain anomaly: Tool '{tool_call.tool_name}' was not in the session start registry.",
                recommended_action=ActionDecision.ESCALATE,
                tier_used=ClassificationTier.TIER1_DETERMINISTIC,
            )

        # 2. Check for malicious argument patterns
        matches = self._scan_args(tool_call.tool_args)
        tool_category = self._categorize_tool(tool_call.tool_name)

        # Prioritize checking code_injection first
        for pat_name, matched_text in matches:
            if pat_name == "code_injection":
                return ClassificationVerdict(
                    risk_level=RiskLevel.HIGH,
                    attack_type=AttackType.ASI05_CODE_EXECUTION,
                    confidence_score=0.99,
                    reasoning=f"Code injection signature '{matched_text}' detected in tool arguments.",
                    recommended_action=ActionDecision.BLOCK,
                    tier_used=ClassificationTier.TIER1_DETERMINISTIC,
                )

        # For filesystem tools, check path traversal and dangerous paths
        if tool_category == "filesystem":
            # 1. Traversal check (if '..' is present in any matched text)
            for pat_name, matched_text in matches:
                if (pat_name in ["path_traversal", "dangerous_paths"]) and ".." in matched_text:
                    return ClassificationVerdict(
                        risk_level=RiskLevel.HIGH,
                        attack_type=AttackType.ASI02_TOOL_MISUSE,
                        confidence_score=0.97,
                        reasoning=f"Path traversal signature '{matched_text}' detected in tool arguments.",
                        recommended_action=ActionDecision.BLOCK,
                        tier_used=ClassificationTier.TIER1_DETERMINISTIC,
                    )
            # 2. Direct sensitive path access check (if it matched but does not contain '..')
            for pat_name, matched_text in matches:
                if pat_name in ["dangerous_paths", "path_traversal"]:
                    return ClassificationVerdict(
                        risk_level=RiskLevel.HIGH,
                        attack_type=AttackType.ASI06_MEMORY_POISONING,
                        confidence_score=0.96,
                        reasoning=f"Attempted access to dangerous filesystem path '{matched_text}' via '{tool_call.tool_name}'.",
                        recommended_action=ActionDecision.BLOCK,
                        tier_used=ClassificationTier.TIER1_DETERMINISTIC,
                    )

        for pat_name, matched_text in matches:
            # Domain exfiltration is universally blocked
            if pat_name == "known_exfil_domains":
                return ClassificationVerdict(
                    risk_level=RiskLevel.HIGH,
                    attack_type=AttackType.ASI01_GOAL_HIJACK,
                    confidence_score=0.99,
                    reasoning=f"Network exfiltration domain '{matched_text}' detected in tool arguments.",
                    recommended_action=ActionDecision.BLOCK,
                    tier_used=ClassificationTier.TIER1_DETERMINISTIC,
                )

            # Credential exposure (HIGH risk in most tools, except maybe credential managers)
            if pat_name == "credential_exposure":
                # If network or email tool, this is high risk
                if tool_category in ["network", "email"]:
                    return ClassificationVerdict(
                        risk_level=RiskLevel.HIGH,
                        attack_type=AttackType.ASI03_IDENTITY_ABUSE,
                        confidence_score=0.95,
                        reasoning=f"Credential exposure signature '{matched_text}' detected in outbound '{tool_call.tool_name}' call.",
                        recommended_action=ActionDecision.BLOCK,
                        tier_used=ClassificationTier.TIER1_DETERMINISTIC,
                    )

        # 3. Check sequence logic
        seq_verdict = self.check_sequence(session, tool_call)
        if seq_verdict:
            return seq_verdict

        return None
