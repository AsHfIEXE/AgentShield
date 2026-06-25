"""
AgentShield — Runtime Authorization Gateway for AI Agents.

The world's first pre-execution semantic authorization middleware
for agentic AI tool calls. Intercepts, classifies, and enforces
policy on every tool invocation before it executes.
"""

__version__ = "0.1.0"
__author__ = "AgentShield Team"

from agentshield.models import (
    ToolCallRequest,
    ClassificationVerdict,
    PolicyAction,
    AuditEntry,
    RiskLevel,
    AttackType,
    ActionDecision,
)
