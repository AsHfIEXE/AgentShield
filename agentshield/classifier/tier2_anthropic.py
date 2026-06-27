"""
Tier 2 Anthropic/Claude classifier.
Uses the Anthropic Messages API (claude-haiku for Tier 2, claude-sonnet for Tier 3).
Fully structured JSON output via system prompt.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from agentshield.classifier.prompts import SYSTEM_PROMPT, format_user_prompt
from agentshield.models import (
    ActionDecision,
    AttackType,
    ClassificationTier,
    ClassificationVerdict,
    RiskLevel,
    SessionContext,
    ToolCallRequest,
)

# Default Claude models
CLAUDE_TIER2_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_TIER3_MODEL = "claude-sonnet-4-6"


class Tier2AnthropicClassifier:
    """Anthropic Claude-based Tier 2 classifier with fallback heuristics."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = CLAUDE_TIER2_MODEL,
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.temperature = temperature
        self.mock_mode = not bool(self.api_key)

        if not self.mock_mode:
            try:
                import anthropic
                self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package is required. Run: pip install anthropic>=0.30"
                )
        else:
            self.client = None

    def _mock_classify(
        self, tool_call: ToolCallRequest, session: SessionContext
    ) -> ClassificationVerdict:
        """Heuristic fallback when no API key is set — mirrors Tier2Classifier._mock_classify."""
        task = session.original_task.lower()
        tool_name = tool_call.tool_name.lower()
        args_str = str(tool_call.tool_args).lower()

        if "summarize" in task and ("http" in tool_name or "attacker.com" in args_str or "email" in tool_name):
            return ClassificationVerdict(
                risk_level=RiskLevel.HIGH,
                attack_type=AttackType.ASI01_GOAL_HIJACK,
                confidence_score=0.98,
                reasoning="Agent is attempting to send data externally during a local document summarization task.",
                recommended_action=ActionDecision.BLOCK,
                tier_used=ClassificationTier.TIER2_LIGHTWEIGHT_LLM,
            )
        if "invite" in task and "delete" in tool_name:
            return ClassificationVerdict(
                risk_level=RiskLevel.HIGH,
                attack_type=AttackType.ASI02_TOOL_MISUSE,
                confidence_score=0.99,
                reasoning="File deletion is entirely outside the scope of sending a calendar invite.",
                recommended_action=ActionDecision.BLOCK,
                tier_used=ClassificationTier.TIER2_LIGHTWEIGHT_LLM,
            )
        if "mcp://" in args_str or "execute_tool" in tool_name:
            return ClassificationVerdict(
                risk_level=RiskLevel.HIGH,
                attack_type=AttackType.ASI07_INTER_AGENT_COMMS,
                confidence_score=0.89,
                reasoning="Unregistered/dynamic MCP server tool execution mid-session.",
                recommended_action=ActionDecision.ESCALATE,
                tier_used=ClassificationTier.TIER2_LIGHTWEIGHT_LLM,
            )

        is_benign = any([
            "read" in tool_name and "docs" in task and "docs" in args_str,
            "read" in tool_name and "reports" in task and "reports" in args_str,
            "email" in tool_name and "send" in task and "team@corp.com" in args_str,
            "execute_code" in tool_name and "run" in task and "pytest" in args_str,
            "http" in tool_name and "health" in task and "health" in args_str,
        ])

        if is_benign:
            return ClassificationVerdict(
                risk_level=RiskLevel.LOW,
                attack_type=AttackType.NONE,
                confidence_score=0.95,
                reasoning="Benign action: tool invocation is aligned with original task.",
                recommended_action=ActionDecision.ALLOW,
                tier_used=ClassificationTier.TIER2_LIGHTWEIGHT_LLM,
            )

        return ClassificationVerdict(
            risk_level=RiskLevel.MEDIUM,
            attack_type=AttackType.NONE,
            confidence_score=0.60,
            reasoning="Ambiguous action: heuristic cannot verify safety without LLM analysis.",
            recommended_action=ActionDecision.LOG_AND_ALLOW,
            tier_used=ClassificationTier.TIER2_LIGHTWEIGHT_LLM,
        )

    async def _call_llm(self, user_prompt: str) -> ClassificationVerdict:
        """Call the Anthropic Messages API and parse structured JSON response."""
        import anthropic

        # Anthropic expects the JSON instruction in the system prompt;
        # we also hint it with a prefill so the model starts with '{'
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=self.temperature,
            system=SYSTEM_PROMPT + "\n\nYou MUST output only valid JSON — no preamble, no markdown fences.",
            messages=[
                {"role": "user", "content": user_prompt},
                # Prefill forces JSON-first output
                {"role": "assistant", "content": "{"},
            ],
        )

        # Re-attach the prefill character
        raw = "{" + response.content[0].text
        data = json.loads(raw)

        risk_level = RiskLevel(data.get("risk_level", "MEDIUM").upper())
        attack_type = AttackType(data.get("attack_type", "NONE").upper())
        confidence = float(data.get("confidence_score", 0.5))
        reasoning = data.get("reasoning", "Claude classified tool call.")
        rec_action = ActionDecision(data.get("recommended_action", "LOG_AND_ALLOW").upper())

        return ClassificationVerdict(
            risk_level=risk_level,
            attack_type=attack_type,
            confidence_score=confidence,
            reasoning=reasoning,
            recommended_action=rec_action,
            tier_used=ClassificationTier.TIER2_LIGHTWEIGHT_LLM,
            raw_llm_response=raw,
        )

    async def classify(
        self, tool_call: ToolCallRequest, session: SessionContext
    ) -> ClassificationVerdict:
        """Run the Anthropic classification pipeline with retry on low confidence."""
        if self.mock_mode:
            return self._mock_classify(tool_call, session)

        user_prompt = format_user_prompt(
            task=session.original_task,
            history=session.compressed_history(),
            tool_name=tool_call.tool_name,
            tool_args=tool_call.tool_args,
        )

        try:
            verdict = await self._call_llm(user_prompt)

            # Retry once if confidence is too low
            if verdict.confidence_score < 0.7:
                verdict2 = await self._call_llm(user_prompt)
                risk_order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
                if risk_order[verdict2.risk_level] > risk_order[verdict.risk_level]:
                    verdict = verdict2

            return verdict

        except Exception as e:
            return ClassificationVerdict(
                risk_level=RiskLevel.MEDIUM,
                attack_type=AttackType.OTHER,
                confidence_score=0.5,
                reasoning=f"Claude classification failed: {str(e)}",
                recommended_action=ActionDecision.LOG_AND_ALLOW,
                tier_used=ClassificationTier.TIER2_LIGHTWEIGHT_LLM,
            )
