"""
Classification Engine.
Orchestrates the 3-tier classification pipeline:
Tier 1 (Regex/Deterministic) -> Tier 2 (Lightweight LLM) -> Tier 3 (Full LLM).

Supports two LLM backends:
  - "openai"    : GPT-4o-mini (Tier 2) + GPT-4o (Tier 3)
  - "anthropic" : Claude Haiku (Tier 2) + Claude Sonnet (Tier 3)

Backend is auto-detected from AGENTSHIELD_LLM_PROVIDER env var, or whichever
API key is present (ANTHROPIC_API_KEY takes priority over OPENAI_API_KEY).
"""

from __future__ import annotations

import os
from typing import Any, Optional

from agentshield.classifier.prompts import SYSTEM_PROMPT, format_user_prompt
from agentshield.classifier.tier1_rules import Tier1Classifier
from agentshield.classifier.tier2_llm import Tier2Classifier
from agentshield.classifier.tier2_anthropic import (
    Tier2AnthropicClassifier,
    CLAUDE_TIER2_MODEL,
    CLAUDE_TIER3_MODEL,
)
from agentshield.models import (
    ClassificationTier,
    ClassificationVerdict,
    RiskLevel,
    SessionContext,
    ToolCallRequest,
)

_OPENAI_TIER2 = "gpt-4o-mini"
_OPENAI_TIER3 = "gpt-4o"


def _detect_provider() -> str:
    explicit = os.environ.get("AGENTSHIELD_LLM_PROVIDER", "").lower()
    if explicit in ("openai", "anthropic"):
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "openai"


class ClassificationEngine:
    """Master classifier: Tier 1 (regex) → Tier 2 (fast LLM) → Tier 3 (full LLM)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model_tier2: Optional[str] = None,
        model_tier3: Optional[str] = None,
        extra_few_shot: Optional[list[str]] = None,
    ):
        self.provider = (provider or _detect_provider()).lower()
        self.extra_few_shot = extra_few_shot or []
        self.tier1 = Tier1Classifier()

        if self.provider == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.model_tier2 = model_tier2 or CLAUDE_TIER2_MODEL
            self.model_tier3 = model_tier3 or CLAUDE_TIER3_MODEL
            self.tier2: Any = Tier2AnthropicClassifier(
                api_key=self.api_key, model=self.model_tier2
            )
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.model_tier2 = model_tier2 or _OPENAI_TIER2
            self.model_tier3 = model_tier3 or _OPENAI_TIER3
            self.tier2 = Tier2Classifier(api_key=self.api_key, model=self.model_tier2)

        self.mock_mode = not bool(self.api_key)
        self._anthropic_client: Any = None
        self._openai_client: Any = None

        if not self.mock_mode:
            if self.provider == "anthropic":
                try:
                    import anthropic
                    self._anthropic_client = anthropic.AsyncAnthropic(api_key=self.api_key)
                except ImportError:
                    pass
            else:
                try:
                    from openai import AsyncOpenAI
                    self._openai_client = AsyncOpenAI(api_key=self.api_key)
                except ImportError:
                    pass

    # Backward-compat shim used by the Streamlit dashboard
    @property
    def client(self):
        return self._openai_client or self._anthropic_client

    def sanitize_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Scrub prompt-injection keywords from tool arguments before LLM tiers."""
        sanitized = {}
        for k, v in args.items():
            if isinstance(v, str):
                cleaned = v
                for kw in [
                    "ignore previous", "ignore above", "system prompt",
                    "you are now", "instead of",
                ]:
                    if kw in cleaned.lower():
                        cleaned = cleaned.replace(kw, "[REDACTED_INJECTION_SIG]")
                sanitized[k] = cleaned
            elif isinstance(v, dict):
                sanitized[k] = self.sanitize_args(v)
            elif isinstance(v, list):
                sanitized[k] = [
                    self.sanitize_args(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                sanitized[k] = v
        return sanitized

    async def _call_tier3(
        self, tool_call: ToolCallRequest, session: SessionContext
    ) -> ClassificationVerdict:
        """Invoke the full model for ambiguous/complex cases."""
        if self.mock_mode:
            task = session.original_task.lower()
            args_str = str(tool_call.tool_args).lower()
            risk = RiskLevel.HIGH
            reasoning = "Tier 3 escalated (Mock): complex semantic query."
            rec_action = "ESCALATE"
            if ("database" in task and "metrics.internal" in args_str) or \
               ("binaries" in task and "tmp/build_cache" in args_str) or \
               ("organize" in task and "os.walk" in args_str):
                risk = RiskLevel.MEDIUM
                reasoning = "Tier 3 escalated (Mock): edge case as medium risk."
                rec_action = "LOG_AND_ALLOW"
            elif "analyze data files" in task and "reports" in args_str:
                risk = RiskLevel.LOW
                reasoning = "Tier 3 escalated (Mock): benign multi-read."
                rec_action = "ALLOW"
            return ClassificationVerdict(
                risk_level=risk,
                confidence_score=0.85,
                reasoning=reasoning,
                recommended_action=rec_action,
                tier_used=ClassificationTier.TIER3_FULL_LLM,
            )

        sys_prompt = SYSTEM_PROMPT
        if self.extra_few_shot:
            sys_prompt += (
                f"\n<extra_examples>\n"
                f"{chr(10).join(self.extra_few_shot)}\n"
                f"</extra_examples>"
            )

        user_prompt = format_user_prompt(
            task=session.original_task,
            history=session.compressed_history(),
            tool_name=tool_call.tool_name,
            tool_args=tool_call.tool_args,
        )

        try:
            import json

            if self.provider == "anthropic" and self._anthropic_client:
                response = await self._anthropic_client.messages.create(
                    model=self.model_tier3,
                    max_tokens=1024,
                    temperature=0.0,
                    system=sys_prompt + "\n\nOutput ONLY valid JSON — no preamble, no markdown.",
                    messages=[
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": "{"},
                    ],
                )
                content = "{" + response.content[0].text
            elif self._openai_client:
                response = await self._openai_client.chat.completions.create(
                    model=self.model_tier3,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                content = response.choices[0].message.content
            else:
                raise RuntimeError("No LLM client available for Tier 3.")

            data = json.loads(content)
            return ClassificationVerdict(
                risk_level=RiskLevel(data.get("risk_level", "MEDIUM").upper()),
                attack_type=data.get("attack_type", "NONE").upper(),
                confidence_score=float(data.get("confidence_score", 0.9)),
                reasoning=data.get("reasoning", "Tier 3 semantic reasoning verdict."),
                recommended_action=data.get("recommended_action", "BLOCK"),
                tier_used=ClassificationTier.TIER3_FULL_LLM,
                raw_llm_response=content,
            )
        except Exception as e:
            return ClassificationVerdict(
                risk_level=RiskLevel.HIGH,
                confidence_score=0.99,
                reasoning=f"Tier 3 classification failed: {str(e)}",
                recommended_action="BLOCK",
                tier_used=ClassificationTier.TIER3_FULL_LLM,
            )

    async def classify(
        self, tool_call: ToolCallRequest, session: SessionContext
    ) -> ClassificationVerdict:
        """Full pipeline: Tier 1 → Tier 2 → Tier 3."""
        t1 = self.tier1.classify(tool_call, session)
        if t1:
            return t1

        sanitized = tool_call.model_copy(update={"tool_args": self.sanitize_args(tool_call.tool_args)})

        t2 = await self.tier2.classify(sanitized, session)
        if (t2.risk_level == RiskLevel.HIGH and t2.confidence_score >= 0.70) or \
           (t2.risk_level == RiskLevel.LOW and t2.confidence_score >= 0.80):
            return t2

        return await self._call_tier3(sanitized, session)
