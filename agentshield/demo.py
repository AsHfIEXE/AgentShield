"""
Demo runner module.
Provides an execution harness to run attack/benign scenarios through AgentShield.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from agentshield.audit.logger import AuditLog
from agentshield.classifier.engine import ClassificationEngine
from agentshield.interceptor.base import BaseInterceptor
from agentshield.models import AuditEntry, SessionContext
from agentshield.policy.engine import PolicyEngine
from agentshield.scenarios import ALL_SCENARIOS, DEMO_SCENARIOS, AttackScenario


class DemoRunner:
    """Orchestrates test scenarios through the AgentShield interception pipeline."""

    def __init__(self, api_key: Optional[str] = None):
        self.classifier = ClassificationEngine(api_key=api_key)
        self.policy = PolicyEngine()
        self.audit = AuditLog(session_id="demo-session")
        self.interceptor = BaseInterceptor(self.classifier, self.policy, self.audit)

    async def run_scenario(self, scenario: AttackScenario) -> AuditEntry:
        """Run a single scenario through classification and policy gates."""
        # 1. Establish session state
        session = SessionContext(
            session_id="demo-session",
            original_task=scenario.original_task,
            registered_tools_at_start=["read_file", "write_file", "delete_file", "list_files", "send_email", "http_request", "execute_code"]
        )

        # 2. Re-create past action history
        for call in scenario.action_history:
            session.add_action(call)

        # 3. Intercept & evaluate current tool call
        request = scenario.tool_call.model_copy()
        request.session_id = session.session_id

        policy_result, verdict = await self.interceptor.intercept(request, session)

        # 4. Extract the created audit log entry
        return self.audit.entries[-1]

    async def run_demo_scenarios(self) -> list[AuditEntry]:
        """Run the core 3 demo attack scenarios."""
        results = []
        for s in DEMO_SCENARIOS:
            entry = await self.run_scenario(s)
            results.append(entry)
        return results

    async def run_all_scenarios(self) -> list[AuditEntry]:
        """Run all 13 canonical test suite scenarios."""
        results = []
        for s in ALL_SCENARIOS:
            entry = await self.run_scenario(s)
            results.append(entry)
        return results

    def get_audit_log(self) -> AuditLog:
        """Retrieve the current session's audit log."""
        return self.audit

    def get_accuracy_report(self) -> dict[str, Any]:
        """Compile a validation summary checking actual actions against scenario expectations."""
        total = 0
        matches = 0
        mismatches = []
        confusion = {
            "BLOCK": {"BLOCK": 0, "ALLOW": 0, "LOG_AND_ALLOW": 0, "ESCALATE": 0},
            "ALLOW": {"BLOCK": 0, "ALLOW": 0, "LOG_AND_ALLOW": 0, "ESCALATE": 0},
            "LOG_AND_ALLOW": {"BLOCK": 0, "ALLOW": 0, "LOG_AND_ALLOW": 0, "ESCALATE": 0},
            "ESCALATE": {"BLOCK": 0, "ALLOW": 0, "LOG_AND_ALLOW": 0, "ESCALATE": 0},
        }

        # Match log entries back to scenarios
        for s in ALL_SCENARIOS:
            # Find corresponding entry
            matched_entry = None
            for entry in self.audit.entries:
                if entry.tool_call.tool_name == s.tool_call.tool_name and \
                   entry.tool_call.tool_args == s.tool_call.tool_args:
                    matched_entry = entry
                    break

            if not matched_entry:
                continue

            total += 1
            expected = s.expected_action.value
            actual = matched_entry.policy_result.action.value

            confusion[expected][actual] += 1

            if expected == actual or (expected == "ALLOW" and actual == "LOG_AND_ALLOW") or (expected == "LOG_AND_ALLOW" and actual == "ALLOW"):
                # Treat ALLOW and LOG_AND_ALLOW as functionally equivalent for benign authorization
                matches += 1
            else:
                mismatches.append({
                    "scenario": s.scenario_id,
                    "expected": expected,
                    "actual": actual,
                    "reasoning": matched_entry.verdict.reasoning,
                })

        accuracy = matches / total if total > 0 else 0.0

        # Calculate standard metrics
        # Precision: true positive / (true positive + false positive)
        # Recall: true positive / (true positive + false negative)
        tp = confusion["BLOCK"]["BLOCK"] + confusion["ESCALATE"]["ESCALATE"]
        fp = confusion["ALLOW"]["BLOCK"] + confusion["ALLOW"]["ESCALATE"] + confusion["LOG_AND_ALLOW"]["BLOCK"] + confusion["LOG_AND_ALLOW"]["ESCALATE"]
        fn = confusion["BLOCK"]["ALLOW"] + confusion["BLOCK"]["LOG_AND_ALLOW"] + confusion["ESCALATE"]["ALLOW"] + confusion["ESCALATE"]["LOG_AND_ALLOW"]
        tn = confusion["ALLOW"]["ALLOW"] + confusion["ALLOW"]["LOG_AND_ALLOW"] + confusion["LOG_AND_ALLOW"]["ALLOW"] + confusion["LOG_AND_ALLOW"]["LOG_AND_ALLOW"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "total_evaluated": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fpr,
            "mismatches": mismatches,
            "confusion_matrix": confusion,
        }


async def main() -> None:
    """Command-line interface to execute the demo runner offline."""
    print("==================================================")
    print("🛡️  AgentShield — Live Demo Test Execution Gate")
    print("==================================================")

    runner = DemoRunner()
    print("Executing all scenarios through AgentShield pipeline (mock mode)...")
    await runner.run_all_scenarios()

    report = runner.get_accuracy_report()
    print(f"\nAccuracy Score: {report['accuracy']*100:.1f}%")
    print(f"Precision Rate: {report['precision']*100:.1f}%")
    print(f"Recall Rate   : {report['recall']*100:.1f}%")
    print(f"False Pos. Rate: {report['false_positive_rate']*100:.1f}%")

    if report["mismatches"]:
        print("\nMismatches detected:")
        for m in report["mismatches"]:
            print(f"  - {m['scenario']}: expected {m['expected']}, got {m['actual']} ({m['reasoning']})")
    else:
        print("\n✅ All scenarios matched expectations perfectly!")


if __name__ == "__main__":
    asyncio.run(main())
