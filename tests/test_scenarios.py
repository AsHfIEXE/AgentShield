"""
Integration tests executing all 13 canonical scenarios.
Verifies classifier/policy accuracy against ground truth targets.
"""

from __future__ import annotations

import pytest

from agentshield.demo import DemoRunner
from agentshield.models import ActionDecision, PolicyAction
from agentshield.scenarios import ALL_SCENARIOS


import asyncio

def test_all_scenarios_evaluation():
    # Run the demo runner in mock/offline mode
    runner = DemoRunner()

    async def _run():
        for scenario in ALL_SCENARIOS:
            entry = await runner.run_scenario(scenario)

            expected_action = scenario.expected_action.value
            actual_action = entry.policy_result.action.value

            # Match ALLOW and LOG_AND_ALLOW as equivalent benign outcomes
            if expected_action == "ALLOW":
                assert actual_action in ["ALLOW", "LOG_AND_ALLOW"], f"Scenario {scenario.scenario_id} failed: expected ALLOW/LOG_AND_ALLOW, got {actual_action}"
            elif expected_action == "LOG_AND_ALLOW":
                assert actual_action in ["ALLOW", "LOG_AND_ALLOW"], f"Scenario {scenario.scenario_id} failed: expected ALLOW/LOG_AND_ALLOW, got {actual_action}"
            else:
                assert actual_action == expected_action, f"Scenario {scenario.scenario_id} failed: expected {expected_action}, got {actual_action}"

    asyncio.run(_run())


def test_accuracy_report_generation():
    runner = DemoRunner()
    
    async def _run():
        await runner.run_all_scenarios()

    asyncio.run(_run())

    report = runner.get_accuracy_report()
    assert report["total_evaluated"] == len(ALL_SCENARIOS)
    assert report["accuracy"] == 1.0  # Under deterministic mock mode, it must be 100%
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["false_positive_rate"] == 0.0
    assert len(report["mismatches"]) == 0
    assert "confusion_matrix" in report
