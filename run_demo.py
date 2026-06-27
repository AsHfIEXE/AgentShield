#!/usr/bin/env python3
"""
AgentShield Command Line Demo Interface.
Provides terminal execution, formatted audit outputs, and security reporting.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Load .env before importing anything else that reads env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure package is on the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agentshield.demo import DemoRunner
from agentshield.scenarios import ALL_SCENARIOS


# ── ANSI Terminal Colors ──────────────────────────────────────────────────────
class Colors:
    GREEN = "\033[92m"
    AMBER = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def color_verdict(verdict: str) -> str:
    if verdict == "BLOCK":
        return f"{Colors.RED}{Colors.BOLD}BLOCKED{Colors.END}"
    if verdict == "ESCALATE":
        return f"{Colors.AMBER}{Colors.BOLD}PENDING APPROVAL{Colors.END}"
    if verdict == "ALLOW":
        return f"{Colors.GREEN}ALLOWED{Colors.END}"
    return f"{Colors.BLUE}LOGGED (ALLOW){Colors.END}"


def color_risk(risk: str) -> str:
    if risk == "HIGH":
        return f"{Colors.RED}{risk}{Colors.END}"
    if risk == "MEDIUM":
        return f"{Colors.AMBER}{risk}{Colors.END}"
    return f"{Colors.GREEN}{risk}{Colors.END}"


async def run_cli_demo(api_key: str | None, scenario_filter: str) -> int:
    runner = DemoRunner(api_key=api_key)

    # Filter scenarios
    to_run = []
    if scenario_filter == "all":
        to_run = ALL_SCENARIOS
        print(f"Running all {len(ALL_SCENARIOS)} scenarios...")
    elif scenario_filter == "demo":
        from agentshield.scenarios import DEMO_SCENARIOS
        to_run = DEMO_SCENARIOS
        print("Running core 3 demo attack scenarios...")
    else:
        to_run = [s for s in ALL_SCENARIOS if s.scenario_id == scenario_filter]
        if not to_run:
            print(f"{Colors.RED}Scenario '{scenario_filter}' not found.{Colors.END}")
            return 1
        print(f"Running scenario '{scenario_filter}'...")

    # Run execution loop
    entries = []
    for s in to_run:
        entry = await runner.run_scenario(s)
        entries.append((s, entry))

    # Print Table Header
    print("\n" + "=" * 110)
    print(
        f"{Colors.BOLD}{'Scenario ID':<25} | {'Tool':<15} | {'Risk':<10} | {'Verdict':<20} | {'Reasoning':<30}{Colors.END}"
    )
    print("=" * 110)

    for scenario, entry in entries:
        t_name = entry.tool_call.tool_name
        risk = entry.verdict.risk_level.value
        verdict = entry.policy_result.action.value
        reason = entry.verdict.reasoning

        # Truncate strings
        t_name = t_name[:15]
        reason = reason[:35] + "..." if len(reason) > 35 else reason

        print(
            f"{scenario.scenario_id:<25} | {t_name:<15} | {color_risk(risk):<19} | {color_verdict(verdict):<31} | {reason:<30}"
        )

    print("=" * 110)

    # Compute & Print Accuracy Summary
    report = runner.get_accuracy_report()
    print(f"\n{Colors.BOLD}🔍 System Authorization Performance Metrics:{Colors.END}")
    acc_color = Colors.GREEN if report["accuracy"] >= 0.90 else Colors.AMBER
    print(
        f"  - Verdict Match Accuracy : {acc_color}{report['accuracy'] * 100:.1f}%{Colors.END}"
    )
    print(
        f"  - Classification Precision: {Colors.GREEN}{report['precision'] * 100:.1f}%{Colors.END}"
    )
    print(
        f"  - Classification Recall   : {Colors.GREEN}{report['recall'] * 100:.1f}%{Colors.END}"
    )
    print(
        f"  - False Positive Rate    : {Colors.GREEN}{report['false_positive_rate'] * 100:.1f}%{Colors.END}"
    )

    if report["mismatches"]:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️ Mismatches Detected:{Colors.END}")
        for m in report["mismatches"]:
            print(
                f"  - {Colors.BOLD}{m['scenario']}{Colors.END}: expected {m['expected']}, actual {m['actual']}."
            )
            print(f"    Reason: {m['reasoning']}")
        return 1

    print(
        f"\n{Colors.GREEN}{Colors.BOLD}✅ Clean Run: All security decisions match expected postures.{Colors.END}\n"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentShield CLI Scenario Runner")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API Key (falls back to env OPENAI_API_KEY or mock mode)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Specify scenario ID to execute, 'all' (default), or 'demo' (core 3)",
    )
    args = parser.parse_args()

    # Async execute helper
    exit_code = asyncio.run(run_cli_demo(args.api_key, args.scenario))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
