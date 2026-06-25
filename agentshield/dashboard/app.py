"""
Streamlit Audit Dashboard app.
Provides a premium visual interface, live feeds, policy editor, and control panel.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Put agentshield on the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agentshield.audit.feedback import MisclassificationStore
from agentshield.demo import DemoRunner
from agentshield.models import ActionDecision, PolicyAction, ToolCallRequest
from agentshield.scenarios import ALL_SCENARIOS, DEMO_SCENARIOS

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="AgentShield — Runtime Authorization Gateway",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Premium Styling ──────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Gradient Header Accent */
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.85;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #eef2f6;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }

    /* Status Badges */
    .badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-high {
        background-color: #ffe5e5;
        color: #d93838;
        border: 1px solid #ffcccc;
    }
    .badge-medium {
        background-color: #fff2cc;
        color: #b27a00;
        border: 1px solid #ffe599;
    }
    .badge-low {
        background-color: #e2f0d9;
        color: #385723;
        border: 1px solid #c5e0b4;
    }

    /* Highlighted Rows & Pulse */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(217, 56, 56, 0.4); }
        70% { box-shadow: 0 0 0 8px rgba(217, 56, 56, 0); }
        100% { box-shadow: 0 0 0 0 rgba(217, 56, 56, 0); }
    }
    .alert-block {
        border-left: 6px solid #d93838;
        background-color: #fff9f9;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
        animation: pulse-red 2s infinite;
    }
    .alert-escalate {
        border-left: 6px solid #f5b041;
        background-color: #fefaf0;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .alert-allow {
        border-left: 6px solid #2ecc71;
        background-color: #f4fbf7;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Auto-Refresh Setup (Every 2 Seconds) ──────────────────────────────────────
st_autorefresh(interval=2000, key="audit_refresh")

# ── Initialize App State ──────────────────────────────────────────────────────
if "demo_runner" not in st.session_state:
    st.session_state.demo_runner = DemoRunner()
if "feedback_store" not in st.session_state:
    st.session_state.feedback_store = MisclassificationStore()
if "last_accuracy_report" not in st.session_state:
    st.session_state.last_accuracy_report = None

runner = st.session_state.demo_runner
audit_log = runner.get_audit_log()
policy_engine = runner.policy
approval_queue = policy_engine.human_approval_queue
feedback_store = st.session_state.feedback_store

# ── Sidebar Panel: Controls & Settings ────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/000000/shield-with-crown.png",
        width=80,
    )
    st.markdown("### **AgentShield Control**")

    # API Configuration
    api_key_input = st.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Provide a key to use live GPT-4o-mini classification. Otherwise mock rules will fire.",
    )
    if api_key_input:
        runner.classifier.api_key = api_key_input
        runner.classifier.tier2.api_key = api_key_input
        runner.classifier.mock_mode = False
        runner.classifier.tier2.mock_mode = False
        runner.classifier.client = runner.classifier.tier2.client = (
            st.session_state.demo_runner.classifier.tier2.client
        )
    else:
        runner.classifier.mock_mode = True
        runner.classifier.tier2.mock_mode = True

    st.markdown("---")

    # Action Triggers
    st.markdown("#### **Harness Scenarios**")

    if st.button("▶️ Run Core 3 Demos", use_container_width=True):
        asyncio.run(runner.run_demo_scenarios())
        st.success("Successfully executed core attacks!")

    if st.button("🔄 Run Full Test Suite (13)", use_container_width=True):
        asyncio.run(runner.run_all_scenarios())
        st.session_state.last_accuracy_report = runner.get_accuracy_report()
        st.success("Ran entire test suite!")

    if st.button("🗑️ Clear Logs", use_container_width=True):
        audit_log.entries.clear()
        approval_queue.clear_resolved()
        st.session_state.last_accuracy_report = None
        st.info("Logs cleared.")

    st.markdown("---")

    # Replay Attack Scenario Selectbox
    st.markdown("#### **Replay Attack Scenarios**")
    attack_options = {s.scenario_id: s for s in ALL_SCENARIOS if s.expected_risk == "HIGH"}
    selected_attack_id = st.selectbox(
        "Select Attack Scenario",
        options=list(attack_options.keys()),
    )
    if st.button("💥 Replay Attack Live", use_container_width=True):
        scenario = attack_options[selected_attack_id]
        # Temporarily clear the matching entry in log to demonstrate block firing
        asyncio.run(runner.run_scenario(scenario))
        st.warning(f"Replayed {selected_attack_id} live!")

    st.markdown("---")

    # Policy Editor
    st.markdown("#### **Policy Rule Configuration**")
    policy_yaml = st.text_area(
        "Edit YAML Policy Rules",
        value=policy_engine.to_yaml(),
        height=320,
    )
    if st.button("💾 Apply Updated Policy", use_container_width=True):
        try:
            policy_engine.load_yaml(policy_yaml)
            st.success("Policy reloaded successfully!")
        except Exception as e:
            st.error(f"Failed to parse policy YAML: {str(e)}")

# ── Main View Panel ──────────────────────────────────────────────────────────

# Header banner
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">🛡️ AgentShield</h1>
        <p class="header-subtitle">
            The world's first runtime authorization gateway for AI agents.
            Intercepting and validating semantic intent at the tool-call layer before execution.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Statistics Dashboard Row
stats = audit_log.stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <h5 style="color: #6c757d; margin:0;">Total Interceptions</h5>
            <h2 style="margin: 0.5rem 0 0 0; color: #1e3c72; font-weight:800;">{stats['total']}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <h5 style="color: #6c757d; margin:0;">Blocked Attacks</h5>
            <h2 style="margin: 0.5rem 0 0 0; color: #d93838; font-weight:800;">{stats['blocked']}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <h5 style="color: #6c757d; margin:0;">Escalated Calls</h5>
            <h2 style="margin: 0.5rem 0 0 0; color: #f5b041; font-weight:800;">{stats['escalated']}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <h5 style="color: #6c757d; margin:0;">False Positives</h5>
            <h2 style="margin: 0.5rem 0 0 0; color: #2ecc71; font-weight:800;">{stats['human_overrides_allow']}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Dynamic Escalation Approval Queue ─────────────────────────────────────────
pending_items = approval_queue.get_pending()
if pending_items:
    st.markdown("### ⚠️ **Human Operator Escalation Queue**")
    for item in pending_items:
        call_id = item["call_id"]
        st.markdown(
            f"""
            <div class="alert-escalate">
                <strong>[ESCALATED ACTION ENCOUNTERED]</strong> Agent is attempting to invoke tool 
                <code>{item['tool_name']}</code>.
                <br><em>Risk Reasoning: {item['reasoning']}</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_details, col_actions = st.columns([3, 1])
        with col_details:
            st.json(item["tool_args"])
        with col_actions:
            # We must use unique keys since Streamlit reruns frequently
            st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
            if st.button("🟢 Approve & Allow", key=f"app_{call_id}", use_container_width=True):
                # Resolve as approved, and log false positive feedback
                approval_queue.resolve(call_id, True)
                # Find matching request and verdict in audit log to store
                for entry in audit_log.entries:
                    if entry.tool_call.tool_call_id == call_id:
                        feedback_store.record_override(entry.tool_call, entry.verdict, True)
                        # Add feedback into class engine examples
                        runner.classifier.extra_few_shot = (
                            feedback_store.as_few_shot_examples()
                        )
                        break
                st.success("Action approved.")
                st.rerun()

            if st.button("🔴 Block Execution", key=f"blk_{call_id}", use_container_width=True):
                approval_queue.resolve(call_id, False)
                for entry in audit_log.entries:
                    if entry.tool_call.tool_call_id == call_id:
                        feedback_store.record_override(entry.tool_call, entry.verdict, False)
                        break
                st.error("Action blocked.")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

# ── Live Gateway Interception Feed ────────────────────────────────────────────
st.markdown("### 📋 **Live Gateway Interception Feed**")
df = audit_log.as_dataframe()

if not df.empty:
    # Color code cells or display a styled table using custom containers
    # Streamlit dataframe with clean columns
    st.dataframe(
        df[
            [
                "timestamp",
                "tool_name",
                "args_summary",
                "risk_level",
                "confidence",
                "verdict",
                "reasoning",
                "human_override",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # Highlight blocked records below with premium styles
    st.markdown("#### **Detailed Incident Log Analysis**")
    for entry in reversed(audit_log.entries):
        action = entry.policy_result.action
        timestamp_str = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        if action == PolicyAction.BLOCK:
            st.markdown(
                f"""
                <div class="alert-block">
                    <strong>[BLOCKED] {entry.tool_call.tool_name}</strong> at {timestamp_str}
                    <br><strong>Args:</strong> <code>{entry.tool_call.args_summary(150)}</code>
                    <br><strong>Reason:</strong> {entry.verdict.reasoning} (Confidence: {entry.verdict.confidence_score:.2f}) | <strong>Rule Trigger:</strong> <code>{entry.policy_result.matched_rule_id}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif action == PolicyAction.ESCALATE:
            status = "Awaiting review"
            if entry.human_override is True:
                status = "<span style='color:green;'>Approved by Operator</span>"
            elif entry.human_override is False:
                status = "<span style='color:red;'>Denied by Operator</span>"

            st.markdown(
                f"""
                <div class="alert-escalate">
                    <strong>[ESCALATED] {entry.tool_call.tool_name}</strong> at {timestamp_str} | Status: <strong>{status}</strong>
                    <br><strong>Args:</strong> <code>{entry.tool_call.args_summary(150)}</code>
                    <br><strong>Reason:</strong> {entry.verdict.reasoning}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="alert-allow">
                    <strong>[ALLOWED] {entry.tool_call.tool_name}</strong> at {timestamp_str}
                    <br><strong>Args:</strong> <code>{entry.tool_call.args_summary(150)}</code>
                    <br><strong>Reason:</strong> {entry.verdict.reasoning}
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("No tool calls intercepted yet. Click 'Run Core 3 Demos' in the sidebar to populate the feed.")

st.markdown("---")

# ── Accuracy Validation Dashboard ─────────────────────────────────────────────
report = st.session_state.last_accuracy_report
if report:
    st.markdown("### 🎯 **Authorization Performance Metrics (13-Case Harness)**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Verdict Match Accuracy", f"{report['accuracy']*100:.1f}%")
    with c2:
        st.metric("Classification Precision", f"{report['precision']*100:.1f}%")
    with c3:
        st.metric("Classification Recall", f"{report['recall']*100:.1f}%")
    with c4:
        st.metric("False Positive Rate", f"{report['false_positive_rate']*100:.1f}%")

    # Confusion matrix display
    st.markdown("#### **Confusion Matrix Analysis**")
    matrix_df = pd.DataFrame(report["confusion_matrix"]).T
    st.table(matrix_df)

    # Scenarios run breakdown table
    st.markdown("#### **Test Cases Validation Matrix**")
    breakdown = []
    for scenario in ALL_SCENARIOS:
        # Find matching entry
        matched_entry = None
        for entry in audit_log.entries:
            if entry.tool_call.tool_name == scenario.tool_call.tool_name and \
               entry.tool_call.tool_args.get("path") == scenario.tool_call.tool_args.get("path") and \
               entry.tool_call.tool_args.get("url") == scenario.tool_call.tool_args.get("url"):
                matched_entry = entry
                break

        expected = scenario.expected_action.value
        actual = matched_entry.policy_result.action.value if matched_entry else "NOT RUN"
        match_symbol = "✅ PASS" if expected == actual or (expected == "ALLOW" and actual == "LOG_AND_ALLOW") or (expected == "LOG_AND_ALLOW" and actual == "ALLOW") else "❌ FAIL"
        if actual == "NOT RUN":
            match_symbol = "⚪ PENDING"

        breakdown.append({
            "Scenario ID": scenario.scenario_id,
            "Description": scenario.description,
            "Expected Action": expected,
            "Actual Action": actual,
            "Result": match_symbol,
        })
    st.table(pd.DataFrame(breakdown))

# ── Interception Gateway Architecture Section ──────────────────────────────────
st.markdown("### 🏗️ **System Architecture & Data Flow**")
with st.expander("Show AgentShield Interception Engine Architecture"):
    st.markdown(
        """
        ```mermaid
        graph TD
            UserTask[User Task / Goal] --> AgentOrchestrator[Agent Orchestrator: LangChain/AutoGen/CrewAI]
            AgentOrchestrator --> |Generate ToolCallRequest| InterceptionLayer[AgentShield Interception Layer]
            
            subgraph Gateway Engine
                InterceptionLayer --> |Pause Execution & Query| Classifier[Intent Classification Engine]
                Classifier --> |Tier 1| Deterministic[Regex / Sequence Checker]
                Classifier --> |Tier 2| GptMini[gpt-4o-mini LLM Classifier]
                Classifier --> |Tier 3| GptFull[gpt-4o LLM Classifier]
                
                Deterministic --> Verdict[Risk & Intent Verdict]
                GptMini --> Verdict
                GptFull --> Verdict
                
                Verdict --> PolicyGate[Policy Enforcement Gate]
                PolicyGate --> |Evaluate rules.yaml| Action[Action Decider: BLOCK / ALLOW / LOG / ESCALATE]
            end
            
            Action --> |BLOCK| FailMessage[Return ToolMessage Error to Agent]
            Action --> |ALLOW / LOG| ExecuteTool[Forward request to Tool Registry / Exec]
            Action --> |ESCALATE| ApprovalQueue[Human-in-the-Loop Approval Queue]
            
            ApprovalQueue --> |Approve| ExecuteTool
            ApprovalQueue --> |Deny| FailMessage
            
            ExecuteTool --> |Log Output| AuditDashboard[Streamlit Audit Dashboard Feed]
            FailMessage --> |Log Output| AuditDashboard
        ```
        """
    )
