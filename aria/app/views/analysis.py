"""
aria/app/views/analysis.py

View 2: Per-submission analysis

Shows the full agent pipeline output for one submission:
- Left: original broker email (what the broker sent)
- Middle: what each agent extracted and decided
- Right: generated reply email + Approve button

The underwriter reviews all three panels before approving.
Clicking Approve writes approved_by to uw_decisions table
and updates email status to completed.

WHY THREE PANELS:
The underwriter needs to verify the agents got it right.
Left panel = ground truth (original email).
Middle panel = what the AI understood.
Right panel = what the AI will say back.
If middle panel misunderstood something, underwriter can
edit the reply in the right panel before approving.
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession


def get_spark():
    """
    Get Spark session for Databricks Apps.
    Uses DatabricksSession with serverless warehouse.
    Works in both notebooks and Databricks Apps.
    """
    try:
        from databricks.connect import DatabricksSession
        return DatabricksSession.builder.serverless(True).getOrCreate()
    except Exception:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()



def load_submission(email_id: str) -> dict:
    """Load full submission data for one email_id."""
    spark = get_spark()

    # Get email
    email = spark.sql(f"""
        SELECT * FROM uw_broker.underwriting.broker_emails
        WHERE email_id = '{email_id}'
        LIMIT 1
    """).collect()

    # Get decision
    decision = spark.sql(f"""
        SELECT * FROM uw_broker.underwriting.uw_decisions
        WHERE email_id = '{email_id}'
        ORDER BY decided_at DESC
        LIMIT 1
    """).collect()

    # Get audit log
    audit = spark.sql(f"""
        SELECT agent_name, action, input_summary,
               output_summary, latency_ms, logged_at
        FROM uw_broker.underwriting.audit_log
        WHERE email_id = '{email_id}'
        ORDER BY logged_at ASC
    """).toPandas()

    return {
        "email":    email[0] if email else None,
        "decision": decision[0] if decision else None,
        "audit":    audit,
    }


def approve_decision(decision_id: str, email_id: str, approver: str):
    """Write approval to Delta tables."""
    spark = get_spark()
    spark.sql(f"""
        UPDATE uw_broker.underwriting.uw_decisions
        SET approved_by = '{approver}'
        WHERE decision_id = '{decision_id}'
    """)
    spark.sql(f"""
        UPDATE uw_broker.underwriting.broker_emails
        SET status = 'completed'
        WHERE email_id = '{email_id}'
    """)


def appetite_color(appetite: str) -> str:
    return {
        "IN_APPETITE": "🟢",
        "REFER":       "🟡",
        "DECLINE":     "🔴",
    }.get(appetite, "⚪")


def render_analysis(email_id: str, on_back):
    """
    Render the per-submission analysis view.

    Args:
        email_id: ID of the submission to show
        on_back:  callback to return to inbox
    """
    if st.button("← Back to Inbox"):
        on_back()
        return

    data = load_submission(email_id)

    if not data["email"]:
        st.error("Submission not found")
        return

    email    = data["email"]
    decision = data["decision"]
    audit    = data["audit"]

    # ── Header ────────────────────────────────────────────────────
    st.title(f"ARIA Analysis")
    st.caption(f"Email ID: {email_id}")

    if decision:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            appetite = decision["appetite_decision"]
            st.metric("Decision",
                      f"{appetite_color(appetite)} {appetite}")
        with col2:
            pf = decision["premium_from"]
            pt = decision["premium_to"]
            if pf and pf > 0:
                st.metric("Premium Indication",
                          f"₹{pf/100000:.1f}L – ₹{pt/100000:.1f}L")
            else:
                st.metric("Premium", "Manual pricing")
        with col3:
            conf = decision["confidence_score"]
            st.metric("Confidence", f"{conf:.0%}")
        with col4:
            st.metric("Status", email["status"].replace("_", " ").title())

    st.divider()

    # ── Three panel layout ────────────────────────────────────────
    left, middle, right = st.columns([1, 1, 1])

    # Left: original email
    with left:
        st.subheader("📧 Broker Email")
        st.caption(f"From: {email['broker_email']}")
        st.caption(f"Subject: {email['subject']}")
        st.caption(f"Scenario: {email['scenario']}")
        st.divider()
        st.text_area(
            "Original email",
            value=email["raw_email_text"],
            height=400,
            disabled=True,
            label_visibility="collapsed"
        )

    # Middle: agent outputs
    with middle:
        st.subheader("🤖 Agent Analysis")

        if decision and decision["extracted_risk_json"]:
            try:
                risk = json.loads(decision["extracted_risk_json"])

                with st.expander("📋 Extracted Risk", expanded=True):
                    st.write(f"**Insured:** {risk.get('insured_name', 'N/A')}")
                    st.write(f"**Business:** {risk.get('business_description', 'N/A')}")
                    st.write(f"**Coverage:** {', '.join(risk.get('coverage_requested', []))}")
                    si = risk.get('sum_insured_inr')
                    st.write(f"**Sum Insured:** {'₹{:,.0f}'.format(si) if si else 'Not specified'}")
                    st.write(f"**Turnover:** {'{:.1f} Cr'.format(risk.get('turnover_cr')) if risk.get('turnover_cr') else 'Not specified'}")
                    st.write(f"**Territories:** {', '.join(risk.get('territories', []))}")
                    missing = risk.get('missing_fields', [])
                    if missing:
                        st.warning(f"Missing: {', '.join(missing)}")

            except Exception:
                st.info("Risk data not available")

        # Agent audit trail
        with st.expander("🔍 Agent Audit Trail", expanded=False):
            if not audit.empty:
                for _, row in audit.iterrows():
                    agent = row["agent_name"].replace("_agent", "").replace("_", " ").title()
                    latency = row["latency_ms"]
                    st.markdown(f"**{agent}** `{latency}ms`")
                    st.caption(row["output_summary"])
                    st.divider()
            else:
                st.info("No audit data available")

        # Client history
        with st.expander("📁 Client History", expanded=False):
            history = spark.sql(f"""
                SELECT record_type, coverage_type, amount, period, status
                FROM uw_broker.underwriting.lookup_client_history(
                    '{email["insured_name"] if "insured_name" in email else ""}\',
                    '{email["broker_email"]}'
                )
            """) if False else None  # Skip for now — show from decision
            st.info("View broker intelligence tab for full history")

    # Right: reply email + approve
    with right:
        st.subheader("✉️ Generated Reply")

        if decision:
            # Show gaps if any
            if decision["gaps_identified"]:
                try:
                    gaps = json.loads(decision["gaps_identified"])
                    if gaps:
                        with st.expander(f"⚠️ {len(gaps)} Information Gap(s)", expanded=True):
                            for g in gaps:
                                st.markdown(f"• {g}")
                except Exception:
                    pass

            # Editable reply email
            reply_text = st.text_area(
                "Reply email (editable before approval)",
                value=decision["reply_email_text"] or "",
                height=350,
                key=f"reply_{email_id}"
            )

            st.divider()

            # Approval section
            if not decision["approved_by"]:
                approver = st.text_input(
                    "Your name (approver)",
                    placeholder="e.g. Pooja Rani, Senior Underwriter"
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Approve & Complete",
                                 type="primary",
                                 disabled=not approver):
                        approve_decision(
                            decision["decision_id"],
                            email_id,
                            approver
                        )
                        st.success("✓ Decision approved!")
                        st.balloons()
                        st.rerun()

                with col_b:
                    if st.button("🔄 Re-run Pipeline"):
                        st.info("Re-run feature coming soon")
            else:
                st.success(f"✅ Approved by: {decision['approved_by']}")

        else:
            st.info("Pipeline still processing — check back shortly")
            if st.button("🔄 Refresh"):
                st.rerun()
