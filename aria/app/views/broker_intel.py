"""
aria/app/views/broker_intel.py

View 3: Broker Intelligence

Per-broker analytics showing:
- Submission history and hit rate
- Client portfolio
- Claims record across all clients
- Premium volume over time

Used by relationship managers to assess broker quality.
Reads directly from Delta tables — no agent calls needed.
"""

import streamlit as st
import pandas as pd
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



def load_broker_list() -> list[str]:
    """Get all unique broker emails."""
    spark = get_spark()
    rows = spark.sql("""
        SELECT DISTINCT broker_email
        FROM uw_broker.underwriting.broker_emails
        ORDER BY broker_email
    """).collect()
    return [r["broker_email"] for r in rows]


def load_broker_data(broker_email: str) -> dict:
    """Load all data for one broker."""
    spark = get_spark()

    # All submissions
    submissions = spark.sql(f"""
        SELECT
            e.email_id,
            e.subject,
            e.scenario,
            e.status,
            e.received_at,
            d.insured_name,
            d.appetite_decision,
            d.premium_from,
            d.premium_to,
            d.confidence_score
        FROM uw_broker.underwriting.broker_emails e
        LEFT JOIN uw_broker.underwriting.uw_decisions d
            ON e.email_id = d.email_id
        WHERE e.broker_email = '{broker_email}'
        ORDER BY e.received_at DESC
    """).toPandas()

    # Existing policies
    policies = spark.sql(f"""
        SELECT insured_name, coverage_type,
               premium_paid, status, policy_period
        FROM uw_broker.underwriting.existing_policies
        WHERE broker_email = '{broker_email}'
        ORDER BY insured_name
    """).toPandas()

    # Claims history
    claims = spark.sql(f"""
        SELECT insured_name, claim_type, claim_amount,
               claim_year, status, description
        FROM uw_broker.underwriting.claims_history
        WHERE broker_email = '{broker_email}'
        ORDER BY claim_year DESC
    """).toPandas()

    return {
        "submissions": submissions,
        "policies":    policies,
        "claims":      claims,
    }


def render_broker_intel():
    """Render the broker intelligence view."""
    st.title("🏢 Broker Intelligence")
    st.caption("Per-broker submission history and client portfolio")

    # ── Broker selector ───────────────────────────────────────────
    brokers = load_broker_list()
    if not brokers:
        st.info("No broker data available yet.")
        return

    selected = st.selectbox(
        "Select broker",
        brokers,
        format_func=lambda x: x
    )

    if not selected:
        return

    st.divider()

    data        = load_broker_data(selected)
    submissions = data["submissions"]
    policies    = data["policies"]
    claims      = data["claims"]

    # ── Summary metrics ───────────────────────────────────────────
    total    = len(submissions)
    accepted = len(submissions[
        submissions["appetite_decision"] == "IN_APPETITE"
    ]) if total > 0 else 0
    referred = len(submissions[
        submissions["appetite_decision"] == "REFER"
    ]) if total > 0 else 0
    total_premium = policies["premium_paid"].sum() if not policies.empty else 0
    total_claims  = claims["claim_amount"].sum() if not claims.empty else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Submissions", total)
    with col2:
        rate = f"{accepted/total*100:.0f}%" if total > 0 else "0%"
        st.metric("Acceptance Rate", rate)
    with col3:
        st.metric("Referred", referred)
    with col4:
        st.metric("Premium Volume",
                  f"₹{total_premium/100000:.1f}L" if total_premium > 0 else "₹0")
    with col5:
        st.metric("Total Claims",
                  f"₹{total_claims/100000:.1f}L" if total_claims > 0 else "₹0")

    st.divider()

    # ── Submission history ────────────────────────────────────────
    st.subheader("📋 Submission History")
    if not submissions.empty:
        display = submissions[[
            "insured_name", "scenario", "appetite_decision",
            "premium_from", "premium_to", "status"
        ]].copy()

        display["premium_indication"] = display.apply(
            lambda r: f"₹{r['premium_from']/100000:.1f}L–₹{r['premium_to']/100000:.1f}L"
            if r["premium_from"] and r["premium_from"] > 0
            else "Manual pricing",
            axis=1
        )
        display = display.drop(
            columns=["premium_from", "premium_to"]
        )
        display.columns = [
            "Insured", "Scenario", "Decision",
            "Status", "Premium"
        ]
        st.dataframe(display, use_container_width=True)
    else:
        st.info("No submissions found for this broker.")

    # ── Two columns: policies + claims ────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📄 Policy Portfolio")
        if not policies.empty:
            st.dataframe(
                policies[[
                    "insured_name", "coverage_type",
                    "premium_paid", "status"
                ]].rename(columns={
                    "insured_name":  "Insured",
                    "coverage_type": "Coverage",
                    "premium_paid":  "Premium",
                    "status":        "Status"
                }),
                use_container_width=True
            )
        else:
            st.info("No prior policies on record.")

    with col_right:
        st.subheader("⚠️ Claims History")
        if not claims.empty:
            st.dataframe(
                claims[[
                    "insured_name", "claim_type",
                    "claim_amount", "claim_year", "status"
                ]].rename(columns={
                    "insured_name": "Insured",
                    "claim_type":   "Type",
                    "claim_amount": "Amount (₹)",
                    "claim_year":   "Year",
                    "status":       "Status"
                }),
                use_container_width=True
            )
            # Highlight open claims
            open_claims = claims[claims["status"] == "open"]
            if not open_claims.empty:
                st.warning(
                    f"⚠️ {len(open_claims)} open claim(s) — "
                    f"total exposure: ₹{open_claims['claim_amount'].sum()/100000:.1f}L"
                )
        else:
            st.info("No claims on record — clean history.")
