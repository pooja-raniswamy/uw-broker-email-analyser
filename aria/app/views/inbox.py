"""
aria/app/views/inbox.py

View 1: Submission Inbox

Shows all broker email submissions with:
- Status badge (pending / processing / awaiting_review / completed)
- Appetite decision badge (IN_APPETITE / REFER / DECLINE)
- Confidence score
- Premium indication
- Click to open analysis view

This is the landing page of the ARIA Streamlit app.
The underwriter opens this every morning to see what
came in overnight and what needs attention.
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



def load_inbox_data() -> pd.DataFrame:
    """
    Load all submissions joining broker_emails with latest uw_decision.
    Uses window function to get latest decision per email.
    """
    spark = get_spark()

    # Get latest decision per email using window function
    df = spark.sql("""
        WITH latest_decisions AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY email_id
                    ORDER BY decided_at DESC
                ) AS rn
            FROM uw_broker.underwriting.uw_decisions
        )
        SELECT
            e.email_id,
            e.broker_name,
            e.broker_email,
            e.subject,
            e.scenario,
            e.status,
            e.received_at,
            d.insured_name,
            d.appetite_decision,
            d.premium_from,
            d.premium_to,
            d.confidence_score,
            d.decision_id
        FROM uw_broker.underwriting.broker_emails e
        LEFT JOIN latest_decisions d
            ON e.email_id = d.email_id
            AND d.rn = 1
        ORDER BY e.received_at DESC
    """).toPandas()

    # Deduplicate by email_id — keep first (latest)
    df = df.drop_duplicates(subset=["email_id"], keep="first").reset_index(drop=True)
    return df


def status_badge(status: str) -> str:
    colours = {
        "pending":        "🟡",
        "processing":     "🔵",
        "awaiting_review":"🔴",
        "completed":      "🟢",
        "failed":         "❌",
    }
    return f"{colours.get(status, '⚪')} {status.replace('_', ' ').title()}"


def appetite_badge(appetite: str) -> str:
    if appetite == "IN_APPETITE":
        return "✅ In Appetite"
    elif appetite == "REFER":
        return "⚠️ Refer"
    elif appetite == "DECLINE":
        return "❌ Decline"
    return "⏳ Pending"


def render_inbox(on_select_email):
    """
    Render the submission inbox.

    Args:
        on_select_email: callback function called with email_id
                        when user clicks on a submission
    """
    st.title("ARIA — Submission Inbox")
    st.caption("Automated Risk Intelligence for Underwriting")

    # ── Summary metrics ───────────────────────────────────────────
    df = load_inbox_data()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Submissions", len(df))
    with col2:
        awaiting = len(df[df["status"] == "awaiting_review"])
        st.metric("Awaiting Review", awaiting,
                  delta="needs attention" if awaiting > 0 else None,
                  delta_color="inverse")
    with col3:
        in_appetite = len(df[df["appetite_decision"] == "IN_APPETITE"])
        st.metric("In Appetite", in_appetite)
    with col4:
        completed = len(df[df["status"] == "completed"])
        st.metric("Completed", completed)

    st.divider()

    # ── Filter controls ───────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        status_filter = st.selectbox(
            "Filter by status",
            ["All", "pending", "awaiting_review", "completed", "failed"]
        )
    with col_f2:
        scenario_filter = st.selectbox(
            "Filter by scenario",
            ["All", "new_submission", "renewal", "followup"]
        )

    # Apply filters
    filtered = df.copy()
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]
    if scenario_filter != "All":
        filtered = filtered[filtered["scenario"] == scenario_filter]

    st.caption(f"Showing {len(filtered)} of {len(df)} submissions")
    st.divider()

    # ── Submission cards ──────────────────────────────────────────
    if filtered.empty:
        st.info("No submissions matching the selected filters.")
        return

    for _, row in filtered.iterrows():
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])

            with col1:
                st.markdown(f"**{row.get('insured_name') or row['subject'][:40]}**")
                st.caption(f"📧 {row['broker_email']} · {row['scenario']}")

            with col2:
                st.markdown(status_badge(row["status"]))

            with col3:
                appetite = row.get("appetite_decision")
                if appetite:
                    st.markdown(appetite_badge(appetite))
                else:
                    st.markdown("⏳ Processing")

            with col4:
                premium_from = row.get("premium_from")
                premium_to   = row.get("premium_to")
                if premium_from and premium_from > 0:
                    st.caption(f"₹{premium_from/100000:.1f}L – ₹{premium_to/100000:.1f}L")
                elif appetite:
                    st.caption("Manual pricing needed")
                else:
                    st.caption("—")

            with col5:
                if st.button("View", key=f"view_{row['email_id']}"):
                    on_select_email(row["email_id"])
