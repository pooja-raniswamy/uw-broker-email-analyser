"""
aria/app/views/submit.py

View 4: Submit New Email

Allows underwriters to paste a broker email or upload a PDF
and run the ARIA pipeline in real time.
"""

import sys
import os

# Add repo root to path at module load time
REPO_ROOT = "/Workspace/Users/swamy.poojarani@gmail.com/uw-broker-email-analyser"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import streamlit as st
import asyncio
import uuid
import nest_asyncio
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType
)

nest_asyncio.apply()


def get_spark():
    try:
        from databricks.connect import DatabricksSession
        return DatabricksSession.builder.serverless(True).getOrCreate()
    except Exception:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from uploaded PDF using pypdf."""
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        return f"PDF extraction failed: {str(e)}"


def save_email_to_delta(
    broker_name: str,
    broker_email: str,
    subject: str,
    email_text: str,
) -> str:
    """Save submitted email to broker_emails Delta table."""
    spark = get_spark()
    email_id = f"em_{uuid.uuid4().hex[:12]}"
    now = datetime.now()

    schema = StructType([
        StructField("email_id",       StringType(),    False),
        StructField("broker_name",    StringType(),    True),
        StructField("broker_email",   StringType(),    True),
        StructField("subject",        StringType(),    True),
        StructField("raw_email_text", StringType(),    True),
        StructField("pdf_path",       StringType(),    True),
        StructField("scenario",       StringType(),    True),
        StructField("status",         StringType(),    True),
        StructField("received_at",    TimestampType(), True),
        StructField("created_at",     TimestampType(), True),
    ])

    row = [(
        email_id, broker_name, broker_email,
        subject, email_text,
        None, "pending", "pending",
        now, now,
    )]

    spark.createDataFrame(row, schema)\
         .write.mode("append")\
         .saveAsTable("uw_broker.underwriting.broker_emails")

    return email_id


def render_submit(on_view_result):
    """Render the submit new email view."""
    st.title("📨 Submit New Email")
    st.caption("Paste a broker email or upload a PDF to run through ARIA")
    st.divider()

    # ── Broker details ────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        broker_name = st.text_input(
            "Broker name",
            placeholder="e.g. Rajesh Sharma"
        )
    with col2:
        broker_email = st.text_input(
            "Broker email",
            placeholder="e.g. rajesh@abcbrokers.com"
        )

    subject = st.text_input(
        "Email subject",
        placeholder="e.g. New liability submission - TechNova Solutions"
    )

    st.divider()

    # ── Input method ──────────────────────────────────────────────
    input_method = st.radio(
        "Submission format",
        ["📝 Paste email text", "📄 Upload PDF"],
        horizontal=True
    )

    email_text = ""

    if input_method == "📄 Upload PDF":
        uploaded_pdf = st.file_uploader(
            "Upload broker submission PDF",
            type=["pdf"],
            help="Upload a PDF containing the broker submission"
        )
        if uploaded_pdf:
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.markdown(f"📄 **{uploaded_pdf.name}**")
                st.caption(f"{uploaded_pdf.size / 1024:.1f} KB")
            with col_b:
                with st.spinner("Extracting text from PDF..."):
                    extracted = extract_text_from_pdf(uploaded_pdf)
                if extracted.startswith("PDF extraction failed"):
                    st.error(extracted)
                else:
                    st.success(f"✓ {len(extracted)} characters extracted")
                    email_text = extracted

            if email_text:
                with st.expander("👁 Preview extracted text"):
                    st.text(email_text[:1000] + (
                        "..." if len(email_text) > 1000 else ""
                    ))
    else:
        email_text = st.text_area(
            "Email body",
            placeholder="""Paste the full broker email here...

Example:
Dear Underwriting Team,

We are seeking liability cover for our client ABC Manufacturing Ltd.
- Business: Industrial equipment manufacturer
- Employees: 150
- Turnover: INR 35 Crores
- Coverage required: Product Liability - INR 2 Crores
- Territory: India and UAE

Please indicate appetite and premium range.

Regards,
Rajesh""",
            height=280
        )

    st.divider()

    # ── Submit button ─────────────────────────────────────────────
    can_submit = all([broker_name, broker_email, subject, email_text])

    if not can_submit:
        st.info("Fill in all fields above to submit")

    if st.button("🚀 Submit to ARIA", type="primary", disabled=not can_submit):

        with st.spinner("Saving submission..."):
            email_id = save_email_to_delta(
                broker_name=broker_name,
                broker_email=broker_email,
                subject=subject,
                email_text=email_text,
            )
        st.success(f"✓ Submission saved: {email_id}")
        st.divider()

        st.subheader("🤖 ARIA Processing...")

        status1 = st.status("Agent 2: Extracting risk data...", expanded=True)
        status2 = st.status("Agent 3+4: History + Policy lookup...", expanded=False)
        status3 = st.status("Agent 5: Composing reply...", expanded=False)

        import sys
        sys.path.insert(
            0,
            "/Workspace/Users/swamy.poojarani@gmail.com/uw-broker-email-analyser"
        )
        from aria.agents.orchestrator import run_pipeline

        try:
            result = asyncio.get_event_loop().run_until_complete(
                run_pipeline(
                    email_id=email_id,
                    broker_email=broker_email,
                    raw_email_text=email_text,
                    checkpointer=None,
                )
            )

            ext   = result.get("extracted_risk")
            hist  = result.get("client_history")
            dec   = result.get("uw_decision")
            reply = result.get("reply_email_text")

            if ext:
                with status1:
                    st.write(f"✓ Insured: **{ext.insured_name}**")
                    st.write(f"✓ Scenario: **{ext.scenario}**")
                    st.write(f"✓ Coverage: {', '.join(ext.coverage_requested)}")
                    st.write(f"✓ Territories: {', '.join(ext.territories)}")
                    if ext.missing_fields:
                        st.warning(f"Missing: {', '.join(ext.missing_fields)}")
                status1.update(label="✅ Risk extracted", state="complete")

            if hist and dec:
                with status2:
                    st.write(f"✓ Known client: **{hist.is_known_client}**")
                    st.write(f"✓ Prior policies: {len(hist.prior_policies)}")
                    st.write(f"✓ Appetite: **{dec.appetite_decision}**")
                    if dec.premium_from > 0:
                        st.write(
                            f"✓ Premium: ₹{dec.premium_from/100000:.1f}L"
                            f" – ₹{dec.premium_to/100000:.1f}L"
                        )
                status2.update(label="✅ Lookups complete", state="complete")

            if reply:
                with status3:
                    if dec:
                        st.write(f"✓ Reply type: **{dec.reply_type}**")
                    st.write(f"✓ {len(reply)} chars generated")
                status3.update(label="✅ Reply composed", state="complete")

            st.divider()
            st.subheader("✅ Pipeline Complete")

            col1, col2, col3 = st.columns(3)
            with col1:
                appetite = dec.appetite_decision if dec else result.get("uw_decision_appetite", "Processed")
                icon = {"IN_APPETITE": "🟢", "REFER": "🟡",
                        "DECLINE": "🔴"}.get(appetite, "🟢")
                st.metric("Decision", f"{icon} {appetite}")
            with col2:
                if dec and dec.premium_from and dec.premium_from > 0:
                    st.metric(
                        "Premium Indication",
                        f"₹{dec.premium_from/100000:.1f}L"
                        f"–₹{dec.premium_to/100000:.1f}L"
                    )
                else:
                    st.metric("Premium", "See full analysis")
            with col3:
                conf = dec.confidence_score if dec else 0.9
                st.metric("Confidence", f"{conf:.0%}")

            if st.button("📄 View Full Analysis", type="primary"):
                on_view_result(email_id)

        except Exception as e:
            status1.update(label="❌ Error", state="error")
            st.error(f"Pipeline error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
