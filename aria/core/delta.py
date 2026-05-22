"""
aria/core/delta.py

Delta table write utilities for ARIA agents.

WHY THIS FILE EXISTS:
All agents need to write to Delta tables — audit_log and
uw_decisions. Centralising these writes means:
1. One place to fix if table schema changes
2. Consistent error handling across all agents
3. Type safety — we define explicit schemas so Delta
   never gets confused about INT vs BIGINT

IMPORTANT LESSON (from production debugging):
Never use spark.sql() with Python string formatting to write data.
Example of what NOT to do:
    spark.sql(f"INSERT INTO table VALUES ('{python_var}')")

This fails when:
- python_var contains quotes or special characters
- You try to embed Python expressions in SQL strings
- Spark SQL parser sees Python syntax as SQL syntax

The correct pattern is always:
    spark.createDataFrame([row_dict]).write.mode("append").saveAsTable(...)

This passes Python values directly to Spark — no SQL parsing.
"""

import uuid
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, TimestampType, BooleanType
)


def get_spark() -> SparkSession:
    """Get or create SparkSession."""
    return SparkSession.builder.getOrCreate()


# ── Explicit schemas ──────────────────────────────────────────────
# Why explicit schemas: Spark infers Python int as BIGINT but our
# tables have BIGINT columns so this matches correctly.
# Without explicit schema, type mismatches cause silent failures.

AUDIT_LOG_SCHEMA = StructType([
    StructField("log_id",         StringType(),    False),
    StructField("email_id",       StringType(),    True),
    StructField("agent_name",     StringType(),    True),
    StructField("action",         StringType(),    True),
    StructField("input_summary",  StringType(),    True),
    StructField("output_summary", StringType(),    True),
    StructField("tokens_used",    StringType(),    True),
    StructField("latency_ms",     StringType(),    True),
    StructField("logged_at",      TimestampType(), True),
])

UW_DECISIONS_SCHEMA = StructType([
    StructField("decision_id",          StringType(),    False),
    StructField("email_id",             StringType(),    False),
    StructField("insured_name",         StringType(),    True),
    StructField("scenario",             StringType(),    True),
    StructField("extracted_risk_json",  StringType(),    True),
    StructField("appetite_decision",    StringType(),    True),
    StructField("premium_from",         DoubleType(),    True),
    StructField("premium_to",           DoubleType(),    True),
    StructField("gaps_identified",      StringType(),    True),
    StructField("reply_email_text",     StringType(),    True),
    StructField("confidence_score",     DoubleType(),    True),
    StructField("approved_by",          StringType(),    True),
    StructField("decided_at",           TimestampType(), True),
])


def write_audit_log(
    email_id:      str,
    agent_name:    str,
    action:        str,
    input_summary: str,
    output_summary: str,
    tokens_used:   int = 0,
    latency_ms:    int = 0,
    catalog:       str = "uw_broker",
    schema:        str = "underwriting",
) -> None:
    """
    Write one row to audit_log table.

    Called by every agent after completing its work.
    This is the regulatory compliance trail — every AI action
    is logged with who did what, when, and how long it took.

    Args:
        email_id:       The broker email being processed
        agent_name:     Which agent is logging (e.g. "risk_extractor_agent")
        action:         What it did (e.g. "extract_risk_from_email")
        input_summary:  Brief description of input (truncated to 500 chars)
        output_summary: Brief description of output (truncated to 500 chars)
        tokens_used:    LLM tokens consumed (0 if no LLM call)
        latency_ms:     Time taken in milliseconds
    """
    spark = get_spark()

    row = [(
        f"log_{uuid.uuid4().hex[:12]}",
        email_id,
        agent_name,
        action,
        str(input_summary)[:500],
        str(output_summary)[:500],
        str(tokens_used),
        str(latency_ms),
        datetime.now(),
    )]

    spark.createDataFrame(row, AUDIT_LOG_SCHEMA)          .write.mode("append")          .saveAsTable(f"{catalog}.{schema}.audit_log")


def write_uw_decision(
    email_id:            str,
    insured_name:        str,
    scenario:            str,
    extracted_risk_json: str,
    appetite_decision:   str,
    premium_from:        float,
    premium_to:          float,
    gaps_identified:     str,
    reply_email_text:    str,
    confidence_score:    float,
    catalog:             str = "uw_broker",
    schema:              str = "underwriting",
) -> str:
    """
    Write completed UW decision to uw_decisions table.

    Returns the decision_id for tracking.
    Called by reply_composer_agent as the final step
    before the decision appears in the Streamlit UI.
    """
    spark = get_spark()
    decision_id = f"dec_{uuid.uuid4().hex[:12]}"

    row = [(
        decision_id,
        email_id,
        insured_name,
        scenario,
        extracted_risk_json,
        appetite_decision,
        float(premium_from),
        float(premium_to),
        gaps_identified,
        reply_email_text,
        float(confidence_score),
        None,           # approved_by — set when UW approves in UI
        datetime.now(),
    )]

    spark.createDataFrame(row, UW_DECISIONS_SCHEMA)          .write.mode("append")          .saveAsTable(f"{catalog}.{schema}.uw_decisions")

    return decision_id


def update_email_status(
    email_id: str,
    status:   str,
    catalog:  str = "uw_broker",
    schema:   str = "underwriting",
) -> None:
    """
    Update broker_emails status as pipeline progresses.

    Status flow:
        pending → processing → awaiting_review → completed
                                               → failed

    Called by orchestrator at each stage transition.
    The Streamlit inbox reads this status for the badge colour.
    """
    spark = get_spark()
    spark.sql(f"""
        UPDATE {catalog}.{schema}.broker_emails
        SET status = '{status}'
        WHERE email_id = '{email_id}'
    """)
