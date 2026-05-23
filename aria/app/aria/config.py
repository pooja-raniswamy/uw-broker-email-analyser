"""
aria/config.py

Central configuration for ARIA.

WHY ONE CONFIG FILE:
All settings in one place means:
- Change warehouse ID once, all agents pick it up
- No hardcoded values scattered across files
- Easy to swap between dev/staging/prod environments

Pydantic Settings reads from environment variables automatically.
In Databricks, we set these via Databricks Secrets or cluster env vars.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    All ARIA configuration in one place.
    
    Pydantic Settings automatically reads from:
    1. Environment variables (uppercase)
    2. .env file (if present locally)
    3. Default values defined here
    
    In Databricks: set via cluster environment variables
    or Databricks Secrets mounted as env vars.
    """

    # ── Databricks ────────────────────────────────────────────────
    databricks_host: str = Field(
        default="https://dbc-67c325a6-f0af.cloud.databricks.com",
        description="Databricks workspace URL"
    )

    warehouse_id: str = Field(
        default="739424d97a740adf",
        description="Databricks SQL warehouse ID for UC Function execution"
    )

    # ── Unity Catalog ─────────────────────────────────────────────
    catalog_name: str = Field(
        default="uw_broker",
        description="Unity Catalog catalog name"
    )

    schema_name: str = Field(
        default="underwriting",
        description="Unity Catalog schema name"
    )

    # ── AI Gateway ────────────────────────────────────────────────
    gateway_endpoint: str = Field(
        default="uw-broker-claude",
        description="Unity AI Gateway endpoint name for Claude"
    )

    llm_temperature: float = Field(
        default=0.1,
        description="LLM temperature — low for consistent UW decisions"
    )

    llm_max_tokens: int = Field(
        default=4096,
        description="Max tokens — enough for full UW report + reply email"
    )

    # ── Lakebase ──────────────────────────────────────────────────
    lakebase_instance_id: str = Field(
        default="",
        description="Lakebase instance ID for LangGraph checkpointer"
    )

    # ── Business rules ────────────────────────────────────────────
    hitl_confidence_threshold: float = Field(
        default=0.75,
        description="Confidence below this triggers human review"
    )

    hitl_sum_insured_threshold: float = Field(
        default=50000000.0,
        description="Sum insured above this (5 Crore) triggers senior review"
    )

    max_extraction_retries: int = Field(
        default=3,
        description="Max retries for structured output extraction"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# ── Singleton instance ────────────────────────────────────────────
# Import this everywhere: from aria.config import settings
settings = Settings()
