"""
ARIA — Automated Risk Intelligence for Underwriting

A multi-agent AI system that transforms unstructured broker
liability submissions into structured underwriting decisions
and professional reply emails.

Architecture:
- 5 LangGraph agents running on Databricks
- 3 MCP tools as Unity Catalog Functions  
- Lakebase memory for cross-session context
- Unity AI Gateway for all LLM calls
- Streamlit UI deployed as Databricks App

Stack:
    LangGraph + LangChain + Claude claude-sonnet-4-5
    Databricks (Unity Catalog, Delta Lake, Lakebase)
    Azure (AWS S3 for storage, us-west-2)
"""

__version__ = "0.1.0"
__author__ = "Pooja Rani"
