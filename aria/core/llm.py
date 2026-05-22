"""
aria/core/llm.py

LLM client and MCP tools loader for ARIA.

WHY EVERYTHING GOES THROUGH UNITY AI GATEWAY:
All LLM calls in ARIA use the Unity AI Gateway endpoint
"uw-broker-claude" rather than calling Anthropic API directly.

Benefits:
1. Centralised auth — API key stored once in Databricks Secrets
2. Audit trail — every LLM call logged to inference table
3. Cost attribution — track spend per user/team
4. Model swap — change claude-sonnet to claude-opus in one place
5. Rate limiting — protect against runaway costs
6. BFSI compliance — all LLM traffic logged and governed

This is the enterprise pattern. In a student project you call
Anthropic directly. In a production BFSI system, everything
goes through the governance layer.

WHY DIRECT TOOL CALLS VS LLM TOOL CALLING:
We use two patterns in ARIA:

Pattern 1 — LLM with tools (risk_extractor_agent):
    Use when: agent needs to REASON about what to extract
    How: llm.with_structured_output(Schema) or llm.bind_tools(tools)
    Cost: higher (LLM call + reasoning)

Pattern 2 — Direct tool call (history_lookup, policy_lookup):
    Use when: parameters are DETERMINISTIC — we always know
              exactly which tool to call and with what inputs
    How: await tool.ainvoke({"param": value})
    Cost: lower (no LLM reasoning, just tool execution)

Knowing when to use which is a senior architect skill.
"""

from databricks_langchain import ChatDatabricks, UCFunctionToolkit
from aria.config import settings


def get_llm() -> ChatDatabricks:
    """
    Returns Claude via Unity AI Gateway.

    All ARIA agents use this function — never instantiate
    ChatDatabricks directly in agent code.

    temperature=0.1: Low randomness for consistent UW decisions.
    Insurance underwriting has right and wrong answers —
    we do not want creative responses.

    max_tokens=4096: Enough for a full UW report + reply email.
    """
    return ChatDatabricks(
        endpoint=settings.gateway_endpoint,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def get_mcp_tools() -> list:
    """
    Load UC Functions as LangChain-compatible MCP tools.

    HOW THIS WORKS:
    1. UCFunctionToolkit connects to the Databricks SQL warehouse
    2. It reads the UC Function definitions from Unity Catalog
    3. It wraps each function as a LangChain BaseTool object
    4. The tool's description = the COMMENT on the UC Function
    5. The LLM reads descriptions to decide when to call each tool

    WHY COMMENTS ON UC FUNCTIONS MATTER:
    The LLM decides which tool to call by reading the description.
    "Look up underwriting appetite rules for a liability risk" →
    LLM calls lookup_policy_rules.
    Bad description = agent never calls the tool or calls wrong tool.

    NOTE ON NAMING:
    Databricks converts dots to double underscores in tool names:
    "uw_broker.underwriting.lookup_policy_rules"
    becomes "uw_broker__underwriting__lookup_policy_rules"
    This is normal — the LLM uses the description not the name.
    """
    toolkit = UCFunctionToolkit(
        warehouse_id=settings.warehouse_id,
        function_names=[
            "uw_broker.underwriting.lookup_policy_rules",
            "uw_broker.underwriting.lookup_client_history",
            "uw_broker.underwriting.log_uw_decision",
        ]
    )
    return toolkit.tools


def get_tool_by_name(tools: list, keyword: str):
    """
    Find a specific tool from the tools list by keyword in name.

    Args:
        tools:   List of tools from get_mcp_tools()
        keyword: Part of tool name to match (e.g. "history", "policy")

    Returns:
        First matching tool or None
    """
    return next((t for t in tools if keyword in t.name), None)
