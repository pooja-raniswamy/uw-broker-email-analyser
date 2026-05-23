"""
aria/agents/history_lookup.py

Agent 3: Looks up client history via MCP tool.

SINGLE RESPONSIBILITY:
Query existing_policies and claims_history for this insured.
Return a ClientHistoryResult with prior policies, claims, red flags.

WHY NO LLM HERE:
This agent calls the MCP tool directly without asking an LLM
to decide when to call it. The logic is deterministic:
- We always call lookup_client_history
- We always pass insured_name and broker_email
- No reasoning needed

Direct tool call = faster + cheaper + more reliable.
Use LLM tool-calling only when reasoning about WHICH tool
to call or WHAT parameters to pass requires intelligence.

RUNS IN PARALLEL with policy_lookup_agent via asyncio.gather()
in the orchestrator. Both start simultaneously.

INTERVIEW TALKING POINT:
"I use two tool-calling patterns: LLM-mediated when parameters
require reasoning from context, direct when parameters are
deterministic. History lookup is always the same call with
the same parameters — no LLM needed."
"""

from datetime import datetime

from aria.agents.state import ARIAState, ClientHistoryResult
from aria.core.llm import get_mcp_tools, get_tool_by_name
from aria.core.delta import write_audit_log
from aria.core.parsers import parse_tool_result


async def history_lookup_agent(state: ARIAState) -> dict:
    """
    Agent 3: Look up client history via MCP tool.

    Input:  state["extracted_risk"] — needs insured_name
    Output: {"client_history": ClientHistoryResult}

    Calls lookup_client_history UC Function via MCP.
    Returns prior policies, claims, red flags, renewal loading.
    """
    if not state.get("extracted_risk"):
        return {"client_history": ClientHistoryResult(
            is_known_client=False,
            red_flags=["No risk data — history lookup skipped"]
        )}

    extracted   = state["extracted_risk"]
    tools       = get_mcp_tools()
    history_tool = get_tool_by_name(tools, "history")

    if not history_tool:
        return {"client_history": ClientHistoryResult(
            is_known_client=False,
            red_flags=["History MCP tool not found"]
        )}

    start_time = datetime.now()

    try:
        # Direct tool call — no LLM reasoning needed
        raw_result = await history_tool.ainvoke({
            "p_insured_name": extracted.insured_name,
            "p_broker_email": state["broker_email"],
        })

        records       = parse_tool_result(raw_result)
        prior_policies = [r for r in records if r.get("record_type") == "policy"]
        prior_claims   = [r for r in records if r.get("record_type") == "claim"]
        total_claims   = sum(r.get("amount", 0.0) for r in prior_claims)

        # ── Red flag detection ────────────────────────────────────
        red_flags = []

        open_claims = [c for c in prior_claims if c.get("status") == "open"]
        if open_claims:
            exposure = sum(c["amount"] for c in open_claims)
            red_flags.append(
                f"{len(open_claims)} open claim(s) not settled — "
                f"exposure: INR {exposure:,.0f}"
            )

        if len(prior_claims) > 2:
            red_flags.append(
                f"High claims frequency: {len(prior_claims)} claims on record"
            )

        declined = [p for p in prior_policies if p.get("status") == "declined"]
        if declined:
            red_flags.append("Previously declined risk — review required")

        if total_claims > 5_000_000:
            red_flags.append(
                f"High total claims: INR {total_claims:,.0f} — "
                f"senior UW review recommended"
            )

        # ── Renewal loading ───────────────────────────────────────
        # Applied on top of base premium in policy_lookup_agent
        renewal_loading = 0.0
        if open_claims:
            renewal_loading = 10.0   # 10% for unsettled claims
        elif prior_claims:
            renewal_loading = 5.0    # 5% for settled claims

        latency = int(
            (datetime.now() - start_time).total_seconds() * 1000
        )

        result = ClientHistoryResult(
            is_known_client=len(prior_policies) > 0,
            prior_policies=prior_policies,
            prior_claims=prior_claims,
            total_claims_amount=total_claims,
            red_flags=red_flags,
            renewal_loading_pct=renewal_loading,
        )

        write_audit_log(
            email_id=state["email_id"],
            agent_name="history_lookup_agent",
            action="lookup_client_history",
            input_summary=f"Insured: {extracted.insured_name}",
            output_summary=(
                f"Known: {result.is_known_client} | "
                f"Policies: {len(prior_policies)} | "
                f"Claims: {len(prior_claims)} | "
                f"Flags: {len(red_flags)} | "
                f"Loading: {renewal_loading}%"
            ),
            latency_ms=latency,
        )

        return {"client_history": result}

    except Exception as e:
        write_audit_log(
            email_id=state["email_id"],
            agent_name="history_lookup_agent",
            action="lookup_client_history_failed",
            input_summary=f"Insured: {extracted.insured_name}",
            output_summary=f"Error: {str(e)[:200]}",
        )
        return {"client_history": ClientHistoryResult(
            is_known_client=False,
            red_flags=[f"History lookup failed: {str(e)[:100]}"],
        )}
