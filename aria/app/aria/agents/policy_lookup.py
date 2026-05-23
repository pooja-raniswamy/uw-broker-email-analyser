"""
aria/agents/policy_lookup.py

Agent 4: Checks appetite and calculates premium via MCP tool.

SINGLE RESPONSIBILITY:
Query policy_rules table via MCP for each requested coverage type.
Calculate indicative premium. Return UWDecision with appetite
decision, premium range, gaps, and confidence score.

WHY NO LLM HERE:
Same reason as history_lookup_agent — parameters are deterministic.
We always look up rules for each coverage_type in extracted_risk.
Premium calculation is pure arithmetic — no reasoning needed.

PREMIUM CALCULATION:
    base = (sum_insured / 1000) * rate_per_mille
    base = max(base, min_premium)
    base = base * (1 + renewal_loading_pct / 100)
    range = base to base * 1.20 (20% uncertainty buffer)

DECISION LOGIC:
    ANY decline rule  → DECLINE
    ANY refer rule    → REFER
    ALL in_appetite   → IN_APPETITE
    NO rules found    → REFER (cannot price unknown risk)

This mirrors real underwriting: one bad line can affect the
whole submission. Each coverage type is assessed independently.

RUNS IN PARALLEL with history_lookup_agent.

INTERVIEW TALKING POINT:
"Rate per mille pricing with minimum premium floor, renewal
loading from claims history, and uncertainty buffer on the
upper bound — this is how liability premiums are actually
calculated in commercial insurance."
"""

from datetime import datetime

from aria.agents.state import ARIAState, PolicyRuleResult, UWDecision
from aria.config import settings
from aria.core.llm import get_mcp_tools, get_tool_by_name
from aria.core.delta import write_audit_log
from aria.core.parsers import parse_tool_result, safe_float


def _determine_territory(territories: list[str]) -> str:
    """
    Map extracted territories to policy_rules territory values.

    Business logic:
        Any USA/Canada presence → worldwide (most expensive,
            often requires Lloyd's placement)
        Any non-India territory → worldwide_excl_us
        India only → india_only (cheapest)

    This matters because rates are significantly different:
        india_only:      0.75 per mille
        worldwide_excl_us: 1.10-1.35 per mille
        worldwide:       2.50-3.20 per mille (US litigation risk)
    """
    if not territories:
        return "india_only"
    upper = [t.upper() for t in territories]
    if "USA" in upper or "CANADA" in upper:
        return "worldwide"
    if any(t not in ["INDIA"] for t in upper):
        return "worldwide_excl_us"
    return "india_only"


def _normalise_coverage(coverage: str) -> str:
    """
    Normalise coverage type string to match policy_rules table values.

    Brokers write coverage types inconsistently:
        "Product Liability" / "product liability" / "PL"
    Our table has: "product_liability"

    Simple normalisation: lowercase + replace spaces with underscores.
    """
    return coverage.lower().replace(" ", "_").replace("-", "_")


def _calculate_premium(
    rule: PolicyRuleResult,
    sum_insured: float,
    renewal_loading_pct: float,
) -> tuple[float, float]:
    """
    Calculate premium range for one coverage line.

    Returns:
        (premium_from, premium_to) tuple

    premium_from: base calculation with renewal loading
    premium_to:   base + 20% uncertainty buffer
    """
    if rule.rate_per_mille <= 0:
        return 0.0, 0.0

    # Base: rate per mille applied to sum insured
    base = (sum_insured / 1000) * rule.rate_per_mille

    # Apply minimum premium floor
    base = max(base, rule.min_premium)

    # Apply renewal loading from claims history
    base = base * (1 + renewal_loading_pct / 100)

    # Round to nearest 1000 for cleaner quotes
    from_val = round(base / 1000) * 1000
    to_val   = round(base * 1.20 / 1000) * 1000

    return from_val, to_val


async def policy_lookup_agent(state: ARIAState) -> dict:
    """
    Agent 4: Check appetite and calculate premium.

    Input:  state["extracted_risk"] + state["client_history"]
    Output: {"policy_rules": list[PolicyRuleResult],
             "uw_decision": UWDecision}
    """
    if not state.get("extracted_risk"):
        return {
            "policy_rules": [],
            "uw_decision": UWDecision(
                appetite_decision="REFER",
                gaps_identified=["No risk data — cannot assess appetite"],
                confidence_score=0.3,
            )
        }

    extracted = state["extracted_risk"]
    tools     = get_mcp_tools()
    policy_tool = get_tool_by_name(tools, "policy_rules")

    if not policy_tool:
        return {
            "policy_rules": [],
            "uw_decision": UWDecision(
                appetite_decision="REFER",
                gaps_identified=["Policy rules tool not found"],
                confidence_score=0.3,
            )
        }

    start_time   = datetime.now()
    all_rules    = []
    gaps         = list(extracted.missing_fields)
    territory    = _determine_territory(extracted.territories)
    sum_insured  = extracted.sum_insured_inr or 10_000_000.0
    renewal_load = 0.0

    if state.get("client_history"):
        renewal_load = state["client_history"].renewal_loading_pct

    # ── Look up rules for each requested coverage type ────────────
    for coverage in extracted.coverage_requested:
        coverage_norm = _normalise_coverage(coverage)
        try:
            raw = await policy_tool.ainvoke({
                "p_coverage_type": coverage_norm,
                "p_territory":     territory,
                "p_sum_insured":   sum_insured,
            })
            records = parse_tool_result(raw)

            if records:
                for r in records:
                    all_rules.append(PolicyRuleResult(
                        rule_id=        r.get("rule_id", ""),
                        appetite_status=r.get("appetite_status", "refer"),
                        min_premium=    safe_float(r.get("min_premium")),
                        max_coverage=   safe_float(r.get("max_coverage")),
                        rate_per_mille= safe_float(r.get("rate_per_mille")),
                        exclusions=     r.get("exclusions", ""),
                    ))
            else:
                gaps.append(
                    f"No rules found for {coverage} in {territory}"
                )

        except Exception as e:
            gaps.append(f"Rule lookup failed for {coverage}: {str(e)[:80]}")

    # ── Appetite decision ─────────────────────────────────────────
    # Filter out rules with no rate (these are gap markers, not real rules)
    valid_rules = [r for r in all_rules if r.rate_per_mille > 0]

    if not valid_rules and not all_rules:
        # No rules found at all — cannot assess, refer
        appetite = "REFER"
        gaps.append("No matching rules found — manual review required")
    elif any(r.appetite_status == "decline" for r in valid_rules):
        appetite = "DECLINE"
    elif any(r.appetite_status == "refer" for r in valid_rules):
        appetite = "REFER"
    elif not valid_rules:
        # Only found zero-rate rules (like cyber decline)
        appetite = "REFER"
        gaps.append("Coverage not clearly in appetite — refer for review")
    else:
        appetite = "IN_APPETITE"

    # ── Premium calculation ───────────────────────────────────────
    total_from = 0.0
    total_to   = 0.0

    # Calculate premium for IN_APPETITE and REFER
    # Even referred risks need indicative premium for the UW report
    # Only skip for DECLINE
    if appetite != "DECLINE":
        for rule in all_rules:
            if rule.rate_per_mille > 0:  # skip zero-rate rules
                pf, pt = _calculate_premium(rule, sum_insured, renewal_load)
                total_from += pf
                total_to   += pt

    # ── Confidence score ──────────────────────────────────────────
    confidence = 0.90
    confidence -= 0.05 * len(gaps)
    if appetite == "REFER":
        confidence -= 0.10
    if not extracted.sum_insured_inr:
        confidence -= 0.15   # big uncertainty without sum insured
    if state.get("client_history") and state["client_history"].red_flags:
        confidence -= 0.05 * len(state["client_history"].red_flags)
    confidence = max(0.30, round(confidence, 2))

    # ── HITL trigger ──────────────────────────────────────────────
    # Read thresholds at runtime
    from aria.config import settings as current_settings
    requires_senior = (
        sum_insured > current_settings.hitl_sum_insured_threshold or
        (appetite == "REFER" and confidence < current_settings.hitl_confidence_threshold) or
        (appetite == "DECLINE") or
        len(gaps) > 3
    )

    # ── Reply type ────────────────────────────────────────────────
    if appetite == "DECLINE":
        reply_type = "DECLINE"
    elif gaps:
        reply_type = "INFO_REQUEST"
    else:
        reply_type = "INDICATIVE"

    latency = int(
        (datetime.now() - start_time).total_seconds() * 1000
    )

    decision = UWDecision(
        appetite_decision=    appetite,
        premium_from=         total_from,
        premium_to=           total_to,
        gaps_identified=      gaps,
        reply_type=           reply_type,
        confidence_score=     confidence,
        requires_senior_review=requires_senior,
    )

    write_audit_log(
        email_id=state["email_id"],
        agent_name="policy_lookup_agent",
        action="lookup_policy_rules",
        input_summary=(
            f"Coverage: {extracted.coverage_requested} | "
            f"Territory: {territory} | "
            f"SI: INR {sum_insured:,.0f}"
        ),
        output_summary=(
            f"Decision: {appetite} | "
            f"Premium: INR {total_from:,.0f}–{total_to:,.0f} | "
            f"Confidence: {confidence} | "
            f"Gaps: {len(gaps)}"
        ),
        latency_ms=latency,
    )

    return {
        "policy_rules": all_rules,
        "uw_decision":  decision,
    }
