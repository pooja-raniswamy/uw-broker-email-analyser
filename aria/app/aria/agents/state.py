"""
aria/agents/state.py

Shared state schema for the ARIA LangGraph pipeline.

This is the most important file in the project.
Every agent reads from and writes to this state.
Get this right and everything else composes cleanly.

DESIGN DECISIONS:
1. TypedDict not dataclass — LangGraph requires TypedDict for
   partial updates and checkpointing to work correctly

2. Optional fields for agent outputs — None means agent has
   not run yet. LangGraph uses this for resuming from checkpoints.

3. Pydantic models for structured data — validation catches
   errors at the source rather than propagating wrong values

4. Annotated[list, add_messages] for messages — append strategy
   builds conversation history rather than replacing it

5. needs_senior_review bool — the HITL trigger. When True,
   LangGraph interrupt() pauses the graph for human review.
"""

from typing import TypedDict, Optional, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


# ── Sub-models ────────────────────────────────────────────────────

class ClaimItem(BaseModel):
    """
    A single claim mentioned in the broker email.

    risk_extractor_agent extracts these from free text.
    history_lookup_agent compares them against claims_history table.
    Discrepancies between broker-stated and our-record claims
    are flagged as red flags.
    """
    year:        Optional[int]   = None
    amount_inr:  Optional[float] = None
    claim_type:  Optional[str]   = None
    description: Optional[str]   = None
    status:      Optional[str]   = None  # settled / open / repudiated


class ExtractedRisk(BaseModel):
    """
    Structured risk data extracted from broker email.

    Output of risk_extractor_agent.
    Input to history_lookup_agent and policy_lookup_agent.

    All fields Optional because brokers rarely provide
    complete information in a single email.
    missing_fields lists what we still need to quote.
    """
    insured_name:        Optional[str]        = None
    business_description: Optional[str]       = None
    scenario:            str                  = "new_submission"
    coverage_requested:  list[str]            = Field(default_factory=list)
    sum_insured_inr:     Optional[float]      = None
    turnover_cr:         Optional[float]      = None
    employee_count:      Optional[int]        = None
    territories:         list[str]            = Field(default_factory=list)
    claims_mentioned:    list[ClaimItem]      = Field(default_factory=list)
    renewal_date:        Optional[str]        = None
    missing_fields:      list[str]            = Field(default_factory=list)


class PolicyRuleResult(BaseModel):
    """
    A single matching rule from the policy_rules table.

    Output of lookup_policy_rules MCP tool.
    Used by policy_lookup_agent to calculate premium
    and determine appetite decision.
    """
    rule_id:        Optional[str] = None
    appetite_status: str          = "unknown"
    min_premium:    float         = 0.0
    max_coverage:   float         = 0.0
    rate_per_mille: float         = 0.0
    exclusions:     Optional[str] = None


class ClientHistoryResult(BaseModel):
    """
    Complete client history from existing_policies + claims_history.

    Output of history_lookup_agent.
    Used by policy_lookup_agent for renewal loading calculation
    and by orchestrator for red flag detection.

    is_known_client: False means new client — no prior relationship.
    renewal_loading_pct: Applied on top of base premium.
        0% = no claims
        5% = settled claims on record
        10% = open/unsettled claims
    """
    is_known_client:      bool        = False
    prior_policies:       list[dict]  = Field(default_factory=list)
    prior_claims:         list[dict]  = Field(default_factory=list)
    total_claims_amount:  float       = 0.0
    red_flags:            list[str]   = Field(default_factory=list)
    renewal_loading_pct:  float       = 0.0


class UWDecision(BaseModel):
    """
    Final underwriting decision produced by policy_lookup_agent
    and refined by reply_composer_agent.

    appetite_decision values:
        IN_APPETITE: risk accepted, can quote
        REFER:       risk needs senior UW review before quoting
        DECLINE:     risk not in appetite, cannot insure

    reply_type determines which email template is used:
        INFO_REQUEST: ask broker for missing information
        INDICATIVE:   provide indicative premium terms
        DECLINE:      professional decline with reasons

    confidence_score: 0.0 to 1.0
        How confident the system is in this decision.
        Below hitl_confidence_threshold → triggers senior review.
    """
    appetite_decision:    str        = "REFER"
    premium_from:         float      = 0.0
    premium_to:           float      = 0.0
    gaps_identified:      list[str]  = Field(default_factory=list)
    reply_type:           str        = "INFO_REQUEST"
    confidence_score:     float      = 0.0
    requires_senior_review: bool     = False


# ── Main state ────────────────────────────────────────────────────

class ARIAState(TypedDict):
    """
    Shared state for the ARIA LangGraph pipeline.

    Flows through all 5 agents in sequence and parallel.
    Checkpointed to Lakebase after each agent completes.

    Field lifecycle:
        email_id, broker_email, raw_email_text:
            Set at pipeline start. Never modified.

        extracted_risk:
            None → set by risk_extractor_agent

        client_history:
            None → set by history_lookup_agent (parallel)

        policy_rules, uw_decision:
            [] / None → set by policy_lookup_agent (parallel)

        reply_email_text:
            None → set by reply_composer_agent

        needs_senior_review:
            False → may be set True by orchestrator
            → triggers interrupt() for HITL
            → reset to False when UW approves

        messages:
            [] → grows throughout pipeline via add_messages
            → persisted in Lakebase for cross-session memory
    """

    # ── Input (set once at start) ─────────────────────────────────
    email_id:         str
    broker_email:     str
    raw_email_text:   str

    # ── Agent outputs (None until each agent runs) ────────────────
    extracted_risk:   Optional[ExtractedRisk]
    policy_rules:     list[PolicyRuleResult]
    client_history:   Optional[ClientHistoryResult]
    uw_decision:      Optional[UWDecision]
    reply_email_text: Optional[str]

    # ── Control flow ──────────────────────────────────────────────
    needs_senior_review: bool
    error_message:       Optional[str]

    # ── Conversation history ──────────────────────────────────────
    # add_messages: append strategy — new messages are added
    # to the list rather than replacing it.
    # This builds full conversation history across agent calls.
    # Lakebase checkpoints this list for cross-session memory.
    messages: Annotated[list, add_messages]
