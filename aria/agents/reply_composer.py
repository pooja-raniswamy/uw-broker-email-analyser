"""
aria/agents/reply_composer.py

Agent 5: Generates UW report and professional broker reply email.

SINGLE RESPONSIBILITY:
Take all agent outputs, generate a professional reply email
in correct insurance underwriting language, write the decision
to uw_decisions table, return the reply text.

WHY LLM HERE (unlike agents 3 and 4):
Writing a professional email requires language generation
and tone judgement — not deterministic logic.
The LLM earns its place here.

THREE REPLY TYPES:
    INFO_REQUEST:  Acknowledge + confirm appetite + list what we need
    INDICATIVE:    Indicative premium terms + conditions + next steps
    DECLINE:       Professional decline + specific reasons + door open

The reply_type is set by policy_lookup_agent based on:
    - Whether all required information was provided
    - Whether the risk is in appetite
    - Whether it needs referral

FINAL AGENT — writes to uw_decisions table.
After this agent completes, the submission appears in the
Streamlit UI as "Awaiting Review" with the reply ready to send.

INTERVIEW TALKING POINT:
"The last agent in the pipeline writes the final output to
Delta and updates the email status. The Streamlit UI reads
from Delta — agents and UI are completely decoupled. The UI
never calls an agent directly."
"""

import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage

from aria.agents.state import ARIAState, UWDecision
from aria.core.llm import get_llm
from aria.core.delta import (
    write_audit_log,
    write_uw_decision,
    update_email_status,
)


# ── Reply prompts per type ────────────────────────────────────────

INFO_REQUEST_PROMPT = """You are a senior insurance underwriter at a leading Indian insurer.
Write a professional reply to this broker requesting missing information.

Tone: Professional but warm. Clear and specific. Not bureaucratic.
Language: Insurance industry standard. Use correct terminology.

Structure your reply as:
1. Brief acknowledgement of the submission
2. Confirm the risk is within appetite (if it is)
3. List EXACTLY the information required — numbered, specific
4. Timeline expectations
5. Professional close

Do NOT:
- Use jargon the broker won't understand
- Be vague about what information is needed
- Sound like a form letter
- Use the word "please" more than once"""

INDICATIVE_PROMPT = """You are a senior insurance underwriter at a leading Indian insurer.
Write a professional reply providing indicative underwriting terms.

Tone: Professional, confident, clear. This is a preliminary quote.
Language: Insurance industry standard terminology throughout.

Structure your reply as:
1. Brief acknowledgement
2. Confirm risk is within appetite
3. INDICATIVE TERMS section (clearly marked as indicative):
   - Coverage: what is covered
   - Sum insured: as requested or recommended
   - Premium indication: the range (from-to)
   - Key conditions and exclusions
   - Territory
4. What is needed to firm up the quote
5. Timeline and next steps
6. Professional close

Be specific about premium figures. Use INR with lakhs/crores format."""

DECLINE_PROMPT = """You are a senior insurance underwriter at a leading Indian insurer.
Write a professional decline letter to this broker.

Tone: Professional, respectful, specific. Not apologetic but not cold.
This is a business decision, not a rejection of the client.

Structure your reply as:
1. Thank the broker for the submission
2. State clearly that you are unable to offer terms
3. Give SPECIFIC reasons (not generic "outside our appetite")
4. Where possible, suggest what might make the risk acceptable
5. Leave the door open for future submissions
6. Professional close

Never say "we regret" — it sounds insincere.
Be specific about which coverage types are declined and why."""


async def reply_composer_agent(state: ARIAState) -> dict:
    """
    Agent 5: Generate professional broker reply + write UW decision.

    Input:  Full state with extracted_risk, client_history, uw_decision
    Output: {"reply_email_text": str}
    Also writes to uw_decisions table and updates email status.
    """
    uw_decision: UWDecision = state.get("uw_decision")
    extracted = state.get("extracted_risk")

    if not uw_decision or not extracted:
        return {"reply_email_text": "Error: missing decision data"}

    llm = get_llm()
    start_time = datetime.now()

    # ── Build context for the LLM ─────────────────────────────────
    # Give the LLM everything it needs in a structured way
    history = state.get("client_history")

    context = f"""BROKER EMAIL:
{state["raw_email_text"]}

EXTRACTED RISK:
- Insured: {extracted.insured_name}
- Business: {extracted.business_description or "Not specified"}
- Coverage requested: {", ".join(extracted.coverage_requested)}
- Sum insured: {"INR {:,.0f}".format(extracted.sum_insured_inr) if extracted.sum_insured_inr else "Not specified"}
- Turnover: {"{:.1f} Crores".format(extracted.turnover_cr) if extracted.turnover_cr else "Not specified"}
- Territories: {", ".join(extracted.territories) if extracted.territories else "Not specified"}
- Renewal date: {extracted.renewal_date or "Not specified"}

CLIENT HISTORY:
- Known client: {history.is_known_client if history else False}
- Prior policies: {len(history.prior_policies) if history else 0}
- Prior claims: {len(history.prior_claims) if history else 0} (total: {"INR {:,.0f}".format(history.total_claims_amount) if history else 0})
- Red flags: {", ".join(history.red_flags) if history and history.red_flags else "None"}

UNDERWRITING DECISION:
- Appetite: {uw_decision.appetite_decision}
- Premium indication: {"INR {:,.0f} to INR {:,.0f}".format(uw_decision.premium_from, uw_decision.premium_to) if uw_decision.premium_from > 0 else "Cannot indicate without complete information"}
- Confidence: {uw_decision.confidence_score}
- Missing information: {chr(10).join(f"  - {g}" for g in uw_decision.gaps_identified) if uw_decision.gaps_identified else "  None"}"""

    # ── Select prompt based on reply type ─────────────────────────
    prompt_map = {
        "INFO_REQUEST": INFO_REQUEST_PROMPT,
        "INDICATIVE":   INDICATIVE_PROMPT,
        "DECLINE":      DECLINE_PROMPT,
    }
    system_prompt = prompt_map.get(
        uw_decision.reply_type, INFO_REQUEST_PROMPT
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context),
        ])

        reply_email = response.content
        latency = int(
            (datetime.now() - start_time).total_seconds() * 1000
        )

        # ── Write decision to Delta ───────────────────────────────
        decision_id = write_uw_decision(
            email_id=           state["email_id"],
            insured_name=       extracted.insured_name or "Unknown",
            scenario=           extracted.scenario,
            extracted_risk_json=json.dumps(extracted.model_dump()),
            appetite_decision=  uw_decision.appetite_decision,
            premium_from=       uw_decision.premium_from,
            premium_to=         uw_decision.premium_to,
            gaps_identified=    json.dumps(uw_decision.gaps_identified),
            reply_email_text=   reply_email,
            confidence_score=   uw_decision.confidence_score,
        )

        # ── Update email status ───────────────────────────────────
        update_email_status(
            email_id=state["email_id"],
            status="awaiting_review",
        )

        write_audit_log(
            email_id=state["email_id"],
            agent_name="reply_composer_agent",
            action="compose_reply_email",
            input_summary=(
                f"Decision: {uw_decision.appetite_decision} | "
                f"Type: {uw_decision.reply_type}"
            ),
            output_summary=(
                f"decision_id: {decision_id} | "
                f"reply length: {len(reply_email)} chars"
            ),
            latency_ms=latency,
        )

        return {"reply_email_text": reply_email}

    except Exception as e:
        write_audit_log(
            email_id=state["email_id"],
            agent_name="reply_composer_agent",
            action="compose_reply_failed",
            input_summary=f"Decision type: {uw_decision.reply_type}",
            output_summary=f"Error: {str(e)[:200]}",
        )
        return {
            "reply_email_text": (
                f"Error generating reply: {str(e)[:100]}"
            )
        }
