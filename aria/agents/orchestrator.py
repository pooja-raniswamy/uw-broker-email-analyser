"""
aria/agents/orchestrator.py

ARIA orchestrator — the main LangGraph StateGraph.

WHAT THIS FILE DOES:
Wires all 5 agents into a directed graph with:
- Sequential steps where order matters
- Parallel execution where order does not matter
- HITL interrupt for human review
- Lakebase checkpointing for cross-session memory

GRAPH STRUCTURE:
    START
      ↓
    classify_and_extract     (Agent 2 — risk_extractor_agent)
      ↓
    run_parallel_lookups     (Agents 3+4 — parallel via asyncio.gather)
      ↓
    check_hitl               (conditional — interrupt if review needed)
      ↓
    compose_reply            (Agent 5 — reply_composer_agent)
      ↓
    END

MEMORY:
    AsyncPostgresSaver (Lakebase) checkpoints state after every node.
    thread_id = broker_email + email_id for cross-session persistence.
    Follow-up emails from same broker load previous state automatically.

INTERVIEW TALKING POINTS:
1. "asyncio.gather() runs history and policy lookup in parallel —
    total latency is max(history, policy) not sum"
2. "interrupt() pauses the graph and persists state to Lakebase —
    human can review hours later and graph resumes exactly where
    it stopped"
3. "thread_id pattern gives broker-scoped episodic memory across
    sessions — the agent remembers previous submissions"
"""

import asyncio
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt

from aria.agents.state import ARIAState
from aria.agents.risk_extractor import risk_extractor_agent
from aria.agents.history_lookup import history_lookup_agent
from aria.agents.policy_lookup import policy_lookup_agent
from aria.agents.reply_composer import reply_composer_agent
from aria.core.delta import write_audit_log, update_email_status
from aria.config import settings


# ── Graph nodes ───────────────────────────────────────────────────

async def classify_and_extract(state: ARIAState) -> dict:
    """
    Node 1: Run risk_extractor_agent.

    Updates email status to "processing" so Streamlit inbox
    shows the submission is being worked on.
    Then extracts structured risk from the broker email.
    """
    update_email_status(state["email_id"], "processing")
    return await risk_extractor_agent(state)


async def run_parallel_lookups(state: ARIAState) -> dict:
    """
    Node 2: Run history_lookup and policy_lookup IN PARALLEL.

    asyncio.gather() starts both coroutines simultaneously.
    Both must complete before this node returns.

    Why parallel:
        Both agents need the same input (extracted_risk).
        Neither depends on the other's output.
        Running sequentially would double latency for no reason.

    The merged result dict updates both client_history
    and (policy_rules + uw_decision) in the state simultaneously.
    LangGraph merges partial update dicts cleanly.
    """
    if not state.get("extracted_risk"):
        # Extraction failed — skip lookups
        return {}

    history_result, policy_result = await asyncio.gather(
        history_lookup_agent(state),
        policy_lookup_agent(state),
        return_exceptions=True,   # one failing does not kill the other
    )

    merged = {}

    # Handle history result
    if isinstance(history_result, Exception):
        write_audit_log(
            email_id=state["email_id"],
            agent_name="orchestrator",
            action="history_lookup_exception",
            input_summary="parallel execution",
            output_summary=str(history_result)[:200],
        )
    else:
        merged.update(history_result)

    # Handle policy result
    if isinstance(policy_result, Exception):
        write_audit_log(
            email_id=state["email_id"],
            agent_name="orchestrator",
            action="policy_lookup_exception",
            input_summary="parallel execution",
            output_summary=str(policy_result)[:200],
        )
    else:
        merged.update(policy_result)

    return merged


async def check_hitl(state: ARIAState) -> dict:
    """
    Node 3: Check if human review is required.

    If uw_decision.requires_senior_review is True:
        interrupt() is called — graph PAUSES here.
        Current state is saved to Lakebase checkpointer.
        Streamlit UI shows submission as "Awaiting Review".
        Graph resumes when human approves via UI.

    If no review needed:
        Returns empty dict — state unchanged.
        Graph continues to compose_reply.

    WHY interrupt() NOT a conditional edge:
        interrupt() preserves state across the pause.
        A conditional edge would skip the node entirely.
        We need the graph to PAUSE at this node, not skip it.
    """
    uw_decision = state.get("uw_decision")

    if not uw_decision:
        return {}

    # Also check client history red flags
    history = state.get("client_history")
    history_flags = len(history.red_flags) if history else 0

    needs_review = (
        uw_decision.requires_senior_review or
        history_flags > 0
    )

    if needs_review:
        # Build a clear summary for the human reviewer
        reasons = []
        if uw_decision.appetite_decision == "REFER":
            reasons.append("Risk requires referral")
        if uw_decision.confidence_score < settings.hitl_confidence_threshold:
            reasons.append(
                f"Low confidence: {uw_decision.confidence_score:.0%}"
            )
        if history and history.red_flags:
            reasons.extend(history.red_flags)
        if uw_decision.gaps_identified:
            reasons.append(
                f"{len(uw_decision.gaps_identified)} information gaps"
            )

        write_audit_log(
            email_id=state["email_id"],
            agent_name="orchestrator",
            action="hitl_triggered",
            input_summary=f"Insured: {state.get('extracted_risk', {}).insured_name if state.get('extracted_risk') else 'unknown'}",
            output_summary=" | ".join(reasons),
        )

        update_email_status(state["email_id"], "awaiting_review")

        # PAUSE HERE — graph resumes when human approves
        interrupt({
            "reason": " | ".join(reasons),
            "email_id": state["email_id"],
        })

    return {}


async def compose_reply(state: ARIAState) -> dict:
    """
    Node 4: Run reply_composer_agent.

    Final node — generates the reply email and writes
    the UW decision to uw_decisions table.
    """
    return await reply_composer_agent(state)


# ── Graph builder ─────────────────────────────────────────────────

def build_graph(checkpointer=None) -> object:
    """
    Build and compile the ARIA StateGraph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                     Pass AsyncPostgresSaver for Lakebase memory.
                     Pass None for testing (in-memory only).

    Returns:
        Compiled LangGraph graph ready to invoke.

    Usage:
        # With Lakebase memory
        checkpointer = await get_lakebase_checkpointer()
        graph = build_graph(checkpointer=checkpointer)

        # Without memory (testing)
        graph = build_graph()

        # Run pipeline
        config = {"configurable": {"thread_id": "broker@email.com_em_001"}}
        result = await graph.ainvoke(initial_state, config=config)
    """
    builder = StateGraph(ARIAState)

    # ── Add nodes ─────────────────────────────────────────────────
    builder.add_node("classify_and_extract", classify_and_extract)
    builder.add_node("run_parallel_lookups", run_parallel_lookups)
    builder.add_node("check_hitl",           check_hitl)
    builder.add_node("compose_reply",         compose_reply)

    # ── Add edges — defines execution order ───────────────────────
    builder.add_edge(START,                    "classify_and_extract")
    builder.add_edge("classify_and_extract",   "run_parallel_lookups")
    builder.add_edge("run_parallel_lookups",   "check_hitl")
    builder.add_edge("check_hitl",             "compose_reply")
    builder.add_edge("compose_reply",          END)

    # ── Compile with checkpointer ─────────────────────────────────
    # interrupt_before: list of node names where interrupt() may fire
    # Without this, interrupt() does not work correctly
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["compose_reply"],
    )


async def run_pipeline(
    email_id:       str,
    broker_email:   str,
    raw_email_text: str,
    checkpointer=None,
) -> dict:
    """
    Convenience function to run the full ARIA pipeline.

    Args:
        email_id:       ID of the broker_emails row
        broker_email:   Broker's email address
        raw_email_text: Full email body text
        checkpointer:   Lakebase checkpointer (optional)

    Returns:
        Final state dict after pipeline completes.

    The thread_id pattern:
        broker_email scopes memory to this broker.
        email_id makes each submission unique.
        Together: same broker, different submissions, all remembered.
    """
    graph = build_graph(checkpointer=checkpointer)

    initial_state = ARIAState(
        email_id=           email_id,
        broker_email=       broker_email,
        raw_email_text=     raw_email_text,
        extracted_risk=     None,
        policy_rules=       [],
        client_history=     None,
        uw_decision=        None,
        reply_email_text=   None,
        needs_senior_review=False,
        error_message=      None,
        messages=           [],
    )

    # thread_id links this run to any previous runs for this broker
    config = {
        "configurable": {
            "thread_id": f"{broker_email}_{email_id}"
        }
    }

    try:
        result = await graph.ainvoke(initial_state, config=config)
        return result
    except Exception as e:
        write_audit_log(
            email_id=email_id,
            agent_name="orchestrator",
            action="pipeline_failed",
            input_summary=f"broker: {broker_email}",
            output_summary=str(e)[:200],
        )
        raise
