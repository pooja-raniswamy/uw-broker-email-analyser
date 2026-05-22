"""
aria/agents/risk_extractor.py

Agent 2: Extracts structured risk data from broker email.

SINGLE RESPONSIBILITY:
This agent does exactly one thing — turns unstructured broker
email text into a clean ExtractedRisk Pydantic object.
Nothing else. No tool calls. No database writes (except audit log).

WHY with_structured_output:
Forces Claude to return valid JSON matching ExtractedRisk schema.
Pydantic validates every field. Wrong type = ValidationError = retry.
This is more reliable than asking Claude to "format as JSON" in
the prompt — LangChain handles the schema injection automatically.

WHY RETRY LOOP:
LLMs occasionally return malformed JSON or wrong field types,
especially on complex nested structures. Three retries with the
error message injected back into the prompt is enough to handle
all but the most adversarial inputs.

INTERVIEW TALKING POINT:
"structured output with retry is the production pattern for
reliable LLM extraction. Never trust a single LLM call for
business-critical data extraction."
"""

from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage

from aria.agents.state import ARIAState, ExtractedRisk
from aria.core.llm import get_llm
from aria.core.delta import write_audit_log


SYSTEM_PROMPT = """You are an expert insurance underwriting analyst.
Extract structured risk information from broker emails.

Rules:
- Extract ONLY what is explicitly stated — never assume or invent
- If a field is not mentioned, leave it as null
- For territories: always include 'India' if the business is Indian
- For scenario:
    'renewal' if email mentions renewal / expiry / renew
    'followup' if it says 'further to' or 'as requested' or 'additional info'
    'new_submission' for everything else
- For missing_fields: list every field needed to quote but not provided
  Common missing fields: sum_insured_inr, turnover_cr, renewal_date
- For territories with USA exposure: add 'USA' to territories list
- Convert turnover to Crores (1 Crore = 10,000,000 INR)
- Convert sum insured to INR numbers (3 Crores = 30000000.0)
- For coverage_requested: use snake_case
  ('Product Liability' → 'product_liability')"""


async def risk_extractor_agent(state: ARIAState) -> dict:
    """
    Agent 2: Extract structured risk from broker email.

    Input:  state["raw_email_text"] — messy broker email
    Output: {"extracted_risk": ExtractedRisk} or
            {"extracted_risk": None, "error_message": str}

    Uses with_structured_output to force Claude to return
    valid JSON matching ExtractedRisk schema.
    Retries up to 3 times on ValidationError.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(ExtractedRisk)
    start_time = datetime.now()
    last_error = None

    for attempt in range(3):
        try:
            user_prompt = f"""Extract all risk information from this broker email:

{state["raw_email_text"]}

Return structured data matching the ExtractedRisk schema exactly."""

            # On retry: include the previous error so Claude
            # knows what to fix
            if attempt > 0 and last_error:
                user_prompt += f"""

Previous attempt failed with: {str(last_error)[:200]}
Please fix the validation error and try again."""

            result = await structured_llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])

            latency = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            write_audit_log(
                email_id=state["email_id"],
                agent_name="risk_extractor_agent",
                action="extract_risk_from_email",
                input_summary=f"Email length: {len(state['raw_email_text'])} chars",
                output_summary=(
                    f"Extracted: {result.insured_name or 'unknown'} | "
                    f"scenario: {result.scenario} | "
                    f"missing: {len(result.missing_fields)} fields"
                ),
                latency_ms=latency,
            )

            return {"extracted_risk": result}

        except Exception as e:
            last_error = e
            if attempt < 2:
                continue

    # All 3 attempts failed
    write_audit_log(
        email_id=state["email_id"],
        agent_name="risk_extractor_agent",
        action="extract_risk_failed",
        input_summary=f"Email length: {len(state['raw_email_text'])} chars",
        output_summary=f"Failed after 3 attempts: {str(last_error)[:200]}",
    )

    return {
        "extracted_risk": None,
        "error_message": f"Risk extraction failed: {str(last_error)[:200]}",
    }
