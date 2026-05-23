"""
aria/app/main.py

ARIA Streamlit application entry point.

Wires all 4 views together using session_state for navigation.
Streamlit does not have real routing — we manage view switching
manually using st.session_state.current_view.

Views:
    inbox      — submission inbox (default landing page)
    analysis   — per-submission agent outputs + approve
    broker     — broker intelligence charts
    submit     — paste new email → run pipeline

Navigation:
    Left sidebar shows current view and navigation buttons.
    Clicking a submission in inbox sets current_view=analysis.
    Clicking Submit in submit view sets current_view=analysis.
    Back button in analysis sets current_view=inbox.

Run locally:
    streamlit run aria/app/main.py

Deploy as Databricks App:
    databricks bundle deploy
    (uses databricks/databricks.yml Asset Bundle config)
"""

import sys
sys.path.insert(
    0,
    "/Workspace/Users/swamy.poojarani@gmail.com/uw-broker-email-analyser"
)

import streamlit as st

from aria.app.views.inbox       import render_inbox
from aria.app.views.analysis    import render_analysis
from aria.app.views.broker_intel import render_broker_intel
from aria.app.views.submit      import render_submit


# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="ARIA — Underwriting Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state defaults ────────────────────────────────────────
if "current_view" not in st.session_state:
    st.session_state.current_view = "inbox"

if "selected_email_id" not in st.session_state:
    st.session_state.selected_email_id = None


# ── Navigation callbacks ──────────────────────────────────────────
def go_to_analysis(email_id: str):
    """Switch to analysis view for a specific email."""
    st.session_state.selected_email_id = email_id
    st.session_state.current_view = "analysis"
    st.rerun()


def go_to_inbox():
    """Return to inbox view."""
    st.session_state.current_view = "inbox"
    st.session_state.selected_email_id = None
    st.rerun()


# ── Sidebar navigation ────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/bank-building.png",
        width=60
    )
    st.title("ARIA")
    st.caption("Automated Risk Intelligence
for Underwriting")
    st.divider()

    # Navigation buttons
    if st.button(
        "📥 Submission Inbox",
        use_container_width=True,
        type="primary" if st.session_state.current_view == "inbox"
             else "secondary"
    ):
        st.session_state.current_view = "inbox"
        st.session_state.selected_email_id = None
        st.rerun()

    if st.button(
        "🏢 Broker Intelligence",
        use_container_width=True,
        type="primary" if st.session_state.current_view == "broker"
             else "secondary"
    ):
        st.session_state.current_view = "broker"
        st.rerun()

    if st.button(
        "📨 Submit New Email",
        use_container_width=True,
        type="primary" if st.session_state.current_view == "submit"
             else "secondary"
    ):
        st.session_state.current_view = "submit"
        st.rerun()

    st.divider()
    st.caption("Built on Databricks")
    st.caption("LangGraph + MCP + Claude")


# ── Route to correct view ─────────────────────────────────────────
view = st.session_state.current_view

if view == "inbox":
    render_inbox(on_select_email=go_to_analysis)

elif view == "analysis":
    email_id = st.session_state.selected_email_id
    if email_id:
        render_analysis(
            email_id=email_id,
            on_back=go_to_inbox
        )
    else:
        st.warning("No submission selected. Go back to inbox.")
        if st.button("← Back to Inbox"):
            go_to_inbox()

elif view == "broker":
    render_broker_intel()

elif view == "submit":
    render_submit(on_view_result=go_to_analysis)

else:
    st.error(f"Unknown view: {view}")
    go_to_inbox()
