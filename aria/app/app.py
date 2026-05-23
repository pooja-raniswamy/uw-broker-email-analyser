import sys
import os

REPO_ROOT = "/Workspace/Users/swamy.poojarani@gmail.com/uw-broker-email-analyser"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# APP_DIR contains a copy of the aria package — add it to path
for p in [APP_DIR, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

st.set_page_config(
    page_title="ARIA — Underwriting Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "current_view" not in st.session_state:
    st.session_state.current_view = "inbox"
if "selected_email_id" not in st.session_state:
    st.session_state.selected_email_id = None

def go_to_analysis(email_id):
    st.session_state.selected_email_id = email_id
    st.session_state.current_view = "analysis"
    st.rerun()

def go_to_inbox():
    st.session_state.current_view = "inbox"
    st.session_state.selected_email_id = None
    st.rerun()

with st.sidebar:
    st.title("ARIA")
    st.caption("Automated Risk Intelligence for Underwriting")
    st.divider()

    if st.button("📥 Submission Inbox", use_container_width=True,
                 type="primary" if st.session_state.current_view == "inbox" else "secondary"):
        st.session_state.current_view = "inbox"
        st.session_state.selected_email_id = None
        st.rerun()

    if st.button("🏢 Broker Intelligence", use_container_width=True,
                 type="primary" if st.session_state.current_view == "broker" else "secondary"):
        st.session_state.current_view = "broker"
        st.rerun()

    if st.button("📨 Submit New Email", use_container_width=True,
                 type="primary" if st.session_state.current_view == "submit" else "secondary"):
        st.session_state.current_view = "submit"
        st.rerun()

    st.divider()
    st.caption("Built on Databricks")
    st.caption("LangGraph + MCP + Claude")

view = st.session_state.current_view

try:
    if view == "inbox":
        from views.inbox import render_inbox
        render_inbox(on_select_email=go_to_analysis)

    elif view == "analysis":
        email_id = st.session_state.selected_email_id
        if email_id:
            from views.analysis import render_analysis
            render_analysis(email_id=email_id, on_back=go_to_inbox)
        else:
            st.warning("No submission selected.")
            if st.button("Back to Inbox"):
                go_to_inbox()

    elif view == "broker":
        from views.broker_intel import render_broker_intel
        render_broker_intel()

    elif view == "submit":
        from views.submit import render_submit
        render_submit(on_view_result=go_to_analysis)

except Exception as e:
    st.error(f"Error loading view: {e}")
    import traceback
    st.code(traceback.format_exc())
