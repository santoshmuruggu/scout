"""Streamlit UI — calls the FastAPI backend (api.py) over HTTP rather
than importing graph.py directly, so the UI and the research pipeline can
be deployed and scaled as separate Railway services.
"""

import os
import time

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("SCOUT_API_URL", "http://localhost:8000")
POLL_INTERVAL_SECONDS = 3
MAX_WAIT_SECONDS = 360

st.set_page_config(page_title="Scout", page_icon="🔎", layout="centered")

st.title("🔎 Scout")
st.caption("Autonomous due-diligence research for UAE companies")
st.warning(
    "Research aid for a first pass, not a KYC/compliance decisioning "
    "tool. Public, non-authenticated web sources only — always verify "
    "before relying on this for a real decision.",
    icon="⚠️",
)

company = st.text_input("Company name", placeholder="e.g. Beehive, Sarwa, Careem")
run_clicked = st.button("Research", type="primary", disabled=not company.strip())


def _md_safe(text: str) -> str:
    # Report text is full of dollar amounts ("$3.1 billion ... $400
    # million"). Streamlit's markdown renderer treats $...$ as LaTeX math
    # delimiters, so two dollar amounts on the same block get parsed as
    # one giant math expression — the actual bug behind the garbled
    # "3.1\nb\ni\nl\nl\ni\no\nn..." rendering. Escaping $ turns it back
    # into a literal character.
    return text.replace("$", "\\$")

def _error_detail(response: httpx.Response) -> str:
    # The error body isn't guaranteed to be our own JSON error shape — a
    # proxy/gateway error returns an HTML page instead, which .json()
    # would choke on and crash the error handler itself.
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return f"HTTP {response.status_code}"


if run_clicked and company.strip():
    report = None
    try:
        # Submit-then-poll instead of one long-held request: each call
        # here is fast (a job kickoff, then cheap status checks), so
        # there's no long-running connection for a platform proxy to
        # time out mid-request regardless of how long the actual
        # research takes.
        submit = httpx.post(
            f"{API_BASE_URL}/research", json={"company": company.strip()}, timeout=15
        )
        submit.raise_for_status()
        job_id = submit.json()["job_id"]

        with st.spinner(f"Researching {company.strip()}… this takes 1–3 minutes"):
            elapsed = 0
            while elapsed < MAX_WAIT_SECONDS:
                status_response = httpx.get(f"{API_BASE_URL}/research/{job_id}", timeout=15)
                status_response.raise_for_status()
                job = status_response.json()

                if job["status"] == "done":
                    report = job["report"]
                    break
                if job["status"] == "error":
                    st.error(f"Research failed: {job['error']}")
                    break

                time.sleep(POLL_INTERVAL_SECONDS)
                elapsed += POLL_INTERVAL_SECONDS
            else:
                st.error("Research is taking longer than expected — try again shortly.")
    except httpx.HTTPStatusError as exc:
        st.error(f"Research failed: {_error_detail(exc.response)}")
    except httpx.RequestError as exc:
        st.error(f"Couldn't reach the research backend: {exc}")

    if report:
        st.subheader(_md_safe(report["company"]))
        st.caption(f"Generated {report['generated_at']}")
        st.write(_md_safe(report["summary"]))

        if report["red_flags"]:
            st.subheader("🚩 Red flags")
            for flag in report["red_flags"]:
                st.error(_md_safe(flag))

        st.subheader("Key facts")
        verified = [f for f in report["key_facts"] if f["confidence"] == "verified"]
        flagged = [f for f in report["key_facts"] if f["confidence"] == "flagged"]

        if verified:
            st.markdown("**✅ Verified** — corroborated by 2+ independent sources")
            for fact in verified:
                sources = " · ".join(
                    f"[source]({url})" for url in fact["sources"]
                )
                st.markdown(f"- {_md_safe(fact['claim'])} ({sources})")

        if flagged:
            st.markdown("**⚑ Flagged** — single-sourced or conflicting, unverified")
            for fact in flagged:
                sources = " · ".join(
                    f"[source]({url})" for url in fact["sources"]
                )
                st.markdown(f"- {_md_safe(fact['claim'])} ({sources})")

        if not verified and not flagged:
            st.info("No public information could be found for this company.")

        with st.expander(f"All {len(report['sources'])} sources"):
            for url in report["sources"]:
                st.markdown(f"- {url}")
