"""Streamlit UI — calls the FastAPI backend (api.py) over HTTP rather
than importing graph.py directly, so the UI and the research pipeline can
be deployed and scaled as separate Railway services.
"""

import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("SCOUT_API_URL", "http://localhost:8000")

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

if run_clicked and company.strip():
    with st.spinner(f"Researching {company.strip()}… this takes 1–2 minutes"):
        try:
            response = httpx.post(
                f"{API_BASE_URL}/research",
                json={"company": company.strip()},
                timeout=240,
            )
            response.raise_for_status()
            report = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json().get("detail", exc.response.text)
            st.error(f"Research failed: {detail}")
            report = None
        except httpx.RequestError as exc:
            st.error(f"Couldn't reach the research backend: {exc}")
            report = None

    if report:
        st.subheader(report["company"])
        st.caption(f"Generated {report['generated_at']}")
        st.write(report["summary"])

        if report["red_flags"]:
            st.subheader("🚩 Red flags")
            for flag in report["red_flags"]:
                st.error(flag)

        st.subheader("Key facts")
        verified = [f for f in report["key_facts"] if f["confidence"] == "verified"]
        flagged = [f for f in report["key_facts"] if f["confidence"] == "flagged"]

        if verified:
            st.markdown("**✅ Verified** — corroborated by 2+ independent sources")
            for fact in verified:
                sources = " · ".join(
                    f"[source]({url})" for url in fact["sources"]
                )
                st.markdown(f"- {fact['claim']} ({sources})")

        if flagged:
            st.markdown("**⚑ Flagged** — single-sourced or conflicting, unverified")
            for fact in flagged:
                sources = " · ".join(
                    f"[source]({url})" for url in fact["sources"]
                )
                st.markdown(f"- {fact['claim']} ({sources})")

        if not verified and not flagged:
            st.info("No public information could be found for this company.")

        with st.expander(f"All {len(report['sources'])} sources"):
            for url in report["sources"]:
                st.markdown(f"- {url}")
