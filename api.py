"""FastAPI entrypoint — thin HTTP layer in front of graph.run().

Streamlit (app.py) is a separate process that calls this over HTTP rather
than importing graph.py directly. That split is what actually needs a
persistent backend process instead of a pure-Streamlit deploy: the graph
run is a slow (~90s), stateful, API-key-holding operation, and keeping it
behind its own service means the UI can be redeployed or scaled
independently of it.
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph import run
from schema import DueDiligenceReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scout.api")

app = FastAPI(
    title="Scout",
    description=(
        "Autonomous due-diligence research agent for UAE companies. "
        "Research aid for a first pass — not a KYC/compliance decisioning "
        "tool. Public, non-authenticated web sources only."
    ),
)


class ResearchRequest(BaseModel):
    company: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/research", response_model=DueDiligenceReport)
def research(req: ResearchRequest) -> DueDiligenceReport:
    company = req.company.strip()
    if not company:
        raise HTTPException(status_code=400, detail="company name is required")

    logger.info("research request: %s", company)
    try:
        report = run(company)
    except Exception:
        logger.exception("research failed for %s", company)
        raise HTTPException(
            status_code=502,
            detail=(
                "Research pipeline failed — usually a search or model API "
                "issue. Try again in a moment."
            ),
        )

    if report is None:
        raise HTTPException(status_code=502, detail="no report was produced")
    return report
