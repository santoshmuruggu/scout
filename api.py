"""FastAPI entrypoint — thin HTTP layer in front of graph.run().

Streamlit (app.py) is a separate process that calls this over HTTP rather
than importing graph.py directly. That split is what actually needs a
persistent backend process instead of a pure-Streamlit deploy: the graph
run is a slow (~90-180s), stateful, API-key-holding operation, and keeping
it behind its own service means the UI can be redeployed or scaled
independently of it.

The research endpoint is submit-then-poll (POST to start, GET to check),
not a single long-held request. A run genuinely takes 1-3 minutes, and a
platform proxy sitting in front of a free-tier service has no obligation
to hold a connection open that long — better to make every individual
request fast and let the client poll than to gamble on some undocumented
timeout number staying above the pipeline's worst case.

Bring-your-own-key: in the default (Claude) mode every research request
must carry the caller's own Anthropic API key. The server never falls
back to an ANTHROPIC_API_KEY configured on the host. The key only lives
in memory for the length of that one job: it isn't stored in the job
table, logged, or put into graph state.
"""

import logging
import threading
import uuid
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import llm
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

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


class ResearchRequest(BaseModel):
    company: str
    # repr=False keeps the key out of any log line or error message that
    # prints the request model.
    anthropic_api_key: str | None = Field(default=None, repr=False)


class ServerConfig(BaseModel):
    provider: str
    model: str
    needs_api_key: bool


class JobAccepted(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    status: Literal["running", "done", "error"]
    report: DueDiligenceReport | None = None
    error: str | None = None


def _run_job(job_id: str, company: str, api_key: str | None) -> None:
    try:
        report = run(company, api_key)
        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "report": report, "error": None}
    except Exception as exc:
        logger.exception("research failed for %s", company)
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "report": None, "error": str(exc)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/config", response_model=ServerConfig)
def server_config() -> ServerConfig:
    """Lets the UI know whether to ask the visitor for an API key."""
    return ServerConfig(
        provider=llm.PROVIDER, model=llm.model_name(), needs_api_key=llm.needs_user_key()
    )


@app.post("/research", response_model=JobAccepted, status_code=202)
def start_research(req: ResearchRequest) -> JobAccepted:
    company = req.company.strip()
    if not company:
        raise HTTPException(status_code=400, detail="company name is required")

    api_key = None
    if llm.PROVIDER != "ollama":
        api_key = (req.anthropic_api_key or "").strip() or llm.local_anthropic_key()
        if not api_key:
            raise HTTPException(status_code=401, detail="An Anthropic API key is required.")
        try:
            llm.check_anthropic_key(api_key)
        except anthropic.AuthenticationError:
            raise HTTPException(status_code=401, detail="Anthropic rejected that API key.")
        except anthropic.APIError as exc:
            # Couldn't verify (network blip, Anthropic overloaded): let
            # the job try anyway rather than block on the pre-check.
            logger.warning("couldn't pre-check API key: %s", type(exc).__name__)

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "report": None, "error": None}

    logger.info("research job %s started: %s", job_id, company)
    threading.Thread(target=_run_job, args=(job_id, company, api_key), daemon=True).start()
    return JobAccepted(job_id=job_id)


@app.get("/research/{job_id}", response_model=JobStatus)
def get_research(job_id: str) -> JobStatus:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(**job)
