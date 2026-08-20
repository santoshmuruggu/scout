"""Eval runner against eval/companies.json.

Reports the four numbers the project's premise actually rests on:
factual accuracy, flagging precision, tool-call reliability, cost/latency.

Matching Scout's free-text key_facts against free-text ground-truth facts
isn't exact-string territory, so an LLM judges each ground-truth fact
against the report: did Scout find it, and does Scout's confidence label
agree with ground truth's. That judgment is the one soft part of this
script — everything it's built on (verified/flagged, tool-call success)
is counted directly from real run data, not judged.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Literal

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from graph import run_state
from schema import DueDiligenceReport

EVAL_PATH = Path(__file__).parent / "companies.json"


class GroundTruthJudgement(BaseModel):
    ground_truth_claim: str
    found_in_report: bool
    scout_confidence: Literal["verified", "flagged", "not_found"]
    agrees_with_ground_truth: bool


class ReportJudgement(BaseModel):
    judgements: list[GroundTruthJudgement]


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_PATH.read_text())["companies"]


def judge_report(company: dict, report: DueDiligenceReport | None) -> ReportJudgement:
    ground_truth = company["ground_truth_facts"]
    if report is None or not report.key_facts:
        return ReportJudgement(
            judgements=[
                GroundTruthJudgement(
                    ground_truth_claim=g["claim"],
                    found_in_report=False,
                    scout_confidence="not_found",
                    agrees_with_ground_truth=False,
                )
                for g in ground_truth
            ]
        )

    scout_facts_text = "\n".join(
        f"- [{f.confidence}] {f.claim}" for f in report.key_facts
    )
    ground_truth_text = "\n".join(
        f"- [{g['confidence']}] {g['claim']}" for g in ground_truth
    )

    llm = ChatAnthropic(model="claude-sonnet-5", max_tokens=8192)
    structured_llm = llm.with_structured_output(ReportJudgement)
    prompt = (
        f"Company: {company['name']}\n\n"
        f"Ground-truth facts (hand-researched):\n{ground_truth_text}\n\n"
        f"Scout's key_facts (what the agent under test produced):\n{scout_facts_text}\n\n"
        "For EACH ground-truth fact, decide:\n"
        "- found_in_report: does any Scout key_fact assert the same "
        "underlying claim (paraphrased is fine, doesn't need exact "
        "wording)?\n"
        "- scout_confidence: the confidence Scout assigned to the "
        "matching claim (\"verified\" or \"flagged\"), or \"not_found\" "
        "if Scout has no matching claim at all.\n"
        "- agrees_with_ground_truth: true only if found_in_report is "
        "true AND Scout's confidence label matches the ground truth's "
        "confidence label.\n"
        "Be strict about 'same claim' — a vaguely related statement "
        "doesn't count as a match.\n"
    )
    return structured_llm.invoke(prompt)


def run_eval(companies: list[dict] | None = None) -> dict:
    companies = companies if companies is not None else load_eval_set()
    per_company = []

    for company in companies:
        start = time.monotonic()
        try:
            state = run_state(company["name"])
            error = None
        except Exception as exc:
            state = None
            error = str(exc)
        elapsed = time.monotonic() - start

        report = state["report"] if state else None
        judgement = judge_report(company, report)

        matched = [j for j in judgement.judgements if j.found_in_report]
        agreeing = [j for j in judgement.judgements if j.agrees_with_ground_truth]

        # Positional pairing with ground_truth_facts — the judge prompt
        # asks for one judgement per ground-truth fact, in order. Used to
        # tell apart the two disagreement directions, which mean very
        # different things: Scout finding MORE corroboration than a quick
        # draft-research pass found (gt flagged -> scout verified, likely
        # Scout being more thorough) versus Scout actually missing
        # corroboration or the claim entirely (the other two directions).
        directions = {"gt_flagged_scout_verified": 0, "gt_verified_scout_flagged": 0}
        misses = 0
        for gt, j in zip(company["ground_truth_facts"], judgement.judgements):
            if j.scout_confidence == "not_found":
                misses += 1
            elif gt["confidence"] == "flagged" and j.scout_confidence == "verified":
                directions["gt_flagged_scout_verified"] += 1
            elif gt["confidence"] == "verified" and j.scout_confidence == "flagged":
                directions["gt_verified_scout_flagged"] += 1

        per_company.append(
            {
                "company": company["name"],
                "tier": company["tier"],
                "run_error": error,
                "elapsed_seconds": round(elapsed, 1),
                "tool_calls": state["tool_calls"] if state else 0,
                "tool_errors": state["tool_errors"] if state else [],
                "report_verified_count": len(
                    [f for f in report.key_facts if f.confidence == "verified"]
                )
                if report
                else 0,
                "report_flagged_count": len(
                    [f for f in report.key_facts if f.confidence == "flagged"]
                )
                if report
                else 0,
                "ground_truth_facts": len(company["ground_truth_facts"]),
                "ground_truth_matched": len(matched),
                "ground_truth_confidence_agrees": len(agreeing),
                "ground_truth_not_found": misses,
                "gt_flagged_scout_verified": directions["gt_flagged_scout_verified"],
                "gt_verified_scout_flagged": directions["gt_verified_scout_flagged"],
                "judgements": [j.model_dump() for j in judgement.judgements],
            }
        )

    return {"summary": summarize(per_company), "per_company": per_company}


def summarize(per_company: list[dict]) -> dict:
    total_gt = sum(c["ground_truth_facts"] for c in per_company)
    total_matched = sum(c["ground_truth_matched"] for c in per_company)
    total_agree = sum(c["ground_truth_confidence_agrees"] for c in per_company)
    total_tool_calls = sum(c["tool_calls"] for c in per_company)
    total_tool_errors = sum(len(c["tool_errors"]) for c in per_company)

    sparse = [c for c in per_company if c["tier"] == "sparse_conflicting"]
    sparse_gt = sum(c["ground_truth_facts"] for c in sparse)
    sparse_agree = sum(c["ground_truth_confidence_agrees"] for c in sparse)

    total_gt_flagged_scout_verified = sum(c["gt_flagged_scout_verified"] for c in per_company)
    total_gt_verified_scout_flagged = sum(c["gt_verified_scout_flagged"] for c in per_company)

    summary = {
        "companies_run": len(per_company),
        "runs_with_errors": len([c for c in per_company if c["run_error"]]),
        "factual_recall": round(total_matched / total_gt, 3) if total_gt else None,
        "confidence_agreement": round(total_agree / total_gt, 3) if total_gt else None,
        "confidence_disagreement_breakdown": {
            "gt_flagged_scout_verified": total_gt_flagged_scout_verified,
            "gt_verified_scout_flagged": total_gt_verified_scout_flagged,
            "note": (
                "gt_flagged_scout_verified is usually NOT a Scout error — "
                "ground truth confidence reflects how many sources a quick "
                "draft-research pass happened to find, and Scout's live "
                "multi-search pipeline often finds more corroboration than "
                "that pass did. gt_verified_scout_flagged is the direction "
                "that would actually indicate Scout under-verifying."
            ),
        },
        "flagging_precision_on_sparse_tier": (
            round(sparse_agree / sparse_gt, 3) if sparse_gt else None
        ),
        "tool_call_success_rate": (
            round(1 - total_tool_errors / total_tool_calls, 3) if total_tool_calls else None
        ),
        "avg_elapsed_seconds": round(
            sum(c["elapsed_seconds"] for c in per_company) / len(per_company), 1
        )
        if per_company
        else None,
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="run only the first N companies"
    )
    parser.add_argument("--out", type=str, default=None, help="write full JSON results here")
    args = parser.parse_args()

    companies = load_eval_set()
    if args.limit:
        companies = companies[: args.limit]

    results = run_eval(companies)
    print(json.dumps(results["summary"], indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nfull results written to {args.out}")
