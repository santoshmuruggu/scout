"""DIFC / DFSA public register lookup tool node.

Two registers matter here, and they behave very differently:

- difc.com's Registrar of Companies pages (public-register-details) render
  their structured fields (status, license type, directors) via a
  client-side call under /api/ — a path difc.com/robots.txt explicitly
  disallows crawling. Scout doesn't work around that: it cites the page
  URL rather than scraping the JS-rendered fields inside it.
- dfsa.ae's Public Register of regulated firms is the opposite: plain
  server-rendered HTML, nothing in robots.txt blocks it, and it carries
  real structured fields (legal status, DFSA reference number, licence
  date, withdrawal date, endorsements, regulatory actions) — genuinely
  useful for a DIFC-based *financial services* firm. It only covers firms
  DFSA regulates, not every DIFC company, so it's a bonus source, not a
  universal one.

Both registers' own search endpoints return 403s to direct queries (no
public API, exactly the problem this project is built around) — so
discovery goes through a broad Tavily web search first, and only the
specific allowed page URLs that search turns up get fetched directly.
"""

import re

from schema import Finding
from tools import domain_of, tavily_client, with_retry

DFSA_PATH = "dfsa.ae/public-register/firms/"
DIFC_REGISTER_PATH = "difc.com/business/public-register/public-register-details"

DFSA_FIELDS = [
    "Legal Status",
    "DFSA Reference Number",
    "Address",
    "Telephone Number",
    "Date of Licence",
    "Date of Withdrawal",
    "Endorsements",
    "Financial Service",
]

# The register's field set differs by entity type (regulated firm vs.
# "Non-DIFC Company" vs. recognised body, etc). This is every label variant
# observed so far, used only to detect "this capture is actually the next
# label, because the field itself was blank" — not an exhaustive schema.
KNOWN_LABELS = set(DFSA_FIELDS) | {"Date of Recognition", "Date of Authorisation"}


@with_retry
def _discover(company: str) -> dict:
    return tavily_client().search(
        query=f'"{company}" DFSA DIFC public register',
        max_results=5,
        search_depth="advanced",
    )


@with_retry
def _extract(url: str) -> str:
    response = tavily_client().extract(urls=[url])
    results = response.get("results", [])
    return results[0]["raw_content"] if results else ""


def _parse_dfsa_fields(text: str) -> dict[str, str]:
    """Pull labelled fields out of the DFSA page's extracted text.

    When a field is blank, the regex below matches straight through to the
    next label instead — there's no other delimiter to catch it on. Detect
    that by checking whether the "value" is itself just another field
    label, and drop it rather than mislabel one field as another's value.
    """
    fields = {}
    for label in DFSA_FIELDS:
        match = re.search(rf"{re.escape(label)}:?\n\n(.{{1,300}}?)\n\n", text, re.S)
        if not match:
            continue
        value = " ".join(match.group(1).split())
        if not value or value.rstrip(":") in KNOWN_LABELS:
            continue
        fields[label] = value
    return fields


def lookup_difc_entity(company: str) -> list[Finding]:
    response = _discover(company)
    findings: list[Finding] = []

    for result in response.get("results", []):
        url = result.get("url", "")

        if DFSA_PATH in url:
            fields = _parse_dfsa_fields(_extract(url))
            for label, value in fields.items():
                findings.append(
                    Finding(
                        claim=f"{company} — DFSA register {label}: {value}",
                        source_url=url,
                        source_domain=domain_of(url),
                        snippet=value,
                        sub_question="DIFC/DFSA registration lookup",
                    )
                )

        elif DIFC_REGISTER_PATH in url:
            findings.append(
                Finding(
                    claim=(
                        f"DIFC Registrar public-register entry exists for "
                        f"'{company}' at this URL (fields are JS-rendered and "
                        f"not scraped — see source for the full record)"
                    ),
                    source_url=url,
                    source_domain=domain_of(url),
                    snippet=result.get("content"),
                    sub_question="DIFC/DFSA registration lookup",
                )
            )

    return findings
