"""Cross-source verification logic. The whole point of this project.

The sketch this was built from assumed raw findings arrive pre-grouped by
identical claim text, then just counts domains per group. Real search
results never share exact wording even when they agree on the fact, so
clustering happens earlier (an LLM pass in graph.py turns raw findings
into ClaimClusters). What's left here is the part that should stay
boring: given a claim and the domains backing it, decide verified vs.
flagged. No model call, no clustering judgment — pure arithmetic on
domain names, fully unit-testable, and the actual rule an auditor can
point to and check.
"""

from schema import ClaimCluster
from tools import domain_of


def verify(clusters: list[ClaimCluster]) -> tuple[list[ClaimCluster], list[ClaimCluster]]:
    verified, flagged = [], []
    for cluster in clusters:
        distinct_domains = {domain_of(str(url)) for url in cluster.source_urls}
        (verified if len(distinct_domains) >= 2 else flagged).append(cluster)
    return verified, flagged
