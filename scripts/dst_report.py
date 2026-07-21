"""Summarise turn traces: how far the orchestrator's synthesis diverges from the fusion.

    python -m scripts.dst_report [path/to/turns.jsonl]

Every turn writes a paired observation — what the model concluded, and what the retrieved
evidence supports — because the orchestrator never sees the fusion. This turns those
pairs into the numbers worth reporting.

The headline is the cherry-pick rate: how often the synthesised answer simply reproduces
one branch (typically the most self-confident one) instead of combining what the
branches collectively found.
"""
import json
import sys
from collections import Counter

from app.config import TRACE_PATH
from app.services.dst import normalize_label


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _branch_diagnoses(row):
    """Each branch's diagnosis and self-reported confidence, normalized."""
    out = {}
    for branch, summary in (row.get("branches") or {}).items():
        if summary.get("status") != "ok":
            continue
        label = normalize_label(summary.get("probable_diagnosis"))
        if label:
            out[branch] = (label, summary.get("self_reported_confidence") or 0.0)
    return out


def _finding_type(finding):
    """Bucket a finding by kind. The audit records findings as prose carrying the actual
    numbers, so they have to be classified rather than counted verbatim — otherwise every
    turn becomes its own bucket."""
    if "conflict" in finding:
        return "sources conflict"
    if "exceeds plausibility" in finding:
        return "overconfident vs evidence"
    if "supports" in finding:
        return "diagnosis unsupported by retrieval"
    return finding


def _pct(n, total):
    return f"{n:>4}/{total}  ({100 * n / total:5.1f}%)" if total else "   n/a"


def main(path):
    rows = load(path)
    clinical = [r for r in rows if (r.get("dst") or {}).get("frame")]

    if not clinical:
        print(f"No clinical turns with a fusion frame in {path} ({len(rows)} rows total).")
        return

    total = len(clinical)
    agree = cherry_picked = unsupported = conflicted = would_change = 0
    ignorances = []
    findings = Counter()

    for row in clinical:
        dst_row = row["dst"]
        orch = normalize_label((row.get("orchestrator") or {}).get("probable_diagnosis"))
        top = normalize_label(dst_row.get("top_hypothesis"))
        branches = _branch_diagnoses(row)

        if orch and orch == top:
            agree += 1

        # Cherry-picking: the answer reproduces the single most self-confident branch,
        # and the pooled evidence points elsewhere. Reproducing a branch that the fusion
        # ALSO favours is agreement, not cherry-picking — the distinction matters.
        if orch and branches and orch != top:
            most_confident = max(branches.items(), key=lambda kv: kv[1][1])
            if orch == most_confident[1][0]:
                cherry_picked += 1

        ignorances.append(dst_row.get("ignorance") or 0.0)
        if (dst_row.get("conflict") or 0.0) >= 0.6:
            conflicted += 1

        audit = row.get("dst_audit") or {}
        for finding in audit.get("findings") or []:
            findings[_finding_type(finding)] += 1
        if any("no branch's retrieval supports" in f for f in audit.get("findings") or []):
            unsupported += 1
        if audit.get("level_with_dst") != audit.get("level_without_dst"):
            would_change += 1

    print(f"\nTurns analysed: {total}  (of {len(rows)} traced)\n")
    print("  Orchestrator agrees with fusion   ", _pct(agree, total))
    print("  Cherry-picked one branch          ", _pct(cherry_picked, total))
    print("  Diagnosis unsupported by any branch", _pct(unsupported, total))
    print("  Sources in material conflict      ", _pct(conflicted, total))
    print("  Triage level DST would change     ", _pct(would_change, total))
    print(f"\n  Mean ignorance m(Theta):           {sum(ignorances) / total:.3f}")

    if findings:
        print("\n  Findings by type:")
        for name, count in findings.most_common():
            print(f"    {count:>4}  {name}")

    enforced = sum(1 for r in clinical if (r.get("dst_audit") or {}).get("enforced"))
    print(f"\n  Recorded under enforcement: {enforced}/{total}"
          f"{'  (log-only — DST_ENFORCE is false)' if not enforced else ''}\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else TRACE_PATH)
