"""Patient-safe explanation of how a turn was decided.

The patient sees a picture of the REASONING, never its contents. That distinction is
the whole design: how many independent sources were consulted, whether they agreed,
how much was left unknown, and what that produced — with no disease named, no branch
identified, and no belief number shown.

Three reasons the whitelist lives here rather than in the client:

- Everything this module emits is derived from `dst.py`, which computes the fusion
  independently of any language model. So the picture explains an actual computation
  rather than narrating a model's account of itself.
- The disclosure rule becomes a property of the architecture instead of a convention
  the UI is trusted to follow. A component cannot render a differential it was never
  sent.
- The raw quantities are genuinely unsafe in isolation. Dempster-Shafer belief is not
  a probability — bel=0.03 with pl=1.0 means "almost nothing established, almost
  nothing excluded", and a patient reads 0.03 as "3% chance". Bands carry the meaning
  the numbers actually have.
"""
import re

from app.config import DST_CONFLICT_ESCALATE, DST_IGNORANCE_RETRY
from app.services.dst import normalize_label

# How many candidates the graph shows. The ontology walk can return dozens; past this
# the picture stops being legible and starts being a wall. The rest are counted, not
# drawn.
_MAX_CANDIDATES = 14

_SEMANTIC_TAG = re.compile(r"\s*\([^)]*\)\s*$")

# Branch order is fixed but the names never ship. The client receives an anonymous
# list — "three sources were consulted, two reported" — because which source is which
# is exactly the internal detail patients must not see.
_BRANCHES = ("rag_output", "kgrag_output", "mcp_output")

# Plain-language outcome reasons. Written here, never by a model: the guardian's own
# `reasoning` field is LLM prose that names soft rules and internal levels.
_OUTCOME_REASONS = {
    "emergency": "Your symptoms need urgent attention.",
    "locked": "This needs a clinician to review before any advice is acted on.",
    "clinician_review": "A clinician checks this before you act on it.",
    "direct": "This was straightforward enough to answer directly.",
}


def _source_status(output) -> str:
    """How a single retrieval source fared, in the three states a patient can act on."""
    if not isinstance(output, dict) or "error" in output:
        # Timed out, unreachable, or never built. Distinguished from "looked and found
        # nothing" because they mean different things about the answer's support.
        return "unavailable"

    if not str(output.get("probable_diagnosis") or "").strip():
        return "no_findings"

    # A source that ran, searched, and committed to nothing is not the same as one that
    # failed — and DST already treats the two identically as vacuous mass, so the
    # distinction has to be made here to survive to the patient.
    if float(output.get("confidence") or 0.0) <= 0.0:
        return "no_findings"

    return "reported"


def _band(value: float, threshold: float) -> str:
    """Three bands around the threshold the control policy already acts on.

    The midpoint is not a tuned constant: the policy flips at `threshold`, so the band
    below it is "approaching the point where the system would stop and ask".
    """
    if value >= threshold:
        return "low"
    if value >= threshold / 2:
        return "moderate"
    return "high"


def _display(fsn: str) -> str:
    """A concept name as a person would say it — SNOMED's semantic tag stripped."""
    return _SEMANTIC_TAG.sub("", str(fsn or "")).strip()


def _differential_graph(state: dict) -> dict | None:
    """The ontology walk, annotated with what the fusion made of each candidate.

    This is the one place the two halves of the system meet. KGRAG knows the SHAPE —
    which disorders sit beneath the finding the patient described. DST knows the
    WEIGHT — which of them any evidence actually supports. Joining them needs a shared
    identity for a disease, which is exactly what canonicalising labels through SNOMED
    bought: a DST hypothesis and an ontology node now reduce to the same string.

    A node is one of:
      finding    — what the patient's own words matched
      settled    — the hypothesis the fusion landed on
      considered — entered the frame; evidence bore on it
      ruled_out  — the ontology offered it and nothing supported it

    The ruled-out nodes are the point. A differential the system examined and dropped
    is what makes the remaining one meaningful, and showing the work is the difference
    between explaining a decision and asserting one.
    """
    kgrag = state.get("kgrag_output")
    if not isinstance(kgrag, dict) or "error" in kgrag:
        return None

    graph = kgrag.get("graph") or {}
    raw_nodes = graph.get("nodes") or []
    if not raw_nodes:
        return None

    dst = state.get("dst_output") or {}
    # The ontology finishes before the fusion does. In that window every candidate is
    # genuinely still live — none has been weighed — so they are reported as
    # "considering" rather than pre-judged. This is the one honest way to animate the
    # narrowing: the candidates really are undecided at this point in the turn.
    resolved = bool(dst)
    top = dst.get("top_hypothesis")
    # Belief, not frame membership, decides whether a candidate survives. A hypothesis
    # can sit in the frame and carry zero belief — that is precisely the state "nothing
    # supported this", and it is what a strike should mean.
    beliefs = {
        h.get("hypothesis"): float(h.get("belief") or 0.0)
        for h in (dst.get("hypotheses") or [])
    }

    findings, candidates = [], []
    # Canonical labels already seen. Qualifier stripping is deliberately aggressive —
    # "acute sore throat" and "chronic sore throat" both reduce to "sore throat", which
    # is correct for the fusion (one hypothesis) but wrong to draw twice. Worse, both
    # would match the top hypothesis and the picture would show two winners.
    seen: set[str] = set()
    settled_taken = False

    for node in raw_nodes:
        label = normalize_label(node.get("label"))
        entry = {"id": node.get("id"), "label": _display(node.get("label")), "status": "finding"}

        if node.get("depth") == 0:
            findings.append(entry)
            continue

        if label and label in seen:
            continue
        if label:
            seen.add(label)

        if not resolved:
            entry["status"] = "considering"
        elif label and label == top and not settled_taken:
            entry["status"] = "settled"
            settled_taken = True
        elif beliefs.get(label, 0.0) > 0.0:
            entry["status"] = "considered"
        else:
            entry["status"] = "ruled_out"

        candidates.append(entry)

    # The ontology walk is not the only source of hypotheses — the guideline and live
    # branches name diagnoses too, and the fusion frequently lands on one of those. Left
    # out, the spine shows a dozen struck candidates and no survivor, which contradicts
    # the answer beside it. So any hypothesis the fusion held that the walk did not
    # produce is added here: the spine is what was WEIGHED, not merely what SNOMED
    # offered. Ids are synthetic and negative, since these have no concept behind them.
    if resolved:
        for offset, hypothesis in enumerate(h.get("hypothesis") for h in (dst.get("hypotheses") or [])):
            if not hypothesis or hypothesis in seen:
                continue
            seen.add(hypothesis)

            if hypothesis == top and not settled_taken:
                status = "settled"
                settled_taken = True
            else:
                status = "considered" if beliefs.get(hypothesis, 0.0) > 0.0 else "ruled_out"

            candidates.append(
                {"id": -(offset + 1), "label": _display(hypothesis).capitalize(), "status": status}
            )

    if not findings or not candidates:
        return None

    # Survivors first, so truncation never drops the node that matters. Before the
    # fusion runs nothing is ranked yet, and every candidate shares a rank — which
    # preserves the ontology's own order until there is a reason to change it.
    rank = {"considering": 0, "settled": 0, "considered": 1, "ruled_out": 2}
    candidates.sort(key=lambda c: rank[c["status"]])
    hidden = max(0, len(candidates) - _MAX_CANDIDATES)
    candidates = candidates[:_MAX_CANDIDATES]

    kept = {c["id"] for c in candidates} | {f["id"] for f in findings}
    edges = [
        e
        for e in (graph.get("edges") or [])
        if e.get("from") in kept and e.get("to") in kept
    ]
    # Candidate-to-candidate links, kept only between nodes that survived truncation.
    links = [
        e
        for e in (graph.get("links") or [])
        if e.get("from") in kept and e.get("to") in kept
    ]

    return {
        "findings": findings,
        "candidates": candidates,
        "edges": edges,
        "links": links,
        "hidden": hidden,
        "ruled_out": sum(1 for c in candidates if c["status"] == "ruled_out") + hidden,
    }


def build_explanation(state: dict) -> dict:
    """Assemble the patient-safe view. Returns only whitelisted, non-clinical fields."""
    dst = state.get("dst_output") or {}
    guardian = state.get("guardian_output") or {}

    statuses = [_source_status(state.get(branch)) for branch in _BRANCHES]

    # Conflict is disagreement between sources; ignorance is nobody having committed.
    # They are different failures and the patient is shown both, because "they
    # disagreed" and "nobody knew" warrant different reactions from them.
    conflict = float(dst.get("conflict") or 0.0)
    ignorance = float(dst.get("ignorance") or 0.0)

    level = str(guardian.get("triage_level") or "level_2")
    if state.get("is_emergency"):
        reason_key = "emergency"
    elif level == "level_3":
        reason_key = "locked"
    elif level == "level_2":
        reason_key = "clinician_review"
    else:
        reason_key = "direct"

    return {
        "sources": [{"status": status} for status in statuses],
        "sources_reported": sum(1 for s in statuses if s == "reported"),
        # Agreement inverts conflict so both bars read "more is better" — a patient
        # should not have to work out that a long bar is bad.
        "agreement": {
            "band": _band(conflict, DST_CONFLICT_ESCALATE),
            "value": round(1.0 - min(1.0, conflict), 3),
        },
        "certainty": {
            "band": _band(ignorance, DST_IGNORANCE_RETRY),
            "value": round(1.0 - min(1.0, ignorance), 3),
        },
        "outcome": {
            "triage_level": int(level.replace("level_", "") or 2),
            "requires_doctor": bool(guardian.get("requires_doctor")),
            "reason": _OUTCOME_REASONS[reason_key],
        },
        # None when the ontology branch produced nothing — the interface then has no
        # differential to draw, which is the honest state rather than an empty frame.
        "differential": _differential_graph(state),
    }
