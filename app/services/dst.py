"""Dempster-Shafer evidence fusion over the retrieval branches.

The orchestrator asks a language model to combine what RAG, KGRAG and MCP found. This
module computes the same fusion as an explicit calculus, so the two can be compared —
and so the system has a quantity to act on that no model authored.

Why belief functions rather than a probability distribution: a Bayesian posterior
cannot distinguish "no evidence" from "evidence for a uniform distribution". Both come
out flat. That distinction is exactly the failure this system had to fix — every branch
retrieving nothing, and a confident answer being produced anyway — so the formalism has
to represent ignorance as a first-class state. Dempster-Shafer does: unassigned mass
sits on the frame Θ itself.

Two properties fall out that the code would otherwise have to special-case:

- A branch that retrieved nothing gets the vacuous mass function m(Θ)=1, which is the
  identity element of Dempster's rule. A dead MCP server contributes mathematically
  nothing instead of being dropped by a hand-written guard.
- Combining vacuous with vacuous stays vacuous. When no branch found anything, belief
  in every hypothesis is zero by construction. The fusion cannot manufacture belief
  from nothing — that is a property of the operator, not a check someone remembered.

Masses are built from RETRIEVAL signal only — similarity scores, match quality, subtype
counts, source authority. The `confidence` float each branch reports is written by a
language model about its own work and is deliberately never used here; building a
rigorous calculus on self-reported numbers would be decoration.
"""
import re
from functools import lru_cache

from app.config import (
    DST_BM25_SATURATION,
    DST_CONFLICT_ESCALATE,
    DST_COVERAGE_FLOOR,
    DST_DISCOUNT,
    DST_IGNORANCE_RETRY,
    DST_LABEL_MATCH_MIN_OVERLAP,
    DST_MCP_TIER,
    DST_MCP_TIER_DEFAULT,
    DST_MCP_VOLUME_SCALE,
    DST_RAG_SATURATION,
    DST_SPECIFICITY_SCALE,
    DST_UNRULED_MIN_PLAUSIBILITY,
    RETRIEVAL_SCORE_THRESHOLD,
)
from app.services import snomed_store

# Total conflict (K=1) makes Dempster's rule undefined — the normalisation divides by
# 1-K. Guarded rather than allowed to raise: total conflict is a meaningful state that
# the control policy must see, not a crash.
_EPSILON = 1e-9

# Qualifiers stripped before two diagnosis strings are compared. These are epistemic or
# temporal hedges, not different diseases: "acute viral pharyngitis" and "viral
# pharyngitis" are one hypothesis. Deliberately conservative — "streptococcal
# pharyngitis" and "pharyngitis" stay distinct, because collapsing a specific disorder
# into its parent would silently merge hypotheses that the whole system exists to tell
# apart.
_QUALIFIERS = {
    "acute", "subacute", "chronic", "suspected", "probable", "possible",
    "likely", "apparent", "presumed", "recurrent",
}

# Strings a branch uses to say it found nothing. These must never become hypotheses.
_NULL_LABELS = ("insufficient evidence", "insufficient detail", "clarification needed",
                "unable to process")


def _normalize_text(raw) -> str:
    """The string half of label normalisation: casing, punctuation, hedges."""
    text = str(raw or "").strip().lower()
    if not text:
        return ""

    # SNOMED FSNs carry a semantic tag: "Diphtheria (disorder)". KGRAG additionally
    # appends the concept id — "Fever due to infection (finding) [722892007]" — and the
    # id must go before anything else: left in, it survives as a bare numeric token that
    # matches no description, which blocks canonicalisation for every KGRAG label.
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if any(null in text for null in _NULL_LABELS):
        return ""

    words = [w for w in text.split() if w not in _QUALIFIERS]
    return " ".join(words)


@lru_cache(maxsize=1024)
def _canonicalize(text: str) -> str:
    """Rewrite a label to its SNOMED concept's fully specified name, when one resolves.

    This is what makes the frame a set of *hypotheses* rather than a set of *strings*.
    RAG writes "COVID-19", KGRAG writes "Fever caused by SARS-CoV-2" — the same disease,
    and left as two labels they are two disjoint singletons. Dempster's rule then reads
    two branches that agree as two branches with no possible world in common, which
    inflates conflict K and splits belief that should have been pooled.

    Degrades to the plain string when SNOMED is unavailable or nothing matches well
    enough: this module must not require the graph to function.
    """
    if not text:
        return ""

    try:
        matches = snomed_store.find_concepts([text], limit=1, per_term=1)
    except Exception:
        # Includes SnomedUnavailable. No graph, no canonicalisation — the string
        # heuristic alone is still a correct, if coarser, comparison.
        return text

    if not matches:
        return text

    # Guard against canonicalising onto whatever bm25 ranked first. The matched
    # DESCRIPTION is what to check, not the FSN: "COVID-19" is a synonym description of
    # a concept whose FSN shares none of its words.
    words = set(text.split())
    matched = set(_normalize_text(matches[0].get("matched_term")).split())
    if not words or len(words & matched) / len(words) < DST_LABEL_MATCH_MIN_OVERLAP:
        return text

    return _normalize_text(matches[0].get("fsn")) or text


def normalize_label(raw) -> str:
    """Reduce a free-text diagnosis to a comparable hypothesis label.

    Returns "" for anything that is not a hypothesis — a branch's way of reporting that
    it found nothing must not enter the frame as a candidate diagnosis.
    """
    return _canonicalize(_normalize_text(raw))


def _hypotheses(output: dict) -> tuple[str, list[str]]:
    """A branch's top hypothesis and its full candidate set, normalized and deduped."""
    top = normalize_label(output.get("probable_diagnosis"))

    candidates = []
    for label in ([top] if top else []) + list(output.get("differentials") or []):
        norm = normalize_label(label)
        if norm and norm not in candidates:
            candidates.append(norm)

    return top, candidates


# --- Retrieval strength (s_b): how much belief the evidence licenses at all ---

def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _strength_rag(signal: dict) -> float:
    scores = signal.get("scores") or []
    if not scores:
        return 0.0

    # Everything here already cleared RETRIEVAL_SCORE_THRESHOLD, so that threshold is
    # the natural zero point: a chunk sitting exactly at it carries no evidence.
    span = DST_RAG_SATURATION - RETRIEVAL_SCORE_THRESHOLD
    mean_score = sum(scores) / len(scores)
    strength = _clip((mean_score - RETRIEVAL_SCORE_THRESHOLD) / span) if span > 0 else 0.0

    n = len(scores)
    coverage = DST_COVERAGE_FLOOR + (1 - DST_COVERAGE_FLOOR) * (n / (n + 1))
    return strength * coverage


def _strength_kgrag(signal: dict) -> float:
    concepts = signal.get("concepts") or []
    if not concepts:
        return 0.0

    best = 0.0
    for concept in concepts:
        rank = concept.get("rank")
        # bm25 is negative and more negative is better; a missing rank means the match
        # quality is unknown, which is not the same as good.
        quality = _clip(-float(rank) / DST_BM25_SATURATION) if rank is not None else 0.0

        # A concept with a large subtype set is a general finding. The ontology has
        # located the patient but cannot license belief in any specific disorder.
        n_subtypes = int(concept.get("n_subtypes") or 0)
        specificity = 1.0 / (1.0 + n_subtypes / DST_SPECIFICITY_SCALE)

        best = max(best, quality * specificity)

    return best


def _strength_mcp(signal: dict) -> float:
    chars = float(signal.get("evidence_chars") or 0)
    if chars <= 0:
        return 0.0

    tool = str(signal.get("tool") or "").lower()
    tier = next(
        (weight for key, weight in DST_MCP_TIER.items() if key in tool),
        DST_MCP_TIER_DEFAULT,
    )

    volume = DST_COVERAGE_FLOOR + (1 - DST_COVERAGE_FLOOR) * (
        chars / (chars + DST_MCP_VOLUME_SCALE)
    )
    return tier * volume


_STRENGTH = {"rag": _strength_rag, "kgrag": _strength_kgrag, "mcp": _strength_mcp}


# --- Mass construction ---

def build_mass(branch: str, output: dict, frame: frozenset) -> dict[frozenset, float]:
    """Build one branch's mass function over the frame.

    Three focal elements, uniformly across branches:

        m({d})  = s · α         committed to its top diagnosis
        m(D)    = s · (1 - α)   D = the branch's full candidate set
        m(Θ)    = 1 - s         ignorance

    where s is retrieval strength and α = 1/(1 + |differentials|) is discrimination.

    That single α formula is doing real work, and it is worth seeing why it is not
    arbitrary: RAG naming a diagnosis with no differentials gets α≈1 and its mass lands
    on a singleton, while KGRAG after an IS_A walk returning a dozen subtypes gets
    α≈0.08 and its mass lands almost entirely on the SET. Both are correct, and neither
    was hand-tuned — an ontology walk genuinely reports the candidate set rather than
    the answer, and this is the representation that can say so. A flat probability
    vector cannot express "one of these twelve, I cannot tell which".
    """
    vacuous = {frame: 1.0}

    if not frame:
        return {}

    # An errored branch is ignorance, not evidence. Vacuous is the identity element of
    # Dempster's rule, so this contributes nothing rather than needing to be dropped.
    if "error" in output or output.get("status") == "not_implemented":
        return vacuous

    signal = output.get("retrieval_signal")
    if not signal:
        return vacuous

    strength = _clip(_STRENGTH[branch](signal))
    if strength <= 0.0:
        return vacuous

    top, candidates = _hypotheses(output)
    if not top:
        # Retrieval succeeded but the branch named no hypothesis. It found something and
        # committed to nothing, which is precisely ignorance.
        return vacuous

    # Discrimination from the branch's own candidate structure, not from any number a
    # model reported about its own certainty.
    n_differentials = max(0, len(candidates) - 1)
    alpha = 1.0 / (1.0 + n_differentials)

    singleton = frozenset([top])
    candidate_set = frozenset(candidates)

    mass: dict[frozenset, float] = {}
    mass[singleton] = strength * alpha
    if candidate_set != singleton:
        mass[candidate_set] = strength * (1.0 - alpha)
    else:
        # Nothing to spread over — a lone hypothesis takes the whole committed mass.
        mass[singleton] = strength

    mass[frame] = mass.get(frame, 0.0) + (1.0 - strength)
    return {k: v for k, v in mass.items() if v > 0.0}


def discount(mass: dict[frozenset, float], beta: float, frame: frozenset) -> dict[frozenset, float]:
    """Shafer's discount operator: trust this source a fraction β, move the rest to Θ.

    This is the standard concession to the fact that Dempster's rule assumes independent
    sources and these are not — one model synthesizes all three branches, and the RAG
    corpus and MCP's literature overlap. Discounting does not make them independent; it
    limits how much any one of them can drive the result.
    """
    if beta >= 1.0:
        return dict(mass)

    discounted = {focal: value * beta for focal, value in mass.items() if focal != frame}
    discounted[frame] = mass.get(frame, 0.0) * beta + (1.0 - beta)
    return {k: v for k, v in discounted.items() if v > 0.0}


# --- Combination ---

def combine(m1: dict[frozenset, float], m2: dict[frozenset, float]) -> tuple[dict, float]:
    """Dempster's rule of combination. Returns the combined mass and the conflict K.

    K is the mass that lands on the empty set — evidence the two sources cannot both be
    right about. It is not an error term to be normalised away and forgotten; it is the
    signal that the sources disagree, and the control policy escalates on it.
    """
    raw: dict[frozenset, float] = {}
    conflict = 0.0

    for focal1, value1 in m1.items():
        for focal2, value2 in m2.items():
            product = value1 * value2
            intersection = focal1 & focal2
            if not intersection:
                conflict += product
            else:
                raw[intersection] = raw.get(intersection, 0.0) + product

    # Total conflict: the sources share no possible world. Normalisation is undefined,
    # and averaging them into a consensus would invent one. Report it instead.
    if conflict >= 1.0 - _EPSILON:
        return {}, 1.0

    normalizer = 1.0 - conflict
    return {focal: value / normalizer for focal, value in raw.items()}, conflict


def belief(mass: dict[frozenset, float], hypothesis: frozenset) -> float:
    """Total mass committed to this hypothesis or something that entails it."""
    return sum(v for focal, v in mass.items() if focal <= hypothesis)


def plausibility(mass: dict[frozenset, float], hypothesis: frozenset) -> float:
    """Total mass not ruled out — the upper bound of the belief interval."""
    return sum(v for focal, v in mass.items() if focal & hypothesis)


# --- Top level ---

def fuse(branch_outputs: dict[str, dict]) -> dict:
    """Fuse the retrieval branches into a belief state and a control decision.

    `branch_outputs` maps branch name to that branch's output dict. Returns the frame,
    the per-branch and combined masses, the conflict and ignorance, a belief interval
    per hypothesis, and the action the (conflict, ignorance) pair implies.
    """
    labels: list[str] = []
    for output in branch_outputs.values():
        if not isinstance(output, dict) or "error" in output:
            continue
        top, candidates = _hypotheses(output)
        # A branch that named no top hypothesis gets a vacuous mass in build_mass, so
        # it commits nothing. It must not enlarge the frame either: its differentials
        # would then surface in unruled_out as open questions no source ever raised —
        # which is how a KGRAG walk that reported "Insufficient evidence" still put
        # Piry virus disease in front of the guardian.
        if not top:
            continue
        for label in candidates:
            if label not in labels:
                labels.append(label)

    frame = frozenset(labels)

    # No branch named a single hypothesis. The frame is empty and there is nothing to
    # hold belief — total ignorance, and the honest answer is that retrieval failed.
    if not frame:
        return {
            "frame": [],
            "masses": {},
            "conflict": 0.0,
            "ignorance": 1.0,
            "hypotheses": [],
            "top_hypothesis": None,
            "action": "retry",
            "reason": "No branch produced a hypothesis — total ignorance.",
        }

    masses = {
        branch: discount(
            build_mass(branch, output if isinstance(output, dict) else {}, frame),
            DST_DISCOUNT.get(branch, 1.0),
            frame,
        )
        for branch, output in branch_outputs.items()
    }

    combined: dict[frozenset, float] = {frame: 1.0}
    conflict = 0.0
    for mass in masses.values():
        if not mass:
            continue
        combined, pairwise_conflict = combine(combined, mass)
        # Conflict accumulates across pairwise combinations; the running maximum is the
        # sharpest disagreement encountered, which is what should drive escalation.
        conflict = max(conflict, pairwise_conflict)
        if not combined:
            break

    ignorance = combined.get(frame, 0.0) if combined else 1.0

    hypotheses = sorted(
        (
            {
                "hypothesis": label,
                "belief": belief(combined, frozenset([label])),
                "plausibility": plausibility(combined, frozenset([label])),
            }
            for label in labels
        ),
        key=lambda h: h["belief"],
        reverse=True,
    )

    action, reason = _decide(conflict, ignorance)

    return {
        "unruled_out": unruled_out(hypotheses),
        "frame": labels,
        "masses": {b: _serialize(m) for b, m in masses.items()},
        "combined": _serialize(combined),
        "conflict": conflict,
        "ignorance": ignorance,
        "hypotheses": hypotheses,
        "top_hypothesis": hypotheses[0]["hypothesis"] if hypotheses else None,
        "action": action,
        "reason": reason,
    }


def unruled_out(hypotheses: list[dict]) -> list[str]:
    """Hypotheses nothing supports and nothing excludes — what has not been ruled out.

    Zero belief with substantial plausibility is a precise statement: no source produced
    evidence for this, and no source produced evidence against it either. That is the
    definition of an open question, and it is where safety-netting advice ("come back
    if...") should come from — derived from what the evidence left unexamined, rather
    than invented as prose by a language model.
    """
    return [
        h["hypothesis"]
        for h in hypotheses
        if h["belief"] <= 0.0 and h["plausibility"] >= DST_UNRULED_MIN_PLAUSIBILITY
    ]


def _decide(conflict: float, ignorance: float) -> tuple[str, str]:
    """Map the (conflict, ignorance) pair onto a control action.

    These are two different kinds of uncertainty and they demand opposite responses,
    which is the thing a single confidence scalar cannot express:

    - Conflict: the sources actively disagree. Retrieving more evidence on a contested
      question entrenches rather than resolves it, so this escalates to a physician.
    - Ignorance: the sources do not disagree, they simply have not committed. Nobody
      knows yet. That is what re-retrieval actually fixes.
    """
    if conflict >= DST_CONFLICT_ESCALATE:
        return "escalate", f"Sources conflict (K={conflict:.2f}) — more retrieval will not resolve a contested question."
    if ignorance >= DST_IGNORANCE_RETRY:
        return "retry", f"Unresolved ignorance (m(Θ)={ignorance:.2f}) — no source committed."
    return "accept", f"Low conflict (K={conflict:.2f}) and low ignorance (m(Θ)={ignorance:.2f})."


def most_ignorant_branch(branch_outputs: dict[str, dict]) -> str | None:
    """Which branch to re-run, by leave-one-out ablation of the fusion.

    The intuition here inverts, and getting it backwards is easy: removing an ignorant
    branch does NOT reduce fused ignorance. A vacuous mass function is the identity
    element of Dempster's rule, so dropping it changes the result by exactly nothing.
    It is removing an *informative* branch that drives ignorance up.

    So a branch's informativeness is how much ignorance rises without it, and the one
    worth re-running is the one that contributes least — the branch whose absence the
    fusion does not notice.

    This replaces asking a model to write a free-text hint about which branch
    underperformed: the target is derived from the fusion rather than guessed. Ties go
    to the first branch in iteration order.
    """
    if len(branch_outputs) < 2:
        return None

    baseline = fuse(branch_outputs)["ignorance"]

    least_informative, lowest = None, None
    for branch in branch_outputs:
        remaining = {b: o for b, o in branch_outputs.items() if b != branch}
        # >= 0: removing evidence can only leave more mass on Θ.
        contribution = fuse(remaining)["ignorance"] - baseline
        if lowest is None or contribution < lowest:
            least_informative, lowest = branch, contribution

    return least_informative


def _serialize(mass: dict[frozenset, float]) -> dict[str, float]:
    """Focal sets as sorted pipe-joined labels, so the state stays JSON-serializable."""
    return {" | ".join(sorted(focal)): value for focal, value in mass.items()}
