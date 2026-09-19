"""Applying the directional check to a published steering result.

Everything else in this study measures concepts we built. That leaves the
obvious question unanswered: does the check change what anybody already
believes? This module replicates one published intervention and re-scores it.

The target is activation addition \\citep{turner2023actadd} and its canonical
demonstration, in which adding a contrast between a wedding prompt and a blank
one makes GPT-2-XL talk about weddings. It is the right target for three
reasons. The intervention is the same shape as ours, a difference between
contrastive activations added at a layer and scaled by a coefficient. The model
is small enough to sweep properly on a free-tier GPU, and its layer layout
(`transformer.h`) is already supported. And the behavior has an unambiguous
rule-based readout -- the fraction of words that are wedding vocabulary -- so
the published effect can be re-scored with no judge at all, using exactly the
machinery validated in `groundtruth.py`.

**What this is and is not.** We do not claim to reproduce their reported
numbers: the sweep, the prompt set and the scoring are ours, and their published
demonstration is a single coefficient rather than a dose-response. The claim is
narrower and is the one that matters here: taking the same intervention on the
same model and behavior, what do the absolute and directional summaries each
say about it, and do they agree?

**Preregistered, before the run.** Both outcomes are reported, in the same
words, whichever occurs:

  - If the signed area resolves positive, the published effect is directional
    and survives the check. That is a case where the check passes, and it shows
    the check is applicable rather than merely a complaint.
  - If the signed area does not resolve, or resolves negative, then a published
    steering result does not survive a directional reading, and the absolute
    summary was carrying it.

  - If the behavior never occurs at any coefficient, including theirs, that is
    not either of the above and is not reported as a directional finding. A
    readout that is constant at zero says the sweep never reproduced the
    effect, and the first thing that implicates is our harness. The first
    attempt returned exactly this, and it was the decoder and the coefficient
    scale, both recorded in the run log. Such a run is a failed replication of
    our own making until the positive control below passes.

**A positive control that gates the null.** Before any null is reported, the
sweep must reproduce the published effect somewhere in its range: the readout
has to rise above baseline at some coefficient by more than its interval. If
it never does, the instrument is not sensitive enough to license a statement
about direction, and the outcome is "not replicated here" rather than either
preregistered branch. This is the same discipline as P9 for the judge-scored
concepts, and it exists because a flat zero is the one result that looks like
a strong finding while actually being the absence of one.

Neither outcome is a better result for us. Committing to that here, in the
module the experiment runs from, is the point of writing it down before the
sweep rather than after.

**One thing to verify against the paper before running.** The exact published
setting -- which layer, which coefficient, the precise contrast strings -- is
recorded in `ACTADD_SETTING` below from our reading of the paper, and should be
checked against the source. Our sweep covers a coefficient range around it
rather than relying on a single point, so a small discrepancy changes the
centre of the sweep and not the conclusion, but the recorded setting should be
right.
"""

from __future__ import annotations

import re

from .concepts import Concept, Pair

# Recorded from the paper and to be verified against it. The sweep brackets this
# rather than sitting on it.
ACTADD_SETTING = {
    "model": "gpt2-xl",
    "positive_prompt": "Weddings",
    "negative_prompt": " ",
    "layer": 6,
    "coefficient": 1.0,
    "source": "turner2023actadd",
}

# Decoding. The study generates greedily everywhere else, on purpose: an effect
# visible only under sampling noise is not an effect. That default cannot be
# used here. The published demonstration is a sampling result on a base model,
# and greedy gpt2-xl falls into a repetition loop, so a coefficient near theirs
# almost never moves the argmax path. The first attempt at this stage swept the
# whole grid greedily and the readout returned 0.0 for all 130 generations,
# including at their own coefficient: a fact about the decoder, not the method.
#
# Recorded from our reading of the paper and to be verified against it, like
# ACTADD_SETTING. `n_samples` is ours, not theirs, and only sets how many draws
# the readout averages over.
ACTADD_DECODING = {
    "temperature": 1.0,
    "n_samples": 5,
    "source": "turner2023actadd",
}

# In multiples of the published coefficient, which is what makes 1.0 below
# their exact setting rather than a guess.
#
# This needs `SteeringSpec(unit_mode="raw_norm", raw_scale=<norm of the raw
# activation difference>)`. The study's own coefficient is in residual-RMS
# units on a unit-normalised direction, and the published one multiplies the
# raw difference; the two differ by whatever ratio those scales stand in, which
# for this contrast at this layer is a factor of about 175. Sweeping +-20 in
# RMS units, as the first attempt did, is not a wide bracket around their
# setting but a sweep that is not shown to contain it at all. Scaling by the
# raw norm removes the conversion instead of guessing it.
PUBLISHED_COEFF_MULTIPLES = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
                             0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

# Kept for the superseded RMS-unit sweep, whose null is recorded in the run log
# as an artifact rather than a result.
PUBLISHED_COEFFS = [-20.0, -15.0, -10.0, -6.0, -3.0, -1.0, 0.0,
                    1.0, 3.0, 6.0, 10.0, 15.0, 20.0]

_WORD = re.compile(r"[A-Za-z']+")

# Deliberately conservative. "ring", "party", "dress" and "cake" are wedding
# vocabulary in context and ordinary words out of it, and a readout that fires
# on them would report a wedding effect wherever the model wrote about jewellery
# or birthdays. Precision matters more than recall here: the readout has to be
# wrong in the direction that understates the published effect, not overstates
# it, or the replication flatters the thing it is checking.
_WEDDING = {
    "wedding", "weddings", "bride", "brides", "bridal", "bridesmaid",
    "bridesmaids", "groom", "grooms", "groomsmen", "marry", "marrying",
    "married", "marriage", "marriages", "honeymoon", "honeymoons",
    "bouquet", "bouquets", "vows", "altar", "betrothed", "nuptial",
    "nuptials", "fiance", "fiancee", "elope", "eloped", "elopement",
    "matrimony", "matrimonial", "wedded",
}


def score_wedding(text: str) -> float:
    """Fraction of words that are wedding vocabulary."""
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return 0.0
    return sum(w in _WEDDING for w in words) / len(words)


# Exposed for DeterministicScorer(extra=...) rather than registered into the
# shared READOUTS on import: a module that mutates a global registry changes
# what another scorer accepts depending on import order.
PUBLISHED_READOUTS = {"pub_wedding": score_wedding}

# ActAdd derives its direction from one contrast, not a corpus. We keep that:
# using our usual many-pair estimate would be testing our method on their
# behavior rather than testing their intervention. The pair is repeated across
# template families only so the harness's held-out-family machinery has
# something to split on; every pair carries the same contrast.
_FAMILIES = ["a", "b", "c", "d", "e", "f"]

# Their demonstration prompt, plus neutral continuations in the same register so
# the sweep is not a single point. None mentions weddings, marriage or
# relationships: a prompt that invites the behavior would make the intervention
# look effective when the prompt was doing the work.
_EVAL_PROMPTS = [
    "I went up to my friend and said",
    "Yesterday afternoon I decided to",
    "The thing I keep meaning to tell you is",
    "When I got home I immediately",
    "My plan for the weekend is to",
    "The first thing I noticed was",
    "I spent most of the morning",
    "What happened next was that I",
    "Later that day someone asked me",
    "I remember thinking that I should",
]


def actadd_wedding() -> Concept:
    """The published intervention, as a concept this harness can sweep."""
    pos, neg = ACTADD_SETTING["positive_prompt"], ACTADD_SETTING["negative_prompt"]
    return Concept(
        name="pub_wedding",
        description=(
            "Replication of the activation-addition wedding demonstration, "
            "scored by a rule rather than a judge."
        ),
        pairs=[Pair(positive=pos, negative=neg, family=f) for f in _FAMILIES],
        eval_prompts=list(_EVAL_PROMPTS),
        behavior_question="Does the response talk about weddings?",
        safety_relevant=False,
        surface_confounded=True,
        notes=(
            "Readout: fraction of words in a conservative wedding vocabulary. "
            "Direction from a single published contrast pair, repeated across "
            "families so the split has something to work with."
        ),
    )


def published_concepts() -> list[Concept]:
    return [actadd_wedding()]


def actadd_direction(lm, layer: int | None = None,
                     positive: str | None = None,
                     negative: str | None = None):
    """The published contrast as a position-wise matrix.

    Returns `(rows, norms, width)` where `rows` is (P, d_model) with every row
    unit-norm, `norms` is the per-position norm of the raw difference, and
    `width` is P. Feed them to `SteeringSpec(direction=rows,
    unit_mode="raw_norm", raw_scale=norms)`, where coefficient 1.0 then adds
    the raw difference at each position, which is the published intervention.

    This replaces taking a pooled last-token difference and adding it at every
    covered position. That was the form the first three attempts at this stage
    used, and it is not activation addition: the vector belonging to position 0
    was being applied at positions 1 and 2 as well. The positive control in the
    preregistration is what caught it.

    A position whose difference is numerically zero, which happens when the two
    padded prompts agree there, keeps a zero row and a zero norm rather than
    being normalised into a spurious unit direction. `SteeringSpec` requires
    unit-norm rows, so such a row is dropped from the covered width instead:
    see the trim below.
    """
    import numpy as np

    from .extraction import capture_positionwise

    layer = int(ACTADD_SETTING["layer"] if layer is None else layer)
    # `positive` and `negative` default to the recorded setting. They are
    # parameters only so the spelling diagnostic can vary them; stage D itself
    # always takes the recorded one.
    pos = ACTADD_SETTING["positive_prompt"] if positive is None else positive
    neg = ACTADD_SETTING["negative_prompt"] if negative is None else negative
    acts = capture_positionwise(lm, [pos, neg], layer)
    raw = acts[0] - acts[1]                      # (P, d_model)
    norms = np.linalg.norm(raw, axis=-1)

    # Trailing positions where the two prompts have become identical carry no
    # contrast, so steering them is adding zero with extra steps. Keep the
    # leading run that does carry one, so the covered width is honest.
    nonzero = norms > 1e-6
    width = int(np.argmin(nonzero)) if not nonzero.all() else int(len(norms))
    if width == 0:
        raise ValueError(
            "the contrast %r vs %r has no difference at its first position; "
            "check ACTADD_SETTING" % (pos, neg)
        )
    raw, norms = raw[:width], norms[:width]
    rows = raw / norms[:, None]
    return rows.astype(np.float32), norms.astype(np.float64), width
