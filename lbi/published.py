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
