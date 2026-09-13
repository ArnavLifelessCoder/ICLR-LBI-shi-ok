"""Concepts whose steering direction is decidable without a judge.

Every controllability number elsewhere in this study is a function of an LLM
judge, which makes one objection unanswerable from the data: a directional
effect that fails to resolve is indistinguishable from a judge that emits noise.
The two hypotheses predict the same thing, so no amount of re-reading the same
scores separates them.

This module removes the judge. Each concept here has a behavioral readout that
is computed from the generated string by a rule -- word count, the fraction of
letters that are capitals, the fraction of characters that are digits, the
fraction of tokens that are French function words -- so "did behavior move, and
which way" is decidable by arithmetic. On these concepts the directional metric
has a ground truth to be checked against, and the judge cannot be blamed for
anything.

Two properties make them useful and one makes them limited.

Useful: the readout is exact and cheap, so intervals come from prompt
resampling alone rather than from prompt resampling on top of judge noise; and
the intended direction is known by construction, so a signed area can be scored
as correct or incorrect rather than merely reported.

Limited: these concepts are surface-confounded on purpose. Length is length,
capitals are capitals. That is the price of decidability and it is not a defect
here, because the claim being tested is about the measurement instrument rather
than about representation geometry. Nothing in this module should be read as
evidence about whether real concepts are steerable.
"""

from __future__ import annotations

import re
from typing import Callable

from .concepts import Concept, Pair

# --------------------------------------------------------------------------
# Readouts. Each maps a generated string to a score in [0, 1] that rises with
# the concept, matching the range of the judge's P(Yes) so the same
# dose-response machinery applies unchanged.
# --------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-zÀ-ſ']+")

# Function words that are common in French and rare or absent in English. "de"
# and "la" are deliberately excluded: both occur in English text via names and
# loanwords, and a false positive here would look like a language shift.
_FRENCH = {
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "est", "sont",
    "une", "les", "des", "du", "au", "aux", "ce", "cette", "ces", "qui",
    "que", "pas", "plus", "avec", "pour", "dans", "sur", "mais", "ou",
    "donc", "car", "tres", "très", "bien", "aussi", "comme", "faire",
    "être", "avoir", "peut", "doit", "vais", "suis", "c'est", "d'un",
}

# Roughly the length at which a short answer stops reading as short. The score
# saturates there rather than growing without bound, so one runaway generation
# cannot dominate an averaged dose-response point.
_LENGTH_SATURATION = 80.0


def score_length(text: str) -> float:
    """Word count, squashed to [0, 1]."""
    return min(len(_WORD.findall(text)) / _LENGTH_SATURATION, 1.0)


def score_uppercase(text: str) -> float:
    """Fraction of letters that are capitals."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def score_digits(text: str) -> float:
    """Fraction of non-space characters that are digits."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(c.isdigit() for c in chars) / len(chars)


def score_french(text: str) -> float:
    """Fraction of words that are French function words."""
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return 0.0
    return sum(w in _FRENCH for w in words) / len(words)


READOUTS: dict[str, Callable[[str], float]] = {
    "gt_length": score_length,
    "gt_uppercase": score_uppercase,
    "gt_digits": score_digits,
    "gt_french": score_french,
}


class DeterministicScorer:
    """Scorer for ground-truth concepts. Implements the `Scorer` protocol.

    Raises on an unknown concept rather than returning a default. A silent
    fallback here would produce a flat dose-response, which is exactly the
    pattern the study's criteria select for, so the failure has to be loud.
    """

    def score(self, texts: list[str], concept_name: str) -> list[float]:
        try:
            fn = READOUTS[concept_name]
        except KeyError:
            raise KeyError(
                f"{concept_name!r} has no deterministic readout; "
                f"available: {sorted(READOUTS)}"
            ) from None
        return [float(fn(t)) for t in texts]


# --------------------------------------------------------------------------
# Stimuli. Six template families per concept, marker vocabulary disjoint across
# families so the held-out-family split still means something.
# --------------------------------------------------------------------------

# Direction estimation and effect measurement must not share subject matter, so
# these two lists are disjoint and the tests enforce it.
_PAIR_SUBJECTS = [
    "the quarterly budget review", "the new coffee shop downtown",
    "the training course I signed up for", "the apartment on Fifth Street",
    "the movie we watched last night", "the volunteer programme at the library",
    "the bus route through the old town", "the summer reading list",
    "the bicycle repair workshop", "the community garden plot",
]

_EVAL_SUBJECTS = [
    "the photography club meeting", "the neighbourhood cleanup day",
    "the pottery class on Thursdays", "the local farmers market",
    "the evening language exchange",
]


def _pairs(templates: list[tuple[str, str, str]]) -> list[Pair]:
    """Cross each (family, positive, negative) template with every pair subject."""
    out = []
    for family, pos, neg in templates:
        for s in _PAIR_SUBJECTS:
            out.append(Pair(positive=pos.format(s=s), negative=neg.format(s=s),
                            family=family))
    return out


# Six phrasings across five subjects gives thirty evaluation prompts. The
# judge-scored concepts use six, which bounds the resolution of every number
# computed from them; there is no reason to inherit that limit here.
_EVAL = [
    "Describe {s}.",
    "What should someone know about {s}?",
    "Give your thoughts on {s}.",
    "Explain {s} to a newcomer.",
    "Summarize the situation with {s}.",
    "Write a short note about {s}.",
]


def _eval_prompts() -> list[str]:
    return [t.format(s=s) for s in _EVAL_SUBJECTS for t in _EVAL]


def ground_truth_concepts() -> list[Concept]:
    """The four concepts with decidable readouts.

    `surface_confounded` is True on all of them and that is intentional: the
    readout is the surface property. They exist to validate the metric, not to
    say anything about concept geometry.
    """
    common = dict(safety_relevant=False, surface_confounded=True)

    length = Concept(
        name="gt_length",
        description="Elaborated versus terse response length.",
        pairs=_pairs([
            ("expand", "Let me walk through {s} in full detail, step by step, "
                       "covering the background and the reasoning.", "About {s}: fine."),
            ("elaborate", "There is a great deal worth saying about {s}, and I "
                          "will go through each part carefully.", "{s}: no issues."),
            ("unpack", "I want to unpack {s} thoroughly, because several "
                       "separate considerations bear on it.", "{s} is okay."),
            ("detail", "A complete account of {s} involves a number of points "
                       "that are worth setting out one at a time.", "{s}, briefly: good."),
            ("lengthy", "The full picture of {s} takes some explaining, so I "
                        "will be comprehensive rather than brief.", "{s}: nothing to add."),
            ("thorough", "Being thorough about {s} means addressing the "
                         "context, the specifics and the implications in turn.", "{s}. Done."),
        ]),
        eval_prompts=_eval_prompts(),
        behavior_question="Is the response long and detailed?",
        notes="Readout: word count, saturating at %d." % int(_LENGTH_SATURATION),
        **common,
    )

    upper = Concept(
        name="gt_uppercase",
        description="Capitalized shouting versus ordinary case.",
        pairs=_pairs([
            ("shout", "{s} IS COMPLETELY UNACCEPTABLE RIGHT NOW",
                      "{s} is completely unacceptable right now"),
            ("caps", "EVERYTHING ABOUT {s} NEEDS TO CHANGE TODAY",
                     "everything about {s} needs to change today"),
            ("yell", "NOBODY WARNED ME ABOUT {s} AND IT SHOWS",
                     "nobody warned me about {s} and it shows"),
            ("loud", "THIS IS THE PROBLEM WITH {s} EVERY SINGLE TIME",
                     "this is the problem with {s} every single time"),
            ("emphatic", "I CANNOT BELIEVE {s} TURNED OUT LIKE THIS",
                         "i cannot believe {s} turned out like this"),
            ("blare", "PAY ATTENTION TO {s} BEFORE IT GETS WORSE",
                      "pay attention to {s} before it gets worse"),
        ]),
        eval_prompts=_eval_prompts(),
        behavior_question="Is the response written in capital letters?",
        notes="Readout: fraction of letters that are uppercase.",
        **common,
    )

    digits = Concept(
        name="gt_digits",
        description="Numeric versus prose description.",
        pairs=_pairs([
            ("counts", "{s}: 47 items, 1982 founded, 365 days, 12 rooms.",
                       "{s}: many items, founded long ago, open year round."),
            ("stats", "{s} scored 88, then 91, then 74 across 3 reviews.",
                      "{s} scored well, then better, then worse across reviews."),
            ("dates", "{s} ran from 2019 to 2024, peaking in 2021 at 350.",
                      "{s} ran for several years, peaking early in the period."),
            ("measures", "{s} covers 240 square metres over 2 floors, 18 seats.",
                         "{s} covers a fair area over two floors, with some seats."),
            ("prices", "{s} costs 15 per hour, 120 per week, 400 per month.",
                       "{s} costs a modest amount hourly, weekly and monthly."),
            ("tallies", "{s} had 6 sessions, 54 attendees, 9 volunteers, 2 vans.",
                        "{s} had several sessions, many attendees and volunteers."),
        ]),
        eval_prompts=_eval_prompts(),
        behavior_question="Does the response contain many numerals?",
        notes="Readout: fraction of non-space characters that are digits.",
        **common,
    )

    french = Concept(
        name="gt_french",
        description="French versus English response language.",
        pairs=_pairs([
            ("fr_desc", "Je pense que {s} est une tres bonne chose pour nous.",
                        "I think that {s} is a very good thing for us."),
            ("fr_opin", "Il est clair que {s} ne peut pas continuer comme ca.",
                        "It is clear that {s} cannot continue like this."),
            ("fr_note", "Nous avons parle de {s} avec les autres hier soir.",
                        "We spoke about {s} with the others last night."),
            ("fr_plan", "Vous devez voir {s} avant de decider quoi que ce soit.",
                        "You must see {s} before deciding anything at all."),
            ("fr_ask", "Est-ce que {s} vous semble utile pour cette semaine ?",
                       "Does {s} seem useful to you for this coming week?"),
            ("fr_tell", "Ils ont dit que {s} etait deja pret depuis longtemps.",
                        "They said that {s} was already ready a long time ago."),
        ]),
        eval_prompts=_eval_prompts(),
        behavior_question="Is the response written in French?",
        notes="Readout: fraction of words that are French function words.",
        **common,
    )

    return [length, upper, digits, french]
