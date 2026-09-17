"""Concepts whose steering direction is decidable without a judge.

Every controllability number elsewhere in this study is a function of an LLM
judge, which makes one objection unanswerable from the data: a directional
effect that fails to resolve is indistinguishable from a judge that emits noise.
The two hypotheses predict the same thing, so no amount of re-reading the same
scores separates them.

This module removes the judge. Each concept here has a behavioral readout that
is computed from the generated string by a rule -- a word count, a fraction of
letters that are capitals, a fraction of words drawn from a list -- so "did
behavior move, and which way" is decidable by arithmetic. On these concepts the
directional metric has a ground truth to be checked against, and the judge
cannot be blamed for anything.

Ten concepts, chosen so that **baseline position varies independently of concept
identity**. The first four did not: three sat at a floor and one in mid-range,
so any effect of how much headroom a behavior has was perfectly confounded with
which behavior it was, and the question could not be asked. The clearest case is
the pair added to break that: `gt_uppercase` and `gt_lowercase` read the same
surface property from opposite ends and sit at roughly 0.02 and 0.98 on ordinary
text.

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



# The hundred or so words that carry most English text. A readout over these
# sits high at baseline by construction, which is the point: the original four
# concepts all sit at a floor except one, so headroom could not be separated
# from concept identity.
_COMMON = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
    "time", "no", "just", "him", "know", "take", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back", "after",
    "use", "two", "how", "our", "work", "first", "well", "way", "even", "new",
    "want", "because", "any", "these", "give", "day", "most", "us", "is",
    "are", "was", "were", "been", "has", "had", "said", "did", "very",
}

_FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
                 "ourselves"}

_SENTENCE_END = re.compile(r"[.!?]")


def score_lowercase(text: str) -> float:
    """Fraction of letters that are lower case.

    The inverse of `score_uppercase`, and included for that reason. It reads the
    same surface property from the opposite end, so it sits near 1.0 at baseline
    where uppercase sits near 0.03. Any effect of baseline position that is
    really an effect of which concept it is has to explain those two differing.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.islower() for c in letters) / len(letters)


def score_common(text: str) -> float:
    """Fraction of words drawn from the common-word list. High at baseline."""
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return 0.0
    return sum(w in _COMMON for w in words) / len(words)


def score_longwords(text: str) -> float:
    """Fraction of words of seven letters or more. Low-to-middling at baseline."""
    words = _WORD.findall(text)
    if not words:
        return 0.0
    return sum(len(w) >= 7 for w in words) / len(words)


def score_firstperson(text: str) -> float:
    """Fraction of words that are first-person pronouns."""
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return 0.0
    return sum(w in _FIRST_PERSON for w in words) / len(words)


def score_question(text: str) -> float:
    """Fraction of sentence-ending marks that are question marks."""
    ends = _SENTENCE_END.findall(text)
    if not ends:
        return 0.0
    return sum(e == "?" for e in ends) / len(ends)


def score_punctuation(text: str) -> float:
    """Fraction of non-space characters that are punctuation."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(not c.isalnum() for c in chars) / len(chars)


READOUTS: dict[str, Callable[[str], float]] = {
    "gt_length": score_length,
    "gt_uppercase": score_uppercase,
    "gt_digits": score_digits,
    "gt_french": score_french,
    "gt_lowercase": score_lowercase,
    "gt_common": score_common,
    "gt_longwords": score_longwords,
    "gt_firstperson": score_firstperson,
    "gt_question": score_question,
    "gt_punctuation": score_punctuation,
}


class DeterministicScorer:
    """Scorer for rule-scored concepts. Implements the `Scorer` protocol.

    `extra` adds readouts for concepts defined elsewhere, such as the published
    replication in `published.py`. It is a constructor argument rather than a
    global registration on purpose: a module that mutated `READOUTS` on import
    would change what this scorer accepts depending on what else happened to be
    imported, which is both hard to test and hard to reason about.

    Raises on an unknown concept rather than returning a default. A silent
    fallback would produce a flat dose-response, which is exactly the pattern
    the study's criteria select for, so the failure has to be loud.
    """

    def __init__(self, extra: dict[str, Callable[[str], float]] | None = None):
        self.readouts = dict(READOUTS)
        if extra:
            clash = set(extra) & set(READOUTS)
            if clash:
                raise ValueError(
                    "extra readouts would shadow built-in ones: %s" % sorted(clash)
                )
            self.readouts.update(extra)

    def score(self, texts: list[str], concept_name: str) -> list[float]:
        try:
            fn = self.readouts[concept_name]
        except KeyError:
            raise KeyError(
                f"{concept_name!r} has no deterministic readout; "
                f"available: {sorted(self.readouts)}"
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


    lower = Concept(
        name="gt_lowercase",
        description="All-lowercase versus capitalized text.",
        pairs=_pairs([
            ("nocaps", "i went to see {s} and it was fine honestly",
                       "I Went To See {s} And It Was Fine Honestly"),
            ("flat", "nobody told me about {s} until last week",
                     "Nobody Told Me About {s} Until Last Week"),
            ("plain", "we talked about {s} for almost an hour",
                      "We Talked About {s} For Almost An Hour"),
            ("soft", "she keeps asking me about {s} every day",
                     "She Keeps Asking Me About {s} Every Day"),
            ("quiet", "they moved {s} to a different afternoon",
                      "They Moved {s} To A Different Afternoon"),
            ("small", "he said {s} would probably be cancelled",
                      "He Said {s} Would Probably Be Cancelled"),
        ]),
        eval_prompts=_eval_prompts(),
        behavior_question="Is the response written in lower case?",
        notes="Readout: fraction of letters that are lower case. Near 1.0 at "
              "baseline, which is the point: it is the inverse of gt_uppercase.",
        **common,
    )

    common_words = Concept(
        name="gt_common",
        description="Plain common vocabulary versus elaborate rare vocabulary.",
        pairs=_pairs([
            ("plainA", "I think {s} is good and we can all go to it",
                       "Ascertaining {s} merits requires substantive deliberation"),
            ("plainB", "We can see that {s} will be at the same time",
                       "Observers anticipate {s} maintaining equivalent scheduling"),
            ("plainC", "They said {s} is not going to be a problem",
                       "Organisers indicated {s} presents negligible complications"),
            ("plainD", "You know {s} is just what we all do now",
                       "Participants regard {s} as customary contemporary practice"),
            ("plainE", "She would like {s} to have more time for it",
                       "Stakeholders advocate extending {s} allotted duration"),
            ("plainF", "He did not want {s} to take up the day",
                       "Respondents disfavoured {s} consuming entire afternoons"),
        ]),
        eval_prompts=_eval_prompts(),
        behavior_question="Is the response written in plain common words?",
        notes="Readout: fraction of words in a common-word list. High at baseline.",
        **common,
    )

    longwords = Concept(
        name="gt_longwords",
        description="Long words versus short ones.",
        pairs=_pairs([
            ("longA", "{s} demonstrated considerable organisational improvement",
                      "{s} went well and the team was glad"),
            ("longB", "{s} required substantial preliminary coordination",
                      "{s} took a lot of work up front"),
            ("longC", "{s} generated unexpectedly enthusiastic participation",
                      "{s} got a lot more people than we thought"),
            ("longD", "{s} necessitated additional administrative preparation",
                      "{s} meant more forms to fill in"),
            ("longE", "{s} exhibited remarkably consistent attendance patterns",
                      "{s} had the same folk turn up each time"),
            ("longF", "{s} encountered significant logistical difficulties",
                      "{s} ran into a few big problems"),
        ]),
        eval_prompts=_eval_prompts(),
        behavior_question="Does the response use long words?",
        notes="Readout: fraction of words of seven letters or more.",
        **common,
    )

    firstperson = Concept(
        name="gt_firstperson",
        description="First-person versus third-person narration.",
        pairs=_pairs([
            ("meA", "I went to {s} and I told my friend about it",
                    "She went to {s} and she told her friend about it"),
            ("meB", "We organised {s} ourselves and we were pleased",
                    "They organised {s} themselves and they were pleased"),
            ("meC", "My view of {s} is that I would go again",
                    "Her view of {s} is that she would go again"),
            ("meD", "I keep meaning to visit {s} with my family",
                    "He keeps meaning to visit {s} with his family"),
            ("meE", "Our part in {s} took us most of the morning",
                    "Their part in {s} took them most of the morning"),
            ("meF", "I asked about {s} because I wanted to know",
                    "He asked about {s} because he wanted to know"),
        ]),
        eval_prompts=_eval_prompts(),
        behavior_question="Is the response written in the first person?",
        notes="Readout: fraction of words that are first-person pronouns.",
        **common,
    )

    question = Concept(
        name="gt_question",
        description="Questions versus statements.",
        pairs=_pairs([
            ("qA", "Is {s} going to happen again this year?",
                   "{s} is going to happen again this year."),
            ("qB", "Did anyone actually enjoy {s} last time?",
                   "Someone actually enjoyed {s} last time."),
            ("qC", "Should we move {s} to a different day?",
                   "We should move {s} to a different day."),
            ("qD", "Has {s} ever been cancelled before now?",
                   "{s} has been cancelled before now."),
            ("qE", "Would more people come to {s} on a weekend?",
                   "More people would come to {s} on a weekend."),
            ("qF", "Can anyone explain why {s} runs so late?",
                   "Someone can explain why {s} runs so late."),
        ]),
        eval_prompts=_eval_prompts(),
        behavior_question="Is the response phrased as a question?",
        notes="Readout: fraction of sentence-ending marks that are question marks.",
        **common,
    )

    punctuation = Concept(
        name="gt_punctuation",
        description="Heavily punctuated versus plainly punctuated text.",
        pairs=_pairs([
            ("pA", "{s} -- yes, really! -- was, well, fine (mostly).",
                   "{s} was mostly fine"),
            ("pB", "{s}: good, bad, and (frankly) odd; all at once!",
                   "{s} was good and bad and odd all at once"),
            ("pC", "{s}? Again?! Honestly, no -- not this time.",
                   "{s} again honestly not this time"),
            ("pD", "{s}; which, as noted, is \"fine\" (apparently).",
                   "{s} which as noted is fine apparently"),
            ("pE", "{s} -- late, again; and nobody, nobody, cared!",
                   "{s} late again and nobody cared"),
            ("pF", "{s}... maybe? Or not. Who knows -- really!",
                   "{s} maybe or not who knows really"),
        ]),
        eval_prompts=_eval_prompts(),
        behavior_question="Is the response heavily punctuated?",
        notes="Readout: fraction of non-space characters that are punctuation.",
        **common,
    )

    return [length, upper, digits, french, lower, common_words, longwords,
            firstperson, question, punctuation]
