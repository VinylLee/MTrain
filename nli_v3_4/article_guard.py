"""Conservative guard for newly introduced, unambiguous a/an mismatches.

Unknown pronunciations are left to the same-model/human grammar checks. This is
not a general grammar checker and never silently repairs a protected article.
"""
from collections import Counter
import re

from .refinement_v2 import DirectTypedEngine, parse_proposals

VOWEL_SOUND = frozenset("adult adults animal animals artist artists actor actors athlete athletes area areas airplane airplanes aircraft elephant elephants engineer engineers individual individuals instrument instruments object objects office offices orange oranges umbrella umbrellas hour hours hourly honest honor honour elderly empty expensive unusual unusually".split())
CONSONANT_SOUND = frozenset("person persons man men woman women child children human humans house houses university universities unicorn unicorns user users useful European Europeans one once".casefold().split())
ARTICLE_WORD = re.compile(r"\b(a|an)\s+([a-z]+)\b", re.I)


def mismatch_counts(text):
    counts = Counter()
    for match in ARTICLE_WORD.finditer(text):
        article, word = (value.casefold() for value in match.groups())
        if ((word in VOWEL_SOUND and article != "an")
                or (word in CONSONANT_SOUND and article != "a")):
            counts[article, word] += 1
    return counts


def introduced_article_errors(source, candidate):
    introduced = mismatch_counts(candidate) - mismatch_counts(source)
    return [{"article": a, "word": w, "occurrences": n}
            for (a, w), n in sorted(introduced.items())]


class ArticleCheckedEngine(DirectTypedEngine):
    """Prospective engineering repair; no completed v2 output is re-filtered."""
    revision = "round4b_v2_article_guard_engineering_repair1"

    @staticmethod
    def contract():
        return {**DirectTypedEngine.contract(),
            "revision": ArticleCheckedEngine.revision,
            "parent_direct_typed_contract": DirectTypedEngine.fingerprint(),
            "article_guard": "new_unambiguous_mismatches_only_never_rewrite_source_or_article",
            "vowel_sound_words": sorted(VOWEL_SOUND),
            "consonant_sound_words": sorted(CONSONANT_SOUND)}

    @staticmethod
    def parse_proposals(raw, source, operation, round_index=0, seed=42):
        candidates = parse_proposals(raw, source, operation, round_index, seed)
        return [p for p in candidates
                if not introduced_article_errors(source["hypothesis"], p["hypothesis"])]
