"""Deterministic INV edit checks; no model scores or corpus identities."""
from collections import Counter
import re

import augment_snli_all_label_mrs as legacy


WORD = re.compile(r"\b[\w]+(?:['’][\w]+)*\b", re.UNICODE)
NUMBER_WORDS = frozenset("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion first second third fourth fifth sixth seventh eighth ninth tenth".split())
FIXED_WORDS = NUMBER_WORDS | legacy.STRICT_QUANTIFIERS | frozenset(
    "not no never none nobody neither nor only always sometimes often usually every each any all some many few most more less same other either both must may might can could should would will shall before after during since until man men woman women boy boys girl girls male female adult adults child children baby babies infant infants teenager teenagers elderly young old".split())


def numeric_signature(text):
    # Protect both digits and spelled quantities, including repeated tokens.
    return Counter(t.lower() for t in WORD.findall(text)
                   if t.lower() in NUMBER_WORDS or any(c.isdigit() for c in t))


def valid_invariant_tokens(source, candidate):
    return (numeric_signature(source) == numeric_signature(candidate)
            and legacy.has_negation(source) == legacy.has_negation(candidate))


def valid_synonym_edit(source, candidate):
    """Exactly one or two aligned content-word substitutions, no deletions."""
    before, after = WORD.findall(source), WORD.findall(candidate)
    if len(before) != len(after) or not valid_invariant_tokens(source, candidate):
        return False
    edits = [(i, a, b) for i, (a, b) in enumerate(zip(before, after)) if a.lower() != b.lower()]
    if not 1 <= len(edits) <= 2:
        return False
    for index, a, b in edits:
        if a.lower() in FIXED_WORDS or b.lower() in FIXED_WORDS:
            return False
        if not legacy.content_tokens(a) or not legacy.content_tokens(b):
            return False
        # Proper names and acronyms inside a sentence are never synonym slots.
        if a[0].isupper() or b[0].isupper():
            return False
    # Punctuation is not an additional rewriting operation.
    return WORD.sub("_", source).strip() == WORD.sub("_", candidate).strip()


def has_actual_pronoun_replacement(source, candidate):
    def content_counts(text):
        return Counter(t.lower() for t in WORD.findall(text)
                       if legacy.content_tokens(t) and t.lower() not in legacy.REFERENCE_TOKENS)
    before, after = content_counts(source), content_counts(candidate)
    removed, added = before - after, after - before
    # Counter rather than sets permits replacement of a repeated NP occurrence.
    # The MR requires an antecedent in the sentence itself. Replacing the only
    # occurrence ("a man" -> "he") cannot satisfy that; deleted attributes must
    # remain in the retained antecedent too, rather than disappear.
    return (1 <= sum(removed.values()) <= 2 and not added
            and all(after[token] > 0 for token in removed))


def has_possible_sentence_antecedent(text):
    counts = Counter(t.lower() for t in WORD.findall(text)
                     if legacy.content_tokens(t) and t.lower() not in legacy.REFERENCE_TOKENS)
    return any(count >= 2 for count in counts.values())
