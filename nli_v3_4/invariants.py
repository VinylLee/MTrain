"""Deterministic INV edit checks; no model scores or corpus identities."""
from collections import Counter
from difflib import SequenceMatcher
import re

import augment_snli_all_label_mrs as legacy


WORD = re.compile(r"\b[\w]+(?:['’][\w]+)*\b", re.UNICODE)
NUMBER_WORDS = frozenset("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion first second third fourth fifth sixth seventh eighth ninth tenth".split())
FIXED_WORDS = NUMBER_WORDS | legacy.STRICT_QUANTIFIERS | frozenset(
    "not no never none nobody neither nor only always sometimes often usually every each any all some many few most more less same other either both must may might can could should would will shall before after during since until man men woman women boy boys girl girls male female adult adults child children baby babies infant infants teenager teenagers elderly young old".split())
CLAUSE_LINKS = frozenset("and but while although because when if yet".split())


def _contains_span(words, span):
    return any(words[i:i+len(span)] == span for i in range(len(words)-len(span)+1))


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
    # A final full stop may be normalized along with a substantive synonym edit.
    # Keep internal punctuation/scope; punctuation alone still has zero edits.
    def skeleton(text):
        return re.sub(r"\.$", "", WORD.sub("_", text).strip()).strip()
    return skeleton(source) == skeleton(candidate)


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
    if not (1 <= sum(removed.values()) <= 2 and not added
            and all(after[token] > 0 for token in removed)):
        return False
    original = [w.lower() for w in WORD.findall(source)]
    rewritten = [w.lower() for w in WORD.findall(candidate)]
    edits = [op for op in SequenceMatcher(a=original,b=rewritten,autojunk=False).get_opcodes() if op[0] != "equal"]
    if len(edits) != 1:
        return False
    kind, start, end, new_start, new_end = edits[0]
    span = original[start:end]
    replacement = rewritten[new_start:new_end]
    if set(span) & {"men","women","boys","girls"} and replacement != ["they"]:
        return False
    if set(span) & {"man","boy"} and replacement != ["he"]:
        return False
    if set(span) & {"woman","girl"} and replacement != ["she"]:
        return False
    # Conservative supported case: replace a repeated definite subject in a
    # later coordinated/subordinate clause. Merely retaining another occurrence
    # of 'man' (e.g. 'a man ... another man') does not establish coreference.
    return (kind == "replace" and new_end-new_start == 1
            and rewritten[new_start] in {"he","she","it","they"}
            and 2 <= len(span) <= 4 and span[0] == "the" and start > 0
            and original[start-1] in CLAUSE_LINKS
            and _contains_span(original[:start], span))


def has_possible_sentence_antecedent(text):
    words = [w.lower() for w in WORD.findall(text)]
    return any(words[start] == "the" and words[start-1] in CLAUSE_LINKS
               and _contains_span(words[:start],words[start:start+length])
               for start in range(1,len(words)) for length in range(2,5)
               if start+length <= len(words))
