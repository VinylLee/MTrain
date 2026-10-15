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
AGE_ALIASES = (frozenset(("child","kid")),frozenset(("children","kids")))
FIRST_COMMON_WORDS = frozenset("children kids thanks immediately instantly quickly rapidly slowly gradually buy purchase start begin outside outdoors photograph photo".split())
FIXED_SPANS = tuple(tuple(s.split()) for s in (
    "about to", "used to", "at least", "at most", "at the very minimum",
    "no longer", "as soon as", "in front of", "in spite of", "rather than"))


def fixed_span_positions(words):
    lowered = [word.lower() for word in words]
    return {i + offset for span in FIXED_SPANS for i in range(len(words)-len(span)+1)
            if tuple(lowered[i:i+len(span)]) == span for offset in range(len(span))}


def same_age_category(left,right):
    return any(left.lower() in group and right.lower() in group for group in AGE_ALIASES)


def realize_synonym_plan(text,plan):
    """Replace one or two explicit source word slots; copy every other byte."""
    if not isinstance(plan,dict) or not isinstance(plan.get("edits"),list) or not 1 <= len(plan["edits"]) <= 2:
        return None
    words=list(WORD.finditer(text));replacements={}
    protected = fixed_span_positions([word.group() for word in words])
    for edit in plan["edits"]:
        if not isinstance(edit,dict):return None
        index=edit.get("word_index");replacement=edit.get("replacement")
        if type(index) is not int or not 0 <= index < len(words) or index in replacements:return None
        if index in protected:return None
        old=words[index].group()
        if edit.get("old_word")!=old or not isinstance(replacement,str):return None
        replacement=replacement.strip()
        if not re.fullmatch(r"[\w]+(?:['’][\w]+)*(?: [\w]+(?:['’][\w]+)*){0,3}",replacement):return None
        if old.lower()==replacement.lower() or not legacy.content_tokens(old):return None
        if old.lower() in {w.lower() for w in WORD.findall(replacement)}:return None
        if not 1 <= len(legacy.content_tokens(replacement)) <= 2:return None
        alias=same_age_category(old,replacement)
        if (old.lower() in FIXED_WORDS or any(w.lower() in FIXED_WORDS for w in WORD.findall(replacement))) and not alias:return None
        if old[0].isupper() and not (index==0 and old.lower() in FIRST_COMMON_WORDS):return None
        if any(w[0].isupper() for w in WORD.findall(replacement)) and not (index==0 and old.lower() in FIRST_COMMON_WORDS):return None
        # Quoted words can be literal quotations or names; do not rewrite them.
        start,end=words[index].span()
        if start and text[start-1] in '\"“‘' and end<len(text) and text[end] in '\"”’':return None
        replacements[index]=replacement
    result=text
    for index in sorted(replacements,reverse=True):
        start,end=words[index].span();result=result[:start]+replacements[index]+result[end:]
    return result if valid_invariant_tokens(text,result) else None


def agentive_passive_count(text):
    words=[w.lower() for w in WORD.findall(text)]
    irregular=set("worn ridden held eaten thrown seen known made done given taken said bought found read shown chosen written hit put cut set built driven taught told sold sent brought caught".split())
    count=0
    for index,word in enumerate(words):
        if word not in {"is","are","was","were","be","been","being","get","gets","got"}:continue
        position=index+1
        while position<len(words) and (words[position] in {"not","never","been","being"} or words[position].endswith('ly')):position+=1
        if position>=len(words):continue
        verb=words[position]
        if verb in irregular or (len(verb)>3 and verb.endswith(('ed','en'))):
            if 'by' in words[position+1:position+16]:count+=1
    return count


def has_agentive_passive(text):
    return agentive_passive_count(text)>0


_IRREGULAR_VERBS = tuple(tuple(group.split()) for group in (
    "wear wore worn", "ride rode ridden", "hold held held", "eat ate eaten",
    "throw threw thrown", "see saw seen", "know knew known", "make made made",
    "do did done", "give gave given", "take took taken", "say said said",
    "buy bought bought", "find found found", "read read read", "show showed shown",
    "choose chose chosen", "write wrote written", "hit hit hit", "put put put",
    "cut cut cut", "set set set", "build built built", "drive drove driven",
    "teach taught taught", "tell told told", "sell sold sold", "send sent sent",
    "bring brought brought", "catch caught caught", "sing sang sung"))
_PAST = {group[1] for group in _IRREGULAR_VERBS}
_PARTICIPLES = {group[2] for group in _IRREGULAR_VERBS}
_MODALS = frozenset("can could may might must shall should will would".split())
_BE = frozenset("am is are was were be been being get gets got".split())
_AUX = _BE | _MODALS | {"have", "has", "had", "do", "does", "did"}
_ADVERBS = frozenset("not never always often sometimes usually already still just".split())
_PREPOSITIONS = frozenset("by in on at near under over with without beside behind before after outside inside to from for through".split())
_PRONOUN_CASE = {"me": "i", "him": "he", "her": "she", "us": "we", "them": "they"}


def _verb_roots(word):
    roots = {word}
    for base, past, participle in _IRREGULAR_VERBS:
        if word in {base, past, participle, base + "s", base + "ing"}:
            roots.add(base)
    if word.endswith("ies"): roots.add(word[:-3] + "y")
    if word.endswith("s") and not word.endswith("ss"): roots.add(word[:-1])
    if word.endswith("es"): roots.add(word[:-2])
    for suffix in ("ed", "ing"):
        if len(word) > len(suffix)+1 and word.endswith(suffix):
            stem = word[:-len(suffix)]
            roots.update((stem, stem + "e"))
            if stem.endswith("i"): roots.add(stem[:-1] + "y")
            if len(stem)>2 and stem[-1] == stem[-2]: roots.add(stem[:-1])
    return roots


def _voice_words(text):
    # Unsupported clauses/contractions abstain instead of guessing a parse.
    if re.search(r"[,;:?!\"“”]", text) or re.search(r"['’]", text):
        return None
    return [word.lower() for word in WORD.findall(text)]


def _is_adverb(word):
    return word in _ADVERBS or word.endswith("ly")


def _aux_start(words, verb):
    start = verb
    while start and (words[start-1] in _AUX or _is_adverb(words[start-1])):
        start -= 1
    return start


def _tense_aspect(words, start, verb, passive):
    chain = words[start:verb]
    modals = tuple(word for word in chain if word in _MODALS)
    perfect = bool(set(chain) & {"have", "has", "had"})
    progressive = "being" in chain if passive else words[verb].endswith("ing") and bool(set(chain) & _BE)
    if modals:
        tense = "modal"
    elif chain:
        finite = next((word for word in chain if word in _AUX), None)
        if finite in {"was", "were", "had", "did", "got"}: tense = "past"
        elif finite in {"am", "is", "are", "have", "has", "do", "does", "get", "gets"}: tense = "present"
        else: return None
    elif words[verb] in _PAST or words[verb].endswith("ed"):
        # read/hit/put/cut/set have ambiguous present/past forms: abstain.
        if words[verb] in {g[0] for g in _IRREGULAR_VERBS if g[0] == g[1]}: return None
        tense = "past"
    elif words[verb].endswith("ing"):
        return None
    else:
        tense = "present"
    return tense, modals, perfect, progressive


def _role_prefix(short, long):
    # The optional remainder is an unchanged PP adjunct, checked by inventory.
    return bool(short) and long[:len(short)] == short and (
        len(short) == len(long) or long[len(short)] in _PREPOSITIONS)


def voice_preserves_form(source, candidate):
    """Conservative simple transitive guard, not an English semantic prover.

    Require a recoverable agent/patient swap and retain lexical material, tense,
    modality and aspect. Unsupported/ambiguous parses return False. The same-model
    grammar/equivalence judge is still required after this structural check.
    """
    source_words, candidate_words = _voice_words(source), _voice_words(candidate)
    if not source_words or not candidate_words: return False
    source_passive, candidate_passive = has_agentive_passive(source), has_agentive_passive(candidate)
    if source_passive == candidate_passive: return False
    active, passive = (candidate_words, source_words) if source_passive else (source_words, candidate_words)
    frames = []
    for pv, word in enumerate(passive):
        if word not in _PARTICIPLES and not word.endswith("ed"): continue
        ps = _aux_start(passive, pv)
        if not ps or not (set(passive[ps:pv]) & _BE): continue
        by = pv+1
        while by < len(passive) and _is_adverb(passive[by]): by += 1
        if by >= len(passive) or passive[by] != "by" or by+1 == len(passive): continue
        for av, verb in enumerate(active):
            if not (_verb_roots(verb) & _verb_roots(word)): continue
            astart = _aux_start(active, av)
            if not astart or av+1 == len(active): continue
            norm = lambda values: [_PRONOUN_CASE.get(v, v) for v in values]
            if not _role_prefix(norm(active[:astart]), norm(passive[by+1:])): continue
            if not _role_prefix(norm(passive[:ps]), norm(active[av+1:])): continue
            signature = _tense_aspect(active, astart, av, False)
            if signature is None or signature != _tense_aspect(passive, ps, pv, True): continue
            def inventory(words, start, verb_index, agent_by=None):
                return Counter(norm([w for i, w in enumerate(words)
                                     if i != verb_index and i != agent_by
                                     and not (start <= i < verb_index and w in _AUX)]))
            if inventory(active, astart, av) == inventory(passive, ps, pv, by):
                frames.append((av, pv))
    return len(frames) == 1 and valid_invariant_tokens(source, candidate)


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
