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


def same_age_category(left,right):
    return any(left.lower() in group and right.lower() in group for group in AGE_ALIASES)


def realize_synonym_plan(text,plan):
    """Replace one or two explicit source word slots; copy every other byte."""
    if not isinstance(plan,dict) or not isinstance(plan.get("edits"),list) or not 1 <= len(plan["edits"]) <= 2:
        return None
    words=list(WORD.finditer(text));replacements={}
    for edit in plan["edits"]:
        if not isinstance(edit,dict):return None
        index=edit.get("word_index");replacement=edit.get("replacement")
        if type(index) is not int or not 0 <= index < len(words) or index in replacements:return None
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


def voice_preserves_form(source,candidate):
    def aspect(text):
        text=text.lower().replace("’","'")
        text=re.sub(r"\b(he|she|it|that|there)'s\b",r"\1 is",text)
        text=re.sub(r"\b(i)'m\b",r"\1 am",text)
        text=re.sub(r"\b(we|they|you)'re\b",r"\1 are",text)
        words=[w.lower() for w in WORD.findall(text)];progressive=perfect=0
        for i,w in enumerate(words):
            if w in {"is","are","am","was","were","be"}:
                after=[v for v in words[i+1:i+5] if v not in {'not','never'} and not v.endswith('ly')]
                if after and (after[0]=='being' or after[0].endswith('ing')):progressive+=1
            if w in {'has','have','had'}:
                after=[v for v in words[i+1:i+4] if v not in {'not','never'}]
                if after and (after[0] in {'been','bought','held','known','thrown','worn','ridden','eaten','given','taken','seen','made','done','said','found','read','shown','chosen','written','hit','put','cut','set','built','driven','taught','told','sold','sent','brought','caught'}
                              or len(after[0])>3 and after[0].endswith(('ed','en'))):perfect+=1
        return progressive,perfect
    def names(text):
        return {w.lower().split("'")[0] for w in WORD.findall(text) if w[0].isupper()
                and w.lower() not in legacy.CONTENT_STOPWORDS|legacy.REFERENCE_TOKENS|legacy.STRICT_QUANTIFIERS}
    return aspect(source)==aspect(candidate) and names(source)<=set(w.lower().split("'")[0] for w in WORD.findall(candidate))


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
