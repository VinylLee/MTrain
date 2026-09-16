"""High-quality, pair-aware SNLI augmentation with an isolated judge pass.

The legacy augmentation code transforms the hypothesis without always exposing
the premise to the generator.  This script makes the NLI pair and target label
part of every generation prompt, asks for several candidates, and uses a fresh
stateless Gemma request to accept only candidates that satisfy the relation.

The default invocation exercises every LABEL_MRS entry on a small sample::

    python augment_snli_all_label_mrs.py --sample-size 12 --overwrite

Use ``--all`` only after the smoke report passes the configured quality gates.
The input file is never modified.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

from provenance_schema import (
    COMPONENT_MRS_FIELD,
    PROVENANCE_SCHEMA_VERSION,
    component_mrs_from_log as _component_mrs_from_log,
    config_fingerprint,
    sha256_file,
)


# Names the *prompt contract* only. Adding the component_mrs provenance field did not
# change any prompt, operation spec, or stage plan, so this deliberately stays v3.3 —
# bumping it would misreport the prompts and break the match with existing reports and
# downstream directory/config names. Schema changes bump PROVENANCE_SCHEMA_VERSION instead.
PROMPT_VERSION = "snli-pair-aware-all-label-mrs-v3.3"
ALLOWED_MODEL = "gemma-3-4b-it-qat"
LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_IDS = {value: key for key, value in LABELS.items()}
NEGATION_TOKENS = {
    "no", "not", "never", "none", "nobody", "nothing", "neither", "without",
    "cannot",
}
CONTENT_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "has", "have", "had", "to", "of", "in",
    "on", "at", "for", "from", "with", "and", "or", "but", "as", "by",
    "this", "that", "these", "those", "there", "it",
}

STRICT_QUANTIFIERS = {"all", "every", "each", "many", "most", "few", "several"}
REFERENCE_TOKENS = {
    "he", "him", "she", "her", "it", "they", "them", "his", "its", "their",
    "one", "ones", "someone", "somebody",
}
VAGUE_CONDITION_TOKENS = {
    "something", "anything", "someone", "somebody", "somewhere", "somehow",
    "stuff", "thing", "things",
}
ANTONYM_PAIRS = {
    frozenset(pair)
    for pair in (
        ("inside", "outside"), ("indoors", "outdoors"), ("in", "out"),
        ("up", "down"), ("above", "below"), ("front", "back"),
        ("left", "right"), ("open", "closed"), ("opening", "closing"),
        ("enter", "exit"), ("entering", "exiting"), ("arrive", "leave"),
        ("arriving", "leaving"), ("rise", "fall"), ("rising", "falling"),
        ("start", "stop"), ("starting", "stopping"), ("stand", "sit"),
        ("standing", "sitting"), ("awake", "asleep"), ("alive", "dead"),
        ("full", "empty"), ("wet", "dry"), ("clean", "dirty"),
        ("hot", "cold"), ("large", "small"), ("big", "small"),
        ("fast", "slow"), ("quickly", "slowly"), ("young", "old"),
        ("light", "dark"), ("visible", "hidden"),
    )
}
DEFAULT_DATA_DIR = Path(
    "nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat"
)

GENERATOR_SYSTEM_PROMPT = """You are a meticulous SNLI data editor.
Create short, natural English hypotheses for natural-language inference.
Treat the premise as the only known world. Use the strict SNLI definitions:
- entailment: the premise makes the hypothesis definitely true;
- contradiction: the premise makes the hypothesis definitely false;
- neutral: the hypothesis may be true, but the premise neither proves nor
  disproves it.
Do not rely on ordinary-world assumptions to call something a contradiction.
Do not explain your reasoning. Return only the requested JSON array."""

JUDGE_SYSTEM_PROMPT = """You are a strict, conservative SNLI quality judge.
Classify from the premise alone. Contradiction requires logical incompatibility,
not merely different, missing, unlikely, or newly added information. Entailment
requires the hypothesis to be guaranteed. Otherwise choose neutral.
Judge grammar and operation compliance independently. Return only JSON and do
not provide hidden reasoning; the reason field must be a short error tag."""

OPERATION_SPECS: Dict[str, Dict[str, str]] = {
    "negation_flip": {
        "mr_type": "flip",
        "family": "flip",
        "instruction": """Directly negate the entire SOURCE HYPOTHESIS with the
smallest grammatical edit. Preserve every entity, action, object, count, attribute,
and location. Quantifier scope is critical: an existential source beginning with
A/An/Some/Someone must become No/Nobody/No one (A man walks -> No man walks);
'A man does not walk' is NOT its logical negation. For definite subjects, negate
the main predicate. Avoid double negation and merely different facts.""",
        "target": "contradiction",
    },
    "antonym_substitution": {
        "mr_type": "flip",
        "family": "flip",
        "instruction": """Replace exactly one key adjective, verb, or adverb in the
SOURCE HYPOTHESIS with a direct contextual antonym. Preserve all other facts and do
not use negation. The result must be logically incompatible with the PREMISE, not
merely different or less specific.""",
        "target": "contradiction",
    },
    "adding_contradiction": {
        "mr_type": "flip",
        "family": "flip",
        "instruction": """Copy the complete SOURCE HYPOTHESIS as the first clause,
then append one short clause that contradicts a DIFFERENT fact guaranteed by the
PREMISE but absent from the source hypothesis. Do not contradict the copied clause
itself and do not add merely unmentioned information. The combined statement must
be natural, internally coherent, and false given the premise.""",
        "target": "contradiction",
    },
    "pronoun_substitution": {
        "mr_type": "inv",
        "family": "inv",
        "instruction": """Replace exactly one unambiguous noun phrase in the SOURCE
HYPOTHESIS with the correct pronoun. The antecedent must remain unambiguous from the
sentence itself. Preserve number, stated gender, grammatical case, all facts, and
the exact truth conditions. If no safe replacement exists, emit no candidate.""",
        "target": "source",
    },
    "voice_switch": {
        "mr_type": "inv",
        "family": "inv",
        "instruction": """Convert the SOURCE HYPOTHESIS between active and passive
voice only when it has a clear agent-action-patient structure. Preserve agent,
patient, tense, aspect, polarity, and exact truth conditions. If a natural voice
conversion is unavailable, emit no candidate.""",
        "target": "source",
    },
    "synonym_replacement": {
        "mr_type": "inv",
        "family": "inv",
        "instruction": """Replace exactly one or two common content words in the
SOURCE HYPOTHESIS with natural contextual synonyms. Do not alter named entities,
numbers, quantifiers, roles, locations, or polarity. The result must have identical
truth conditions, not just a similar topic.""",
        "target": "source",
    },
    "uninformative": {
        "mr_type": "neutral",
        "family": "neutral",
        "instruction": """Copy the complete SOURCE HYPOTHESIS as the first clause,
then append exactly one concrete, plausible detail that the PREMISE neither states
nor rules out. The new detail must make the whole hypothesis neutral. Avoid generic
weather/time boilerplate, modality words, and facts that contradict the premise.""",
        "target": "neutral",
    },
    "conditional_clause": {
        "mr_type": "neutral",
        "family": "neutral",
        "instruction": """Copy the complete SOURCE HYPOTHESIS, then append 'only
if' and exactly one concrete necessary condition that the PREMISE neither states nor
rules out. This asserts an unsupported dependency and makes the result neutral.
Preserve every word and the polarity of the source prefix. Do not use may/maybe,
do not put the supported source after 'if', and avoid tautological conditions.""",
        "target": "neutral",
    },
    "composite_inv": {
        "mr_type": "inv",
        "family": "inv",
        "instruction": """Apply exactly two compatible invariant edits to the
SOURCE HYPOTHESIS, chosen from synonym replacement, unambiguous pronoun substitution,
and active/passive voice conversion. Preserve every fact and exact truth conditions.
Use two genuinely different edit types; if that is unsafe, emit no candidate.""",
        "target": "source",
    },
    "composite_flip": {
        "mr_type": "flip",
        "family": "flip",
        "instruction": """First make one conservative invariant edit to the SOURCE
HYPOTHESIS, then make exactly one decisive flip edit chosen from direct negation,
direct-antonym substitution, or a short explicit contradiction. Preserve entities,
roles, counts, and locations. The final result must contradict the PREMISE.""",
        "target": "contradiction",
    },
    "composite_neutral": {
        "mr_type": "neutral",
        "family": "neutral",
        "instruction": """First make one conservative invariant edit to the SOURCE
HYPOTHESIS, then add exactly one unsupported but plausible clause or unresolved
condition. Preserve the original event and entities. The final result must be
neutral relative to the PREMISE, natural, and free of generic boilerplate.""",
        "target": "neutral",
    },
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .!?\"'")


def content_tokens(text: str) -> set:
    tokens = set(re.findall(r"[a-z]+|[0-9]+", text.lower()))
    return tokens - CONTENT_STOPWORDS - NEGATION_TOKENS


def number_tokens(text: str) -> set:
    return set(re.findall(r"[0-9]+", text.lower()))


def has_negation(text: str) -> bool:
    lowered = text.lower().replace(chr(8217), chr(39))
    tokens = set(re.findall(r"[a-z]+", lowered))
    contraction = "n" + chr(39) + "t"
    return bool(tokens & NEGATION_TOKENS) or contraction in lowered


def has_safe_negation_scope(source_text: str, candidate_text: str) -> bool:
    source_norm = normalize_text(source_text)
    candidate_norm = normalize_text(candidate_text)
    source_is_existential = bool(re.match(
        r"^(a|an|some|someone|somebody|one)\b", source_norm
    ))
    if not source_is_existential:
        return True
    return bool(re.match(
        r"^(no|nobody|no one|there (?:is|are) no)\b", candidate_norm
    ))


def is_direct_negation(source_text: str, candidate_text: str) -> bool:
    if normalize_text(source_text).startswith(("here is ", "here are ")):
        return False
    if has_negation(source_text) or not has_negation(candidate_text):
        return False
    if not has_safe_negation_scope(source_text, candidate_text):
        return False
    source_content = content_tokens(source_text)
    candidate_content = content_tokens(candidate_text)
    return (
        number_tokens(source_text) == number_tokens(candidate_text)
        and len(source_content & candidate_content)
        >= max(1, len(source_content) - 1)
        and len(source_content ^ candidate_content) <= 2
    )



def is_safe_flip_source(source: Dict[str, Any]) -> bool:
    source_words = set(re.findall(r"[a-z]+", str(source["hypothesis"]).lower()))
    premise_words = set(re.findall(r"[a-z]+", str(source["premise"]).lower()))
    return not ((source_words & STRICT_QUANTIFIERS) - premise_words)


def has_voice_predicate(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    auxiliaries = {"is", "are", "was", "were", "be", "been", "being"}
    action_forms = {
        "wear", "wears", "wore", "worn", "wearing", "ride", "rides", "rode",
        "ridden", "riding", "carry", "carries", "carried", "carrying",
        "hold", "holds", "held", "holding", "chase", "chases", "chased",
        "eat", "eats", "ate", "eaten", "eating", "kick", "kicks", "kicked",
        "throw", "throws", "threw", "thrown", "pull", "pulls", "pulled",
        "push", "pushes", "pushed", "watch", "watches", "watched",
    }
    return bool(set(words) & action_forms) or (
        bool(set(words) & auxiliaries)
        and any(word.endswith(("ing", "ed", "en")) for word in words)
    )


def is_whitelisted_antonym_substitution(source_text: str, candidate_text: str) -> bool:
    source_words = re.findall(r"[a-z]+|[0-9]+", source_text.lower())
    candidate_words = re.findall(r"[a-z]+|[0-9]+", candidate_text.lower())
    if len(source_words) != len(candidate_words):
        return False
    differences = [
        (source_word, candidate_word)
        for source_word, candidate_word in zip(source_words, candidate_words)
        if source_word != candidate_word
    ]
    return (
        len(differences) == 1
        and frozenset(differences[0]) in ANTONYM_PAIRS
    )


def strip_think(text: str) -> str:
    return re.sub(
        r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE
    ).strip()


def extract_json(text: str) -> Any:
    """Extract the first JSON object/array without accepting arbitrary prose."""
    cleaned = strip_think(text)
    fence = re.fullmatch(
        r"\s*```(?:json)?\s*([\s\S]*?)\s*```\s*", cleaned, re.IGNORECASE
    )
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("response does not contain valid JSON")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            if not all(key in value for key in ("premise", "hypothesis", "label")):
                raise ValueError(f"{path}:{line_number} lacks an NLI field")
            if int(value["label"]) not in LABELS:
                raise ValueError(f"{path}:{line_number} has invalid label")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def stratified_sample(
    rows: Sequence[Dict[str, Any]], size: int, seed: int
) -> List[Dict[str, Any]]:
    if size >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    buckets = {label: [] for label in LABELS}
    for row in rows:
        buckets[int(row["label"])].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    base, remainder = divmod(size, len(buckets))
    selected: List[Dict[str, Any]] = []
    for position, label in enumerate(sorted(buckets)):
        take = base + (1 if position < remainder else 0)
        selected.extend(buckets[label][:take])
    rng.shuffle(selected)
    return selected


def operation_family(operation: str) -> str:
    return OPERATION_SPECS[operation]["family"]


def target_label(source: Dict[str, Any], operation: str) -> str:
    target = OPERATION_SPECS[operation]["target"]
    return LABELS[int(source["label"])] if target == "source" else target


def eligible(source: Dict[str, Any], operation: str) -> bool:
    family = operation_family(operation)
    if family == "inv":
        if operation == "voice_switch":
            return has_voice_predicate(str(source["hypothesis"]))
        return True
    if int(source["label"]) != LABEL_IDS["entailment"]:
        return False
    if family == "flip" and not is_safe_flip_source(source):
        return False
    if operation in {"negation_flip", "antonym_substitution"}:
        if has_negation(str(source["hypothesis"])):
            return False
        if (
            operation == "antonym_substitution"
            and re.match(
                r"^(a|an|some|someone|somebody|one)\b",
                normalize_text(str(source["hypothesis"])),
            )
        ):
            return False
        return True
    return True


def build_generation_prompt(
    source: Dict[str, Any], operation: str, candidate_count: int
) -> str:
    spec = OPERATION_SPECS[operation]
    target = target_label(source, operation)
    payload = {
        "premise": source["premise"],
        "source_hypothesis": source["hypothesis"],
        "source_label": LABELS[int(source["label"])],
        "target_label": target,
    }
    neutral_contract = ""
    exact_prefix = str(source["hypothesis"]).strip().rstrip(".!?")
    if operation in {"uninformative", "adding_contradiction", "conditional_clause"}:
        connectors = {
            "uninformative": ", and ",
            "adding_contradiction": ", but ",
            "conditional_clause": " only if ",
        }
        connector = connectors[operation]
        label = operation.upper()
        neutral_contract = (
            f"\n{label} FORMAT: Every hypothesis must begin exactly with:\n"
            + json.dumps(exact_prefix + connector, ensure_ascii=False)
            + "\nThen complete exactly one new clause; do not paraphrase the prefix.\n"
        )
        if operation == "adding_contradiction":
            neutral_contract += (
                "Each output object must also contain evidence_fact: the shortest "
                "exact contiguous quote from the PREMISE that the added clause "
                "explicitly negates. Example: premise 'Two dogs are jumping in a "
                "field', source 'The dogs are outside' -> hypothesis 'The dogs are "
                "outside, but the dogs are not jumping', evidence_fact 'dogs are "
                "jumping'. Do not quote the whole premise.\n"
            )
    return f"""TASK
{spec['instruction']}

PAIR
{json.dumps(payload, ensure_ascii=False, indent=2)}

OUTPUT CONTRACT
Return a JSON array containing up to {candidate_count} distinct objects.
Each object must have string fields "hypothesis" and "target_label".
For adding_contradiction it must also have the string field "evidence_fact";
other MRs must not add fields.
{{"hypothesis": "...", "target_label": "{target}"}}
The hypothesis must be one self-contained English statement with no labels,
quotes, commentary, prefixes, or prompt text. Never copy the premise verbatim.
{neutral_contract}

SNLI PITFALLS
- An extra fact absent from the premise is neutral, not contradiction.
- A less specific statement may be entailed but is not an exact paraphrase.
- Changing person, number, color, location, direction, or polarity is forbidden
  unless the named MR explicitly requires that exact change.
- A flip must negate a fact guaranteed by the premise; difference alone is neutral.
- A neutral edit must add uncertainty or one unsupported plausible fact without
  deleting the supported source event.
- Perform the named MR only. A valid target label does not excuse the wrong edit."""


def build_judge_prompt(
    source: Dict[str, Any], operation: str, candidates: Sequence[str]
) -> str:
    payload = {
        "premise": source["premise"],
        "source_hypothesis": source["hypothesis"],
        "source_label": LABELS[int(source["label"])],
        "requested_operation": operation,
        "operation_family": operation_family(operation),
        "operation_requirement": OPERATION_SPECS[operation]["instruction"],
        "target_label": target_label(source, operation),
        "candidates": [
            {"index": index, "hypothesis": text}
            for index, text in enumerate(candidates)
        ],
    }
    return f"""Evaluate every candidate in this payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}

For each candidate independently determine:
1. source_label: the strict SNLI label of premise -> source_hypothesis; do not
   trust the supplied source_label blindly (playing is not necessarily enjoying,
   setting a table is not seated, and multiple does not necessarily mean many);
2. label: the strict SNLI label of premise -> candidate hypothesis;
3. grammatical: whether the candidate is natural, self-contained English;
4. source_equivalent: whether candidate and source have identical truth conditions;
5. operation_valid: whether it performs the named MR exactly, including its edit
   count and structural constraints, with no shortcut or artifact. A correct NLI
   label alone is insufficient.

Return exactly one JSON object of this form and preserve candidate indices:
{{"evaluations":[{{"index":0,"source_label":"entailment","label":"entailment",
"grammatical":true,"source_equivalent":true,"operation_valid":true,"reason":"ok"}}]}}
Allowed reason values: ok, wrong_label, meaning_drift, grammar, artifact,
not_changed, or other. No prose outside JSON."""


def _completion(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def parse_candidates(
    raw: str,
    source: Dict[str, Any],
    operation: str,
    candidate_count: int,
) -> List[str]:
    family = operation_family(operation)
    if family == "flip" and not is_safe_flip_source(source):
        return []
    value = extract_json(raw)
    if not isinstance(value, list):
        raise ValueError("generator response is not a JSON array")
    expected = target_label(source, operation)
    source_norm = normalize_text(source["hypothesis"])
    premise_norm = normalize_text(source["premise"])
    maximum_words = max(30, len(source["hypothesis"].split()) * 2 + 8)
    source_content = content_tokens(str(source["hypothesis"]))
    premise_content = content_tokens(str(source["premise"]))
    minimum_overlap = max(1, (len(source_content) + 1) // 2)
    if operation == "negation_flip":
        minimum_overlap = max(1, (3 * len(source_content) + 3) // 4)
    elif operation == "antonym_substitution":
        minimum_overlap = max(1, len(source_content) - 1)
    candidates: List[str] = []
    seen = set()
    for item in value[:candidate_count]:
        if not isinstance(item, dict):
            continue
        text = item.get("hypothesis")
        declared = str(item.get("target_label", "")).strip().lower()
        if not isinstance(text, str) or declared != expected:
            continue
        text = re.sub(r"\s+", " ", text).strip()
        norm = normalize_text(text)
        if not norm or norm in seen or norm in {source_norm, premise_norm}:
            continue
        if not 3 <= len(text.split()) <= maximum_words:
            continue
        # MR-specific deterministic gates reject cheap shortcuts before judging.
        if operation == "uninformative" and not norm.startswith(
            source_norm + ", and "
        ):
            continue
        if operation == "adding_contradiction" and not norm.startswith(
            source_norm + ", but "
        ):
            continue
        if operation == "conditional_clause" and not norm.startswith(
            source_norm + " only if "
        ):
            continue
        if operation in {"uninformative", "conditional_clause"}:
            connector = ", and " if operation == "uninformative" else " only if "
            suffix = norm[len(source_norm + connector):]
            suffix_content = content_tokens(suffix) - REFERENCE_TOKENS
            if not (suffix_content - premise_content):
                continue
            if (
                operation == "conditional_clause"
                and suffix_content & VAGUE_CONDITION_TOKENS
            ):
                continue
            premise_words = set(re.findall(
                r"[a-z]+", str(source["premise"]).lower()
            ))
            source_words = set(re.findall(
                r"[a-z]+", str(source["hypothesis"]).lower()
            ))
            suffix_words = set(re.findall(r"[a-z]+", suffix))
            female_only = bool(
                premise_words & {"woman", "women", "girl", "girls", "female"}
            ) and not bool(
                premise_words & {"man", "men", "boy", "boys", "male"}
            )
            male_only = bool(
                premise_words & {"man", "men", "boy", "boys", "male"}
            ) and not bool(
                premise_words & {"woman", "women", "girl", "girls", "female"}
            )
            if (
                female_only
                and not (source_words & {"he", "him", "his"})
                and suffix_words & {"he", "him", "his"}
            ):
                continue
            if (
                male_only
                and not (source_words & {"she", "her", "hers"})
                and suffix_words & {"she", "her", "hers"}
            ):
                continue
        if operation == "adding_contradiction":
            evidence = item.get("evidence_fact")
            if not isinstance(evidence, str):
                continue
            evidence_norm = normalize_text(evidence)
            if not evidence_norm or evidence_norm not in premise_norm:
                continue
            suffix = norm[len(source_norm + ", but "):]
            evidence_content = content_tokens(evidence) - REFERENCE_TOKENS
            suffix_content = content_tokens(suffix) - REFERENCE_TOKENS
            if (
                not has_negation(suffix)
                or not evidence_content
                or len(evidence_content & suffix_content)
                < max(1, (len(evidence_content) + 1) // 2)
                or bool(suffix_content - evidence_content)
            ):
                continue
        candidate_content = content_tokens(text)
        source_pronouns = len(re.findall(
            r"(?i)\b(he|him|she|her|it|they|them|his|its|their)\b",
            str(source["hypothesis"]),
        ))
        candidate_pronouns = len(re.findall(
            r"(?i)\b(he|him|she|her|it|they|them|his|its|their)\b", text
        ))
        if operation == "pronoun_substitution":
            source_nonreference = source_content - REFERENCE_TOKENS
            candidate_nonreference = candidate_content - REFERENCE_TOKENS
            if (
                candidate_pronouns <= source_pronouns
                or bool(candidate_nonreference - source_nonreference)
                or len(source_nonreference - candidate_nonreference) > 2
            ):
                continue
        if operation == "voice_switch" and (
            bool(re.search(r"(?i)\bby\b", text))
            == bool(re.search(r"(?i)\bby\b", str(source["hypothesis"])))
        ):
            continue
        if (
            family in {"inv", "flip"}
            and source_content
            and len(source_content & candidate_content) < minimum_overlap
        ):
            continue
        if family == "inv" and (
            number_tokens(str(source["hypothesis"])) != number_tokens(text)
            or has_negation(str(source["hypothesis"])) != has_negation(text)
        ):
            continue
        if operation == "synonym_replacement" and len(
            source_content ^ candidate_content
        ) > 2:
            continue
        if operation == "negation_flip" and not is_direct_negation(
            str(source["hypothesis"]), text
        ):
            continue
        if operation == "antonym_substitution" and (
            has_negation(text)
            or number_tokens(str(source["hypothesis"])) != number_tokens(text)
            or not is_whitelisted_antonym_substitution(
                str(source["hypothesis"]), text
            )
        ):
            continue
        if family == "neutral" and source_content and len(
            source_content & candidate_content
        ) < minimum_overlap:
            continue
        if re.search(
            r"(?i)(premise:|hypothesis:|target_label|source_label|```|<think>|\b(?:1|one)\s+(?:are|were|have)\b|\b(?:likely|maybe|perhaps|probably|possibly)\b|\b\d+\b[^,.!?;]*\b\d+\s+(?:is|are|was|were|has|have)\b)",
            text,
        ):
            continue
        seen.add(norm)
        candidates.append(text)
    return candidates


def parse_judgments(raw: str, candidate_count: int) -> Dict[int, Dict[str, Any]]:
    value = extract_json(raw)
    if not isinstance(value, dict) or not isinstance(value.get("evaluations"), list):
        raise ValueError("judge response lacks evaluations")
    judgments: Dict[int, Dict[str, Any]] = {}
    for evaluation in value["evaluations"]:
        if not isinstance(evaluation, dict):
            continue
        index = evaluation.get("index")
        if not isinstance(index, int) or not 0 <= index < candidate_count:
            continue
        label = str(evaluation.get("label", "")).strip().lower()
        source_label = str(evaluation.get("source_label", "")).strip().lower()
        if label not in LABEL_IDS or source_label not in LABEL_IDS:
            continue
        judgments[index] = {
            "source_label": source_label,
            "label": label,
            "grammatical": evaluation.get("grammatical") is True,
            "source_equivalent": evaluation.get("source_equivalent") is True,
            "operation_valid": evaluation.get("operation_valid") is True,
            "reason": str(evaluation.get("reason", "other")),
        }
    return judgments


def choose_candidate(
    candidates: Sequence[str],
    judgments: Dict[int, Dict[str, Any]],
    source: Dict[str, Any],
    operation: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    expected = target_label(source, operation)
    for index, candidate in enumerate(candidates):
        judgment = judgments.get(index)
        if not judgment:
            continue
        # The curated SNLI source label is authoritative.  Gemma's independent
        # source_label is retained in the log as an audit signal, because using the
        # same small judge as a veto can wrongly reject valid candidates.
        if judgment["label"] != expected:
            continue
        if not judgment["grammatical"] or not judgment["operation_valid"]:
            continue
        if operation_family(operation) == "inv" and not judgment["source_equivalent"]:
            continue
        return candidate, judgment
    return None


def _generate_atomic(
    generator_client: OpenAI,
    judge_client: OpenAI,
    source: Dict[str, Any],
    operation: str,
    generator_model: str,
    judge_model: str,
    candidate_count: int,
    retries: int,
) -> Tuple[Optional[str], Dict[str, Any]]:
    log: Dict[str, Any] = {"operation": operation, "attempts": []}
    for attempt in range(1, retries + 1):
        attempt_log: Dict[str, Any] = {"attempt": attempt}
        try:
            raw_generation = _completion(
                generator_client,
                generator_model,
                GENERATOR_SYSTEM_PROMPT,
                build_generation_prompt(source, operation, candidate_count),
                temperature=0.65,
                max_tokens=420,
            )
            attempt_log["raw_generation"] = strip_think(raw_generation)[:3000]
            candidates = parse_candidates(
                raw_generation, source, operation, candidate_count
            )
            attempt_log["candidates"] = candidates
            if not candidates:
                attempt_log["status"] = "no_valid_json_candidate"
                log["attempts"].append(attempt_log)
                continue
            raw_judgment = _completion(
                judge_client,
                judge_model,
                JUDGE_SYSTEM_PROMPT,
                build_judge_prompt(source, operation, candidates),
                temperature=0.0,
                max_tokens=520,
            )
            attempt_log["raw_judgment"] = strip_think(raw_judgment)[:3000]
            judgments = parse_judgments(raw_judgment, len(candidates))
            attempt_log["judgments"] = judgments
            chosen = choose_candidate(candidates, judgments, source, operation)
            if chosen:
                attempt_log["status"] = "accepted"
                attempt_log["selected"] = chosen[0]
                log["attempts"].append(attempt_log)
                log["status"] = "accepted"
                return chosen[0], log
            attempt_log["status"] = "judge_rejected"
        except Exception as exc:  # preserve error evidence and retry a clean request
            attempt_log["status"] = "error"
            attempt_log["error"] = f"{type(exc).__name__}: {exc}"
        log["attempts"].append(attempt_log)
    log["status"] = "rejected"
    return None, log


COMPOSITE_STAGE_PLANS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "composite_inv": (
        ("pronoun_substitution", "synonym_replacement"),
        ("voice_switch", "synonym_replacement"),
        ("synonym_replacement", "pronoun_substitution"),
    ),
    "composite_flip": (
        ("pronoun_substitution", "negation_flip"),
        ("synonym_replacement", "negation_flip"),
        ("voice_switch", "adding_contradiction"),
    ),
    "composite_neutral": (
        ("pronoun_substitution", "uninformative"),
        ("synonym_replacement", "uninformative"),
        ("voice_switch", "uninformative"),
    ),
}

# The operations whose rows carry a component_mrs provenance field.
COMPOSITE_OPERATIONS = frozenset(COMPOSITE_STAGE_PLANS)


def _generate_composite(
    generator_client: OpenAI,
    judge_client: OpenAI,
    source: Dict[str, Any],
    operation: str,
    generator_model: str,
    judge_model: str,
    candidate_count: int,
    retries: int,
) -> Tuple[Optional[str], Dict[str, Any]]:
    log: Dict[str, Any] = {
        "operation": operation,
        "mode": "verified_two_stage",
        "attempts": [],
    }
    for plan_index, plan in enumerate(COMPOSITE_STAGE_PLANS[operation], 1):
        working = dict(source)
        plan_log: Dict[str, Any] = {
            "attempt": plan_index,
            "stages": list(plan),
            "stage_logs": [],
        }
        stage_failed = False
        for stage_operation in plan:
            if not eligible(working, stage_operation):
                plan_log["status"] = "ineligible_stage"
                plan_log["failed_stage"] = stage_operation
                stage_failed = True
                break
            hypothesis, stage_log = _generate_atomic(
                generator_client,
                judge_client,
                working,
                stage_operation,
                generator_model,
                judge_model,
                candidate_count,
                retries,
            )
            plan_log["stage_logs"].append(stage_log)
            if hypothesis is None:
                plan_log["status"] = "stage_rejected"
                plan_log["failed_stage"] = stage_operation
                stage_failed = True
                break
            working["hypothesis"] = hypothesis
            working["label"] = LABEL_IDS[target_label(working, stage_operation)]
        if stage_failed:
            log["attempts"].append(plan_log)
            continue

        final_hypothesis = str(working["hypothesis"])
        try:
            raw_judgment = _completion(
                judge_client,
                judge_model,
                JUDGE_SYSTEM_PROMPT,
                build_judge_prompt(source, operation, [final_hypothesis]),
                temperature=0.0,
                max_tokens=320,
            )
            plan_log["raw_final_judgment"] = strip_think(raw_judgment)[:3000]
            judgments = parse_judgments(raw_judgment, 1)
            plan_log["final_judgments"] = judgments
            chosen = choose_candidate(
                [final_hypothesis], judgments, source, operation
            )
            if chosen:
                plan_log["status"] = "accepted"
                plan_log["selected"] = final_hypothesis
                log["attempts"].append(plan_log)
                log["status"] = "accepted"
                log["applied_steps"] = list(plan)
                return final_hypothesis, log
            plan_log["status"] = "final_judge_rejected"
        except Exception as exc:
            plan_log["status"] = "error"
            plan_log["error"] = f"{type(exc).__name__}: {exc}"
        log["attempts"].append(plan_log)
    log["status"] = "rejected"
    return None, log


def component_mrs_from_log(
    operation: str, log: Dict[str, Any]
) -> Optional[List[str]]:
    """Which elementary MRs a composite row was composed from, else None for atomic rows.

    Raises when a composite row's ``applied_steps`` is missing or malformed rather than
    dropping the field: an unlabelled composite row would make the composite population
    silently heterogeneous, which is the exact problem this provenance field exists to fix.
    """
    return _component_mrs_from_log(
        operation,
        log,
        composite_operations=COMPOSITE_OPERATIONS,
        operation_specs=OPERATION_SPECS,
    )


def generate_one(
    generator_client: OpenAI,
    judge_client: OpenAI,
    source: Dict[str, Any],
    operation: str,
    generator_model: str,
    judge_model: str,
    candidate_count: int,
    retries: int,
) -> Tuple[Optional[str], Dict[str, Any]]:
    if operation in COMPOSITE_STAGE_PLANS:
        return _generate_composite(
            generator_client,
            judge_client,
            source,
            operation,
            generator_model,
            judge_model,
            candidate_count,
            retries,
        )
    return _generate_atomic(
        generator_client,
        judge_client,
        source,
        operation,
        generator_model,
        judge_model,
        candidate_count,
        retries,
    )


def source_record(source: Dict[str, Any], pair_id: int) -> Dict[str, Any]:
    row = dict(source)
    row.update(
        {
            "pair_id": pair_id,
            "is_syn": False,
            "weight": float(source.get("weight", 1.0)),
            "mr_id": "none",
            "mr_type": "inv",
        }
    )
    return row


def synthetic_record(
    source: Dict[str, Any],
    pair_id: int,
    operation: str,
    hypothesis: str,
    component_mrs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build one synthetic row.

    ``component_mrs`` is supplied only for composite operations and is written directly
    after ``mr_id``; atomic and source rows keep their original key set, so adding this
    field leaves every non-composite row byte-identical to artifacts generated before it
    existed.
    """
    spec = OPERATION_SPECS[operation]
    row: Dict[str, Any] = {
        "premise": source["premise"],
        "hypothesis": hypothesis,
        "label": LABEL_IDS[target_label(source, operation)],
        "pair_id": pair_id,
        "is_syn": True,
        "weight": 1.0,
        "mr_id": operation,
    }
    if component_mrs is not None:
        row[COMPONENT_MRS_FIELD] = list(component_mrs)
    row["mr_type"] = spec["mr_type"]
    return row


def summarize(
    selected: Sequence[Dict[str, Any]],
    output_rows: Sequence[Dict[str, Any]],
    logs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    synthetic = [row for row in output_rows if row.get("is_syn")]
    attempts = Counter(log["operation"] for log in logs)
    accepted = Counter(
        log["operation"] for log in logs if log.get("status") == "accepted"
    )
    duplicate_count = 0
    seen = set()
    for row in synthetic:
        key = (normalize_text(row["premise"]), normalize_text(row["hypothesis"]))
        if key in seen:
            duplicate_count += 1
        seen.add(key)
    operation_report = {}
    for operation in OPERATION_SPECS:
        total = attempts[operation]
        operation_report[operation] = {
            "attempted": total,
            "accepted": accepted[operation],
            "acceptance_rate": round(accepted[operation] / total, 4) if total else None,
        }
    return {
        "prompt_version": PROMPT_VERSION,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "config_fingerprint": config_fingerprint(
            PROMPT_VERSION, OPERATION_SPECS, COMPOSITE_STAGE_PLANS
        ),
        "source_count": len(selected),
        "synthetic_count": len(synthetic),
        "source_label_counts": dict(Counter(int(row["label"]) for row in selected)),
        "synthetic_label_counts": dict(
            Counter(int(row["label"]) for row in synthetic)
        ),
        "operation_results": operation_report,
        "duplicate_synthetic_rows": duplicate_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_DIR / "labeled_data.json")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_DATA_DIR / "augmented_data_all_label_mrs_sample.json"
    )
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--all", action="store_true", help="process every source row")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator-model", default="gemma-3-4b-it-qat")
    parser.add_argument("--judge-model", default="gemma-3-4b-it-qat")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--operations", default=",".join(OPERATION_SPECS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel Gemma requests within each source row",
    )
    parser.add_argument(
        "--min-total-rows",
        type=int,
        default=0,
        help="after the all-MR pass, generate unique refill rows until this total",
    )
    parser.add_argument("--max-fill-rounds", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.generator_model != ALLOWED_MODEL or args.judge_model != ALLOWED_MODEL:
        raise ValueError(
            "Both generator and judge must use gemma-3-4b-it-qat; "
            "other LLMs are disabled for this pipeline"
        )
    from MRAugment import NLIMRTool

    registered_mrs = list(NLIMRTool.LABEL_MRS)
    configured_mrs = list(OPERATION_SPECS)
    if configured_mrs != registered_mrs:
        raise RuntimeError(
            "All-MR configuration is out of sync with NLIMRTool.LABEL_MRS: "
            f"configured={configured_mrs}, registered={registered_mrs}"
        )
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"{args.output} exists; pass --overwrite or --resume explicitly"
        )
    if (
        args.candidate_count < 1
        or args.retries < 1
        or args.checkpoint_every < 1
        or args.max_fill_rounds < 1
        or args.workers < 1
        or args.min_total_rows < 0
    ):
        raise ValueError("generation, checkpoint, and refill values must be positive")
    if args.min_total_rows and not args.all:
        raise ValueError("--min-total-rows requires --all")
    operations = [item.strip() for item in args.operations.split(",") if item.strip()]
    unknown = set(operations) - set(OPERATION_SPECS)
    if unknown:
        raise ValueError(f"unknown operations: {sorted(unknown)}")

    source_rows = read_jsonl(args.input)
    selected = (
        list(source_rows)
        if args.all
        else stratified_sample(source_rows, args.sample_size, args.seed)
    )
    base_url = f"http://{args.host}:{args.port}/v1"
    generator_client = OpenAI(api_key="local", base_url=base_url, timeout=120.0)
    judge_client = OpenAI(api_key="local", base_url=base_url, timeout=120.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".generation_log.jsonl")
    if args.resume:
        if not args.output.exists() or not log_path.exists():
            raise FileNotFoundError("resume requires both output and generation log")
        output_rows = read_jsonl(args.output)
        with log_path.open(encoding="utf-8") as handle:
            logs = [json.loads(line) for line in handle if line.strip()]
    else:
        output_rows = []
        logs = []
        write_jsonl(args.output, output_rows)
        write_jsonl(log_path, logs)
    completed_pair_ids = {
        int(row["pair_id"]) for row in output_rows if row.get("is_syn") is not True
    }
    synthetic_keys = {
        (normalize_text(row["premise"]), normalize_text(row["hypothesis"]))
        for row in output_rows
        if row.get("is_syn") is True
    }
    original_pair_ids = {
        id(row): int(row.get("pair_id", index))
        for index, row in enumerate(source_rows)
    }
    selected_pair_ids = {original_pair_ids[id(row)] for row in selected}
    unexpected = completed_pair_ids - selected_pair_ids
    if unexpected:
        raise ValueError(f"resume output contains unexpected pair_ids: {sorted(unexpected)[:5]}")
    logs = [log for log in logs if int(log.get("pair_id", -1)) in completed_pair_ids]
    resumed_count = len(completed_pair_ids)

    started = time.time()
    executor = (
        ThreadPoolExecutor(max_workers=args.workers)
        if args.workers > 1
        else None
    )
    try:
        for selected_index, source in enumerate(selected, 1):
            pair_id = original_pair_ids[id(source)]
            if pair_id in completed_pair_ids:
                print(
                    f"[{selected_index}/{len(selected)}] pair_id={pair_id} (resumed)",
                    flush=True,
                )
                continue
            output_rows.append(source_record(source, pair_id))
            eligible_operations = [
                operation
                for operation in operations
                if eligible(source, operation)
            ]
            if executor is None:
                generated = [
                    (
                        operation,
                        generate_one(
                            generator_client,
                            judge_client,
                            source,
                            operation,
                            args.generator_model,
                            args.judge_model,
                            args.candidate_count,
                            args.retries,
                        ),
                    )
                    for operation in eligible_operations
                ]
            else:
                futures = {
                    operation: executor.submit(
                        generate_one,
                        generator_client,
                        judge_client,
                        source,
                        operation,
                        args.generator_model,
                        args.judge_model,
                        args.candidate_count,
                        args.retries,
                    )
                    for operation in eligible_operations
                }
                generated = [
                    (operation, futures[operation].result())
                    for operation in eligible_operations
                ]
            for operation, (hypothesis, generation_log) in generated:
                generation_log.update(
                    {
                        "pair_id": pair_id,
                        "source_label": int(source["label"]),
                        "target_label": target_label(source, operation),
                    }
                )
                if hypothesis:
                    synthetic_key = (
                        normalize_text(source["premise"]),
                        normalize_text(hypothesis),
                    )
                    if synthetic_key in synthetic_keys:
                        generation_log["status"] = "duplicate_rejected"
                        generation_log["duplicate_hypothesis"] = hypothesis
                        hypothesis = None
                    else:
                        synthetic_keys.add(synthetic_key)
                logs.append(generation_log)
                if hypothesis:
                    output_rows.append(
                        synthetic_record(
                            source,
                            pair_id,
                            operation,
                            hypothesis,
                            component_mrs=component_mrs_from_log(operation, generation_log),
                        )
                    )
            print(
                f"[{selected_index}/{len(selected)}] pair_id={pair_id}",
                flush=True,
            )
            if (
                selected_index % args.checkpoint_every == 0
                or selected_index == len(selected)
            ):
                write_jsonl(log_path, logs)
                write_jsonl(args.output, output_rows)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    fill_attempts = 0
    fill_accepted = 0
    fill_rounds_completed = 0
    fill_order = (
        "uninformative",
        "conditional_clause",
        "negation_flip",
        "pronoun_substitution",
        "synonym_replacement",
        "antonym_substitution",
        "voice_switch",
        "adding_contradiction",
        "composite_neutral",
        "composite_inv",
        "composite_flip",
    )
    if args.min_total_rows and len(output_rows) < args.min_total_rows:
        print(
            f"primary pass produced {len(output_rows)} rows; "
            f"refilling to {args.min_total_rows}",
            flush=True,
        )
        for fill_round in range(1, args.max_fill_rounds + 1):
            fill_rounds_completed = fill_round
            for operation in fill_order:
                if operation not in operations:
                    continue
                for source in selected:
                    if len(output_rows) >= args.min_total_rows:
                        break
                    if not eligible(source, operation):
                        continue
                    pair_id = original_pair_ids[id(source)]
                    hypothesis, generation_log = generate_one(
                        generator_client,
                        judge_client,
                        source,
                        operation,
                        args.generator_model,
                        args.judge_model,
                        args.candidate_count,
                        args.retries,
                    )
                    fill_attempts += 1
                    generation_log.update(
                        {
                            "pair_id": pair_id,
                            "source_label": int(source["label"]),
                            "target_label": target_label(source, operation),
                            "fill_round": fill_round,
                        }
                    )
                    if hypothesis:
                        synthetic_key = (
                            normalize_text(source["premise"]),
                            normalize_text(hypothesis),
                        )
                        if synthetic_key in synthetic_keys:
                            generation_log["status"] = "duplicate_rejected"
                            generation_log["duplicate_hypothesis"] = hypothesis
                            hypothesis = None
                        else:
                            synthetic_keys.add(synthetic_key)
                    logs.append(generation_log)
                    if hypothesis:
                        output_rows.append(
                            synthetic_record(
                                source,
                                pair_id,
                                operation,
                                hypothesis,
                                component_mrs=component_mrs_from_log(operation, generation_log),
                            )
                        )
                        fill_accepted += 1
                        if (
                            fill_accepted % 25 == 0
                            or len(output_rows) >= args.min_total_rows
                        ):
                            print(
                                f"refill round={fill_round} accepted={fill_accepted} "
                                f"total_rows={len(output_rows)}/{args.min_total_rows}",
                                flush=True,
                            )
                    if fill_attempts % args.checkpoint_every == 0:
                        write_jsonl(log_path, logs)
                        write_jsonl(args.output, output_rows)
                if len(output_rows) >= args.min_total_rows:
                    break
            write_jsonl(log_path, logs)
            write_jsonl(args.output, output_rows)
            if len(output_rows) >= args.min_total_rows:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output_rows)
    log_path = args.output.with_suffix(".generation_log.jsonl")
    write_jsonl(log_path, logs)
    report = summarize(selected, output_rows, logs)
    report.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            # argv without cwd is not reproducible: --input/--output are commonly relative.
            "argv": list(sys.argv),
            "cwd": str(Path.cwd()),
            "effective_args": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "generator_model": args.generator_model,
            "judge_model": args.judge_model,
            "label_mrs": list(OPERATION_SPECS),
            "host": args.host,
            "port": args.port,
            "base_url": base_url,
            "operations": operations,
            "all": args.all,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "candidate_count": args.candidate_count,
            "retries": args.retries,
            "checkpoint_every": args.checkpoint_every,
            "workers": args.workers,
            "min_total_rows": args.min_total_rows,
            "target_reached": (
                not args.min_total_rows or len(output_rows) >= args.min_total_rows
            ),
            "fill_attempts": fill_attempts,
            "fill_accepted": fill_accepted,
            "fill_rounds_completed": fill_rounds_completed,
            "resumed_source_count": resumed_count,
            "elapsed_seconds": round(time.time() - started, 2),
            # Hashed here rather than earlier: the artifact and log are written above, and
            # the input's identity was previously recorded nowhere at all.
            "input_sha256": sha256_file(args.input),
            "output_sha256": sha256_file(args.output),
            "generation_log_sha256": sha256_file(log_path),
            "generator_source_sha256": sha256_file(Path(__file__).resolve()),
        }
    )
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"generation log: {log_path}")
    print(f"report: {report_path}")
    if args.min_total_rows and len(output_rows) < args.min_total_rows:
        print(
            f"ERROR: only {len(output_rows)} rows after "
            f"{args.max_fill_rounds} refill rounds",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
