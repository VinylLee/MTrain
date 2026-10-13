"""Shared v3.4 generation contract. Only the same Gemma generator/judge is used.

External NLI models and hypothesis-only probes are deliberately absent. Corpus
identity and export schema never enter prompts, style selection or candidate ranking.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

import augment_snli_all_label_mrs as legacy
from .data import digest, normalize, source_uid
from .protocol import (parse_judgments, exception_fields, replay_atomic,
                       parse_equivalence_judgments, choose_invariant_candidate)
from .invariants import valid_invariant_tokens, valid_synonym_edit, has_actual_pronoun_replacement

PROMPT_VERSION = "english-nli-pair-aware-all-label-mrs-v3.4"
ALLOWED_MODEL = legacy.ALLOWED_MODEL
LABELS = legacy.LABELS
LABEL_IDS = legacy.LABEL_IDS
OPERATION_SPECS = deepcopy(legacy.OPERATION_SPECS)
COMPOSITE_STAGE_PLANS = deepcopy(legacy.COMPOSITE_STAGE_PLANS)
STYLE_POOL = (
    {"id": "plain", "instruction": "Prefer plain common words and a concise declarative sentence."},
    {"id": "coordinate", "instruction": "Use natural coordination where the operation allows it; avoid a fixed connector."},
    {"id": "subordinate", "instruction": "Use a natural subordinate or relative clause where allowed; keep the edit minimal."},
    {"id": "descriptive", "instruction": "Prefer a natural descriptive construction where allowed; do not add facts to invariant edits."},
    {"id": "compact", "instruction": "Avoid filler; express the required edit compactly while preserving every required fact."},
    {"id": "fluent", "instruction": "Prefer fluent everyday English over repetitive sentence templates."},
)
OPERATION_SPECS["synonym_replacement"]["instruction"] += """ Playing does not mean
enjoying, walking does not mean strolling happily, and 'a man' does not mean
'someone'. Do not add mental states or remove stated gender/age."""
OPERATION_SPECS["synonym_replacement"]["instruction"] += """ Do not weaken a specific
action into broad engagement or participation; keep the same event and specificity."""
OPERATION_SPECS["synonym_replacement"]["instruction"] += """ Change exactly one or two
word positions and copy all other words and punctuation. Copy digit and spelled-out
quantities, names, quantifiers, frequency words and modality exactly. Do not omit
'always', 'every other', 'only', or similar qualifications. Do not substitute a
broader category (hotel/lodging, lying/untruthful) unless exact contextual meaning
is identical. A contextual synonym must preserve specificity in both directions."""
OPERATION_SPECS["uninformative"]["instruction"] = """Retain the entire supported SOURCE
HYPOTHESIS proposition and add exactly one specific plausible detail that the PREMISE
neither states nor rules out. The combined statement must be neutral. Added facts may
be affirmative or negative when both possibilities are genuinely unresolved. Avoid
generic weather/time boilerplate, uncertainty adverbs and internal contradictions.
Do not use a connector solely because it signals a label."""
OPERATION_SPECS["uninformative"]["instruction"] += """ Prefer a detail independent of
the supported action. Routine activity consequences (such as focusing on a screen
while playing a computer game) are poor unresolved additions. Choose a clearly
unstated attribute, preference, possession or background fact about the same subject."""
OPERATION_SPECS["adding_contradiction"]["instruction"] = """Retain the entire SOURCE
HYPOTHESIS proposition and add exactly one claim incompatible with a DIFFERENT fact
guaranteed by the PREMISE but absent from the source hypothesis. Keep the statement
internally coherent. The new clause can use an affirmative incompatible state or
explicit negation, whichever is logically warranted. Unmentioned facts are not
contradictions. Quote the exact premise evidence in evidence_fact."""
OPERATION_SPECS["conditional_clause"]["instruction"] = """Express the SOURCE
HYPOTHESIS with one concrete unsupported NECESSARY condition C. Preserve H => C,
never reverse it to C => H. Use a natural equivalent of 'H only if C', for example
'H only when C', 'H only on condition that C', or a grammatical statement that C
is necessary for H. C must be neither guaranteed nor ruled out by the PREMISE.
Keep the source proposition unchanged. Avoid tautologies, vague conditions, mere
possibility and any sufficient-condition interpretation."""
OPERATION_SPECS["conditional_clause"]["instruction"] += """ Avoid requirements already
implied by the activity, such as a functioning computer for computer gameplay.
Choose one specific independent condition, not a rewording of the premise."""

SYSTEM = """You edit English NLI examples conservatively. The premise is the only
known world. Entailment means guaranteed true, contradiction means guaranteed false,
neutral means neither guaranteed nor ruled out. Perform only the requested operation.
Generate natural varied English, not label-specific boilerplate. Added unsupported
negative facts may be neutral; an affirmative incompatible fact may contradict.
For invariant edits preserve exact truth conditions, not just the NLI label.
An invariant source may be neutral or contradicted by the premise: preserve its
own meaning anyway. Never repair that hypothesis to make it entailed by the premise.
If the operation is unsafe, return []. Return only the requested JSON array."""
JUDGE_SYSTEM = """You are a conservative English NLI quality judge.
Classify each hypothesis from the factual premise alone. Contradiction requires
logical incompatibility, not merely missing or unlikely information. Entailment
requires that every asserted fact is guaranteed. Otherwise choose neutral.
Judge grammar and the requested edit operation independently of the NLI label.
An INV may preserve a source that is neutral or contradicted; never repair it to
be entailed. Hypothesis equivalence is evaluated in a separate isolated pass.
Return JSON only, no hidden reasoning; reason is a short diagnostic tag."""
EQUIVALENCE_SYSTEM = """Judge a conservative invariant text edit. You see only the
original text and its rewritten candidates, with NO external factual premise.
Assume the original text is true to check original -> rewritten entailment, then
assume the rewritten text is true to check rewritten -> original entailment.
Exact equivalence requires BOTH directions. Preserve all qualifications, entities,
roles, counts, gender, tense, aspect and polarity. Same topic is insufficient.
Judge the requested operation independently. Output JSON only with short reason tags."""
EQUIVALENCE_SYSTEM += """ A requested operation is NOT evidence that it was obeyed.
Read the two actual strings; never assume a replacement is a synonym.
Calibration examples (both directions matter):
original='The bell is loud.', rewritten='The bell is quiet.': contradiction,
contradiction, equivalent=false.
original='He always helps.', rewritten='He helps.': entailment, neutral,
equivalent=false because the reverse does not guarantee 'always'.
original='The men carry a box.', rewritten='A box is carried by the men.':
entailment, entailment, equivalent=true. These are examples, not your verdicts."""


def contract():
    return {"prompt_version": PROMPT_VERSION, "operations": OPERATION_SPECS,
            "composite_plans": COMPOSITE_STAGE_PLANS, "styles": STYLE_POOL,
            "system": SYSTEM, "judge_system": JUDGE_SYSTEM, "equivalence_system": EQUIVALENCE_SYSTEM,
            "candidate_count": 3, "rounds": 2, "model": ALLOWED_MODEL,
            "quota_policy": "equal_followup_labels_E_N_C_v1",
            "judge_protocol": "same_gemma_INV_HH_only_other_MRs_PH_v2",
            "inv_label_rule": "strict_bidirectional_equivalence_preserves_source_gold",
            "inv_judge_calls_per_attempt": 1, "other_judge_calls_per_attempt": 1,
            "invariant_parser": "aligned_one_or_two_synonyms_numeric_words_actual_np_v2",
            "external_nli_in_production": False}


def fingerprint():
    return digest(contract())


def eligible(source, operation):
    return legacy.eligible(source, operation)


def target_label(source, operation):
    return legacy.target_label(source, operation)


def style_for(source, operation, round_index, seed=42):
    key = digest([source_uid(source), operation, round_index, seed, PROMPT_VERSION])
    return STYLE_POOL[int(key[:12], 16) % len(STYLE_POOL)]


def build_generation_prompt(source, operation, round_index=0, seed=42):
    style = style_for(source, operation, round_index, seed)
    payload = {"premise": source["premise"], "source_hypothesis": source["hypothesis"],
               "source_label": LABELS[source["label"]], "target_label": target_label(source, operation)}
    addition = ""
    if operation in {"uninformative", "adding_contradiction"}:
        addition = """For each object also provide edit_plan with string fields
added_clause and connector. The code copies the exact source hypothesis and joins
the added clause. Do not reproduce or edit the source clause. Choose a grammatical
connector such as ' and ', ' while ',
', and ', ', while ', ', but ', ' but ', '; ', ' yet ', ' although ',
' even though ', ', although ', or ', even though '. These are a shared pool for
both supported and unsupported edits; use only semantically natural combinations.
Do not infer a label from a connector. Added negative facts need not contradict;
added affirmative facts can contradict. Keep the addition concrete and short.
For adding_contradiction also provide evidence_fact, an exact contiguous quote
from the premise supporting the fact contradicted by the addition."""
    elif operation == "conditional_clause":
        addition = """For each object also provide edit_plan with string fields
added_clause (condition C),
and condition_direction='H_implies_C', plus connector chosen from ' only if ',
' only when ', or ' only on condition that '. This is a NECESSARY condition.
For example 'If C, H', 'H if C', 'H provided that C' are NOT acceptable: these
reverse the direction or ambiguously express a sufficient condition."""
    if addition:
        example = {"target_label": target_label(source, operation), "edit_plan": {
            "added_clause": "<one concrete clause about the same situation>",
            "connector": " only when " if operation == "conditional_clause" else " and "}}
        if operation == "conditional_clause":
            example["edit_plan"]["condition_direction"] = "H_implies_C"
        if operation == "adding_contradiction":
            example["evidence_fact"] = "<exact quote from premise>"
        output_contract = ("Return up to 3 objects with EXACTLY this structure. "
            "Do not output hypothesis; the code realizes the plan.\n" + json.dumps(example))
    else:
        output_contract = "Return up to 3 objects with hypothesis and target_label."
    return f"""OPERATION: {operation}
{OPERATION_SPECS[operation]['instruction']}

PAIR
{json.dumps(payload, ensure_ascii=False)}

STYLE PREFERENCE (never override the operation): {style['instruction']}

{output_contract}
{addition}
Preserve entities, roles, tense, number and polarity unless this MR explicitly changes
one of them. No reasoning, labels inside text, prompt fragments or premise copying.
Do not make an invariant edit longer by adding information. Do not use mere punctuation
changes or weakening/specializing words as synonyms. No safe edit => []."""


def build_judge_prompt(source, operation, candidates):
    payload = {"premise": source["premise"], "source_hypothesis": source["hypothesis"],
               "source_label": LABELS[source["label"]], "requested_operation": operation,
               "operation_requirement": OPERATION_SPECS[operation]["instruction"],
               "target_label": target_label(source, operation),
               "candidates": [{"index": i, "hypothesis": t} for i, t in enumerate(candidates)]}
    return f"""Judge all candidates independently:
{json.dumps(payload, ensure_ascii=False)}
Classify premise -> candidate using strict NLI. Extra unmentioned facts, including
negated ones, are neutral unless disproven. An affirmative incompatible fact may
contradict. Do not use the connector as evidence for a label. For conditional edits
check the necessary direction H => C; reject sufficient conditions C => H.
For INV judge only the PREMISE -> source and PREMISE -> candidate labels here;
a separate pass checks source/candidate equivalence. Do not conflate these tasks.
Preserve roles, number, quantifiers, tense and polarity in the requested edit.
Adding a pronoun without replacing a noun phrase is NOT pronoun_substitution.
Changing an antonym or broadly related word is NOT synonym_replacement.
For adding_contradiction the addition must contradict a different premise fact,
not the retained source proposition. All edits must be grammatical and substantive.
Preserve an original grammatical fragment as a fragment when the MR permits it;
do not add facts to turn it into a complete sentence.
Return an object with key evaluations, a list with exactly one entry for EACH
candidate index supplied above. Do not create an entry for the source hypothesis.
Each entry has: index (integer copied from candidate), source_label (your strict
PREMISE -> source classification), label (PREMISE -> this candidate), grammatical
(boolean), operation_valid (boolean), source_equivalent=false (reserved for the
separate equivalence pass), reason (short tag).
Each NLI label must be exactly one of entailment, neutral, contradiction.
No preset example labels, no prose."""


def build_equivalence_prompt(source, operation, candidates):
    # Deliberately exclude factual premise, gold labels, target labels and role.
    payload = {"operation":operation,"operation_requirement":OPERATION_SPECS[operation]["instruction"],
               "original_text":source["hypothesis"],
               "candidates":[{"index":i,"rewritten_text":text} for i,text in enumerate(candidates)]}
    return f"""{json.dumps(payload, ensure_ascii=False)}
For each supplied candidate, compare ONLY original_text and rewritten_text.
source_to_candidate: take original_text as the premise and rewritten_text as hypothesis.
candidate_to_source: take rewritten_text as the premise and original_text as hypothesis.
Use strict NLI labels entailment, neutral, contradiction for each direction.
Preserve even information that might be false in the real world. No outside text
or world facts are part of these comparisons. Dropping 'always', an object, gender,
a quantifier, or a frequency is not equivalence. Broadly related words are not
exact synonyms. Adding a pronoun without replacing a noun phrase is not that MR.
Keep a grammatical original fragment as a fragment rather than inventing facts.
Return {{"evaluations":[...]}} with one entry per supplied candidate index:
index (integer), source_to_candidate (NLI label), candidate_to_source (NLI label),
source_equivalent (boolean, true only if both directions entailment), grammatical
(boolean), operation_valid (boolean), reason (short tag). Do not evaluate an
unsupplied source candidate or assume any example verdict."""


def judge_current(client, source, operation, candidates, trace, max_tokens=600):
    if OPERATION_SPECS[operation]["family"] == "inv":
        prompt = build_equivalence_prompt(source, operation, candidates)
        raw = trace.get("raw_equivalence_judgment")
        if raw is None:
            trace.setdefault("model_call_events", []).append({"role":"invariant_equivalence_judge", "max_tokens":max_tokens})
            raw = _call(client, EQUIVALENCE_SYSTEM, prompt, 0.0, max_tokens)
        trace["equivalence_prompt_sha256"] = digest([EQUIVALENCE_SYSTEM, prompt])
        trace["raw_equivalence_judgment"] = raw
        trace["raw_judgment"] = raw
        trace["label_binding"] = "INV_equivalence_preserves_source_gold"
        equivalent = parse_equivalence_judgments(raw, len(candidates))
        trace["equivalence_judgments"] = equivalent
        return equivalent
    prompt = build_judge_prompt(source, operation, candidates)
    raw = trace.get("raw_judgment")
    if raw is None:
        trace.setdefault("model_call_events", []).append({"role":"pair_label_judge", "max_tokens":max_tokens})
        raw = _call(client, JUDGE_SYSTEM, prompt, 0.0, max_tokens)
    trace["judge_prompt_sha256"] = digest([JUDGE_SYSTEM, prompt])
    trace["raw_judgment"] = raw
    return parse_judgments(raw, len(candidates))


def choose_current(candidates, judgments, source, operation):
    if OPERATION_SPECS[operation]["family"] == "inv":
        return choose_invariant_candidate(candidates, judgments, source, operation)
    return legacy.choose_candidate(candidates, judgments, source, operation)


CONNECTORS = {" and ", " while ", ", and ", ", while ", ", but ", " but ",
              "; ", " yet ", " although ", " even though ", ", although ", ", even though "}


def parse_candidates(raw, source, operation):
    value = legacy.extract_json(raw)
    if not isinstance(value, list):
        raise ValueError("generator response must be a JSON array")
    if operation not in {"uninformative", "adding_contradiction", "conditional_clause"}:
        if operation == "synonym_replacement":
            candidates = []
            for item in value[:3]:
                if not isinstance(item, dict) or item.get("target_label") != target_label(source, operation):
                    continue
                text = item.get("hypothesis")
                if isinstance(text, str) and valid_synonym_edit(source["hypothesis"], text):
                    if normalize(text) != normalize(source["premise"]) and text not in candidates:
                        candidates.append(text.strip())
        else:
            candidates = legacy.parse_candidates(raw, source, operation, 3)
        if OPERATION_SPECS[operation]["family"] == "inv":
            candidates = [t for t in candidates if valid_invariant_tokens(source["hypothesis"], t)]
        if operation == "pronoun_substitution":
            candidates = [t for t in candidates if has_actual_pronoun_replacement(source["hypothesis"], t)]
        return [t for t in candidates if normalize(t) != normalize(source["hypothesis"])]
    accepted = []
    for item in value[:3]:
        if not isinstance(item, dict) or item.get("target_label") != target_label(source, operation):
            continue
        plan = item.get("edit_plan")
        if not isinstance(plan, dict):
            continue
        added = plan.get("added_clause")
        if not isinstance(added, str) or not added.strip():
            continue
        connector = plan.get("connector")
        if operation == "conditional_clause":
            if plan.get("condition_direction") != "H_implies_C" or connector not in {
                " only if ", " only when ", " only on condition that "}:
                continue
        elif connector not in CONNECTORS:
            continue
        text = source["hypothesis"].strip().rstrip(".!?") + connector + added.strip().rstrip(".!?") + "."
        if normalize(text) in {normalize(source["hypothesis"]), normalize(source["premise"])}:
            continue
        if not 3 <= len(text.split()) <= max(45, len(source["hypothesis"].split()) * 2 + 16):
            continue
        if re.search(r"(?i)(premise:|hypothesis:|target_label|source_label|```|<think>)", text):
            continue
        if operation == "adding_contradiction":
            evidence = item.get("evidence_fact")
            if not isinstance(evidence, str) or not evidence.strip() or evidence not in source["premise"]:
                continue
        else:
            if not (legacy.content_tokens(added) - legacy.content_tokens(source["premise"]) - legacy.REFERENCE_TOKENS):
                continue
        if normalize(text) not in {normalize(t) for t in accepted}:
            accepted.append(text.strip())
    return accepted


def _call(client, system, prompt, temperature, max_tokens):
    return legacy._completion(client, ALLOWED_MODEL, system, prompt, temperature, max_tokens)


def generate_atomic(client, source, operation, seed=42, prior_trace=None):
    log = deepcopy(prior_trace) if prior_trace else {"operation": operation, "source_uid": source_uid(source), "attempts": []}
    if prior_trace:
        text, log = replay_atomic(source, operation, log, parse_candidates,
                                 log.get("selected_hypothesis") if log.get("status") == "accepted" else None,
                                 OPERATION_SPECS[operation]["family"] == "inv")
        if text:
            return text, log
    for round_index in range(len(log["attempts"]), 2):
        prompt = build_generation_prompt(source, operation, round_index, seed)
        attempt = {"round": round_index, "style_id": style_for(source, operation, round_index, seed)["id"],
                   "prompt_sha256": digest([SYSTEM, prompt])}
        try:
            attempt.setdefault("model_call_events", []).append({"role":"generation", "max_tokens":700})
            raw = _call(client, SYSTEM, prompt, 0.7, 700)
            attempt["raw_generation"] = raw
            candidates = parse_candidates(raw, source, operation)
            attempt["candidates"] = candidates
            if candidates:
                judgments = judge_current(client, source, operation, candidates, attempt)
                attempt["judgments"] = judgments
                chosen = choose_current(candidates, judgments, source, operation)
                if chosen:
                    attempt["status"] = "accepted"
                    log["attempts"].append(attempt)
                    log.update(status="accepted", selected_hypothesis=chosen[0])
                    return chosen[0], log
            attempt["status"] = "rejected"
        except Exception as exc:
            attempt.update(exception_fields(exc))
        log["attempts"].append(attempt)
    log["status"] = "rejected"
    return None, log


def generate_one(client, source, operation, seed=42, prior_trace=None):
    if not eligible(source, operation):
        return None, {"status": "ineligible", "operation": operation}
    if operation not in COMPOSITE_STAGE_PLANS:
        return generate_atomic(client, source, operation, seed, prior_trace) if prior_trace else generate_atomic(client, source, operation, seed)
    # One deterministic valid plan, avoiding hidden extra retries across plans.
    plans = COMPOSITE_STAGE_PLANS[operation]
    available = [p for p in plans if eligible(source, p[0])]
    if not available:
        return None, {"status": "ineligible", "operation": operation}
    index = int(digest([source_uid(source), operation, seed])[:8], 16) % len(available)
    plan = tuple(prior_trace["applied_steps"]) if prior_trace else available[index]
    if plan not in COMPOSITE_STAGE_PLANS[operation]:
        raise ValueError("saved composite plan does not belong to its MR")
    working = dict(source)
    log = deepcopy(prior_trace) if prior_trace else {"operation": operation, "applied_steps": list(plan)}
    old_stages = log.get("stage_logs", [])
    log["stage_logs"] = []
    for stage_index, stage in enumerate(plan):
        if not eligible(working, stage):
            log["status"] = "ineligible_stage"
            return None, log
        prior = old_stages[stage_index] if stage_index < len(old_stages) else None
        text, stage_log = generate_atomic(client, working, stage, seed, prior) if prior else generate_atomic(client, working, stage, seed)
        log["stage_logs"].append(stage_log)
        if text is None:
            log["status"] = "rejected"
            return None, log
        working["label"] = LABEL_IDS[target_label(working, stage)]
        working["hypothesis"] = text
    text = working["hypothesis"]
    if normalize(text) == normalize(source["hypothesis"]):
        log["status"] = "no_effect"
        return None, log
    try:
        final_trace = log.setdefault("final_judge_trace", {})
        judgments = judge_current(client, source, operation, [text], final_trace, max_tokens=350)
        log["final_judge_trace"] = final_trace
        log["raw_final_judgment"] = final_trace["raw_judgment"]
        chosen = choose_current([text], judgments, source, operation)
        log["final_judgments"] = judgments
        if chosen:
            log.update(status="accepted", selected_hypothesis=text)
            return text, log
    except Exception as exc:
        log.update(exception_fields(exc))
    log["status"] = "rejected"
    return None, log


def generate_legacy_atomic(client, source, operation, prior_trace=None):
    """Keep v3.3 prompts/filters and budget, with the common response-envelope repair."""
    log = deepcopy(prior_trace) if prior_trace else {"operation": operation, "attempts": []}
    if prior_trace:
        parser = lambda raw, src, op: legacy.parse_candidates(raw, src, op, 3)
        text, log = replay_atomic(source, operation, log, parser,
                                 log.get("selected_hypothesis") if log.get("status") == "accepted" else None)
        if text:
            return text, log
    for round_index in range(len(log["attempts"]), 2):
        attempt = {"round": round_index}
        try:
            prompt = legacy.build_generation_prompt(source, operation, 3)
            attempt.setdefault("model_call_events", []).append({"role":"generation", "max_tokens":420})
            raw = legacy._completion(client, ALLOWED_MODEL, legacy.GENERATOR_SYSTEM_PROMPT,
                                      prompt, 0.65, 420)
            attempt["raw_generation"] = raw
            attempt["prompt_sha256"] = digest([legacy.GENERATOR_SYSTEM_PROMPT, prompt])
            candidates = legacy.parse_candidates(raw, source, operation, 3)
            attempt["candidates"] = candidates
            if candidates:
                judge_prompt = legacy.build_judge_prompt(source, operation, candidates)
                attempt.setdefault("model_call_events", []).append({"role":"pair_and_equivalence_judge", "max_tokens":520})
                raw_judgment = legacy._completion(client, ALLOWED_MODEL, legacy.JUDGE_SYSTEM_PROMPT,
                                                   judge_prompt, 0, 520)
                attempt["raw_judgment"] = raw_judgment
                judgments = parse_judgments(raw_judgment, len(candidates))
                attempt["judgments"] = judgments
                chosen = legacy.choose_candidate(candidates, judgments, source, operation)
                if chosen:
                    attempt["status"] = "accepted";log["attempts"].append(attempt)
                    log.update(status="accepted", selected_hypothesis=chosen[0])
                    return chosen[0], log
            attempt["status"] = "rejected"
        except Exception as exc:
            attempt.update(exception_fields(exc))
        log["attempts"].append(attempt)
    log["status"] = "rejected"
    return None, log


def generate_legacy_bounded(client, source, operation, seed=42, prior_trace=None):
    """Replay unchanged v3.3 prompts with the same single-plan budget as v3.4."""
    if operation not in COMPOSITE_STAGE_PLANS:
        return generate_legacy_atomic(client, source, operation, prior_trace)
    plans = [p for p in COMPOSITE_STAGE_PLANS[operation] if eligible(source, p[0])]
    if not plans:
        return None, {"status": "ineligible", "operation": operation}
    plan = tuple(prior_trace["applied_steps"]) if prior_trace else plans[int(digest([source_uid(source), operation, seed])[:8], 16) % len(plans)]
    if plan not in COMPOSITE_STAGE_PLANS[operation]:
        raise ValueError("saved composite plan does not belong to its MR")
    log = deepcopy(prior_trace) if prior_trace else {"operation": operation, "applied_steps": list(plan)}
    old_stages = log.get("stage_logs", [])
    log["stage_logs"] = []
    working = dict(source)
    for stage_index, stage in enumerate(plan):
        if not eligible(working, stage):
            log["status"] = "ineligible_stage"
            return None, log
        prior = old_stages[stage_index] if stage_index < len(old_stages) else None
        text, stage_log = generate_legacy_atomic(client, working, stage, prior)
        log["stage_logs"].append(stage_log)
        if text is None:
            log["status"] = "rejected"
            return None, log
        working["label"] = LABEL_IDS[target_label(working, stage)]
        working["hypothesis"] = text
    text = working["hypothesis"]
    try:
        raw = log.get("raw_final_judgment")
        if raw is None:
            log.setdefault("model_call_events", []).append({"role":"composite_final_judge", "max_tokens":350})
            raw = legacy._completion(client, ALLOWED_MODEL, legacy.JUDGE_SYSTEM_PROMPT,
                                      legacy.build_judge_prompt(source, operation, [text]), 0, 350)
        log["raw_final_judgment"] = raw
        judgments = parse_judgments(raw, 1)
        log["final_judgments"] = judgments
        if legacy.choose_candidate([text], judgments, source, operation):
            log.update(status="accepted", selected_hypothesis=text)
            return text, log
    except Exception as exc:
        log.update(exception_fields(exc))
    log["status"] = "rejected"
    return None, log
