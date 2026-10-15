"""Opt-in post-review development engine: typed details and single-pair Gemma jury.

This does not replace any completed experiment or the default production engine.
No external NLI model, probe score, or corpus identity participates in acceptance.
"""
from copy import deepcopy
import json
import re
from types import SimpleNamespace

from . import core, redesign as parent
from .data import digest, normalize, source_uid
from .protocol import exception_fields, generator_json, ResponseFormatError
from .realization import realize_added_clause

REVISION = "round4b_typed_details_single_pair_jury_v1"
GEN_TOKENS, NLI_TOKENS, EDIT_TOKENS = 768, 384, 512
NLI_SYSTEM = """Classify one directional English NLI pair: premise -> hypothesis.
Assume the premise is true. Does that guarantee the hypothesis?
entailment: every claim in the hypothesis must be true.
contradiction: the premise and hypothesis cannot both be true.
neutral: the hypothesis may be true or false while the premise remains true.
Missing details are neutral. A more specific hypothesis is not guaranteed by a
general premise. Do not reverse the implication: proving hypothesis -> premise
does not prove premise -> hypothesis. Ordinary contextual synonyms are allowed.
Return label and a short reason, only for the current pair."""
GEN_SYSTEM = """Fill one typed hypothetical detail about the supplied entity.
The proposed detail may conflict with P or be unspecified by P, but must remain
compatible with H. There is no NLI label to predict. Return only the requested JSON.
Do not silently repair H. Abstain on a fragment, question or command."""
EDIT_SYSTEM = """Check English grammar and whether the requested edit was performed.
Judge the actual original and edited text. A claimed operation or typed field is
not evidence of compliance. Reject fragments, commands, agreement errors, unclear
references, extra facts, changed original claims and internally inconsistent text.
Do not predict NLI labels. Return the requested booleans and a short reason."""
GEN_RULES = """Fill only the supplied domain's detail fields. The code supplies the
subject, finite clause, polarity and connector; do not put these in a noun phrase.
Give a concrete property of the SAME referenced entity, independent of what H says.
A hypothetical alternative may disagree with P; an unstated detail is also allowed.
If P states evidence about this property, evidence_fact is an exact quote from P;
otherwise it is empty. A quote is not a proof of any relation.
Do not repeat or deny H. If no safe scheduled property is available, abstain=true.
Neither the source hypothesis nor its entity can be changed. No domain fallback."""
COMMON = ("grammatical", "operation_valid", "source_is_complete_declarative",
          "candidate_is_complete_declarative", "internally_consistent")
ADDITION = ("source_preserved", "one_fact_added", "anchor_unambiguous", "same_entity",
            "scheduled_domain_followed")


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def fmt(name, schema):
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def enum(values):
    return {"type": "string", "enum": values.split("|")}


TEXT = {"type": "string"}
DETAILS = {
    "location": {"preposition": enum("in|on|at|inside|outside|near|under|above|behind|beside"), "place": TEXT},
    "position_or_posture": {"posture": enum("standing|sitting|kneeling|lying down|crouching")},
    "activity": {"action_ing": TEXT},
    "possession": {"object_np": TEXT},
    "material": {"material_np": TEXT},
    "shape_or_size": {"attribute": enum("tall|short|large|small|wide|narrow|round|square")},
    "colour": {"colour": enum("red|blue|green|black|white|yellow|brown|orange|grey|purple|pink")},
    "clothing": {"garment_np": TEXT},
    "count": {"count": {"type": "integer", "minimum": 0, "maximum": 10}, "objects_np": TEXT},
    "time_or_duration": {"age_years": {"type": "integer", "minimum": 1, "maximum": 120}},
}
NLI_FORMAT = fmt("single_nli_comparison", obj({"label": enum("entailment|neutral|contradiction"), "reason": TEXT}))


def subject(hypothesis):
    """Conservative subject-prefix subset, not a general English syntax parser."""
    verbs = ("is|are|was|were|has|have|had|does|do|plays?|stands?|sits?|walks?|runs?|"
             "carries|carry|wears?|holds?|buys?|purchases?|reads?|watches?|eats?|drinks?|works?")
    match = re.match(r"^(.+?)\s+(" + verbs + r")\b", hypothesis, re.I)
    if not match or hypothesis.strip().endswith(("?", "!")):
        return None
    anchor, verb = match[1], match[2].lower()
    if len(anchor) > 70 or re.search(r"[,;!?]|\b(?:and|or|who|which|that)\b", anchor, re.I):
        return None
    if anchor.lower() in {"there", "here", "it", "this", "that"} or core.legacy.has_negation(anchor):
        return None
    reference = re.sub(r"(?i)^(a|an)\s+", "The ", anchor)
    quantified = re.match(r"(?i)^(two|three|four|five|six|seven|eight|nine|ten|\d+)\b", anchor)
    if quantified:
        reference = "The " + anchor[0].lower() + anchor[1:]
    plural = bool(quantified) or verb in {"are", "were", "have", "do", "play", "stand", "sit",
        "walk", "run", "carry", "wear", "hold", "buy", "purchase", "read", "watch", "eat", "drink", "work"}
    if anchor.lower() in {"i", "you"}:
        return None  # avoid person-dependent conjugation in this subset
    return {"anchor": anchor, "reference": reference, "plural": plural}


def preference(source, round_index, seed=42):
    # Preserve the prospective, MR/label-blind parent schedule; one slot per round.
    return parent.schedule(source["hypothesis"], round_index, seed)[0]


def generation_schema(source, operation, round_index=0, seed=42):
    if operation in parent.ADDITIONS:
        domain = preference(source, round_index, seed)["domain"]
        return fmt("one_typed_detail", obj({"abstain": {"type": "boolean"},
            "evidence_fact": TEXT, "detail": obj(DETAILS[domain])}))
    result = deepcopy(parent.generation_schema(source, operation))
    result["json_schema"]["schema"]["maxItems"] = 1
    return result


def generation_prompt(source, operation, round_index=0, seed=42):
    if operation not in parent.ADDITIONS:
        return parent.build_generation_prompt(source, operation, round_index, seed) + "\nReturn at most ONE candidate, or []."
    payload = {"premise": source["premise"], "source_hypothesis": source["hypothesis"],
        "entity": subject(source["hypothesis"]), "preference": preference(source, round_index, seed)}
    return GEN_RULES + "\n" + json.dumps(payload, ensure_ascii=False)


def _validate_detail(detail, properties):
    if not isinstance(detail, dict) or set(detail) != set(properties):
        raise ResponseFormatError("typed detail fields differ from scheduled domain")
    for key, schema in properties.items():
        v = detail[key]
        if schema["type"] == "integer":
            if type(v) is not int or not schema["minimum"] <= v <= schema["maximum"]:
                raise ResponseFormatError("invalid typed integer")
        elif not isinstance(v, str) or not v.strip() or v != v.strip() or len(v.split()) > 9:
            raise ResponseFormatError("invalid typed phrase")
        elif "enum" in schema and v not in schema["enum"]:
            raise ResponseFormatError("value outside typed vocabulary")
        elif re.search(r"[.!?;:\n]|\b(?:and|or|but|he|she|they|it)\b", v, re.I) or core.legacy.has_negation(v):
            raise ResponseFormatError("typed value contains a clause, connector or negation")


def realize_detail(entity, domain, detail, polarity):
    _validate_detail(detail, DETAILS[domain])
    plural = entity["plural"]
    be = "are" if plural else "is"
    negative = polarity == "negative"
    copula = be + (" not" if negative else "")
    have = ("do" if plural else "does") + " not have" if negative else ("have" if plural else "has")
    if domain == "location": predicate = f"{copula} {detail['preposition']} {detail['place']}"
    elif domain == "position_or_posture": predicate = f"{copula} {detail['posture']}"
    elif domain == "activity":
        if not re.match(r"(?i)^\w+ing\b", detail["action_ing"]):
            raise ResponseFormatError("activity must start with an ing verb")
        predicate = f"{copula} {detail['action_ing']}"
    elif domain == "possession": predicate = f"{have} {detail['object_np']}"
    elif domain == "material": predicate = f"{copula} made of {detail['material_np']}"
    elif domain == "shape_or_size": predicate = f"{copula} {detail['attribute']}"
    elif domain == "colour": predicate = f"{copula} {detail['colour']}"
    elif domain == "clothing": predicate = f"{copula} wearing {detail['garment_np']}"
    elif domain == "count": predicate = f"{have} exactly {detail['count']} {detail['objects_np']}"
    else:
        years = "year" if detail["age_years"] == 1 else "years"
        predicate = f"{copula} {detail['age_years']} {years} old"
    clause = entity["reference"] + " " + predicate
    assert core.legacy.has_negation(clause) == negative
    return clause


def proposals(raw, source, operation, round_index=0, seed=42):
    if operation not in parent.ADDITIONS:
        value = generator_json(raw)
        if not isinstance(value, list) or len(value) > 1:
            raise ResponseFormatError("expected at most one invariant candidate")
        return parent.parse_proposals(raw, source, operation, round_index, seed)
    value = generator_json(raw)
    if not isinstance(value, dict) or set(value) != {"abstain", "evidence_fact", "detail"} or type(value["abstain"]) is not bool:
        raise ResponseFormatError("invalid typed proposal envelope")
    if value["abstain"]: return []
    entity = subject(source["hypothesis"])
    if not entity: return []
    evidence = value["evidence_fact"]
    if not isinstance(evidence, str) or (evidence and evidence not in source["premise"]):
        raise ResponseFormatError("nonverbatim evidence")
    pref = preference(source, round_index, seed)
    clause = realize_detail(entity, pref["domain"], value["detail"], pref["polarity"])
    if operation == "adding_contradiction" and (not evidence.strip() or normalize(evidence) in normalize(source["hypothesis"])):
        return []
    text = realize_added_clause(source["hypothesis"], " and ", clause)
    if len(text.split()) > max(45, len(source["hypothesis"].split()) * 2 + 16): return []
    return [{"hypothesis": text, "added_clause": clause, "entity_anchor": entity["anchor"],
             "reference_phrase": entity["reference"], "evidence_fact": evidence,
             "detail_preference": pref, "detail": value["detail"]}]


def edit_request(source, operation, proposal, trajectory=None):
    checks = COMMON + (ADDITION if operation in parent.ADDITIONS else ())
    if trajectory is not None: checks += ("ordered_components_valid",)
    schema = fmt("edit_checks_without_nli", obj({**{k: {"type": "boolean"} for k in checks}, "reason": TEXT}))
    payload = {"source_hypothesis": source["hypothesis"], "candidate": proposal,
        "operation": operation, "requirement": core.OPERATION_SPECS[operation]["instruction"]}
    if trajectory is not None: payload["ordered_trajectory"] = trajectory
    prompt = json.dumps(payload, ensure_ascii=False) + "\nCheck complete grammatical statements and the exact edit. No NLI labels. All checks refer to actual text, not the generator's claims."
    return prompt, schema, checks


def call_cap(operation):
    return 19 if operation in core.COMPOSITE_STAGE_PLANS else 12 if operation in parent.ADDITIONS else 8


def token_cap(operation):
    return 9472 if operation in core.COMPOSITE_STAGE_PLANS else 5632 if operation in parent.ADDITIONS else 4096


class SavedResponses:
    """Read-only replay; missing or modified requests cannot dispatch anything."""
    def __init__(self, records):
        self.records, self.position = records, 0
        self.chat = self.completions = self

    def create(self, **kwargs):
        if self.position >= len(self.records): raise RuntimeError("saved response missing")
        record = self.records[self.position]; self.position += 1
        if digest(kwargs) != record["request_sha256"] or not isinstance(record.get("raw"), str):
            raise RuntimeError("saved response/request identity changed")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=record["raw"]))])


class RefinementEngine(parent.Round4Engine):
    revision = REVISION
    generation_system = GEN_SYSTEM
    generation_prompt = staticmethod(generation_prompt)
    generation_schema = staticmethod(generation_schema)
    parse_proposals = staticmethod(proposals)
    edit_request = staticmethod(edit_request)

    def normalize_edit(self, value, checks):
        return value

    @staticmethod
    def contract():
        return {"revision": REVISION, "status": "post_review_development_not_frozen",
            "parent_contract": parent.Round4Engine.fingerprint(), "nli_system": NLI_SYSTEM,
            "generation_system": GEN_SYSTEM, "generation_rules": GEN_RULES, "edit_system": EDIT_SYSTEM,
            "detail_properties": DETAILS, "nli_format": NLI_FORMAT, "common_checks": list(COMMON),
            "addition_checks": list(ADDITION), "generation_tokens": GEN_TOKENS, "nli_tokens": NLI_TOKENS,
            "edit_tokens": EDIT_TOKENS, "candidate_count": 1, "atomic_rounds": 2,
            "atomic_inv_call_cap": 8, "addition_call_cap": 12, "composite_inv_call_cap": 19,
            "addition_style": "shared_typed_values_reference_and_polarity_realized_by_code",
            "entity_subset": "uncoordinated_subject_prefix_before_finite_verb_v1",
            "time_domain": "entity_age_in_years", "external_nli_in_pipeline": False,
            "gold_modified": False, "all_sources_retained": True, "production_allowed": False}

    @classmethod
    def fingerprint(cls): return digest(cls.contract())

    def eligible(self, source, operation):
        if operation not in parent.ADDITIONS and core.OPERATION_SPECS[operation]["family"] != "inv":
            return False
        if operation in parent.ADDITIONS:
            return core.eligible(source, operation) and subject(source["hypothesis"]) is not None
        return super().eligible(source, operation)

    def _call(self, client, system, prompt, schema, tokens, temperature, trace, role):
        kwargs = {"model": core.ALLOWED_MODEL, "messages": [{"role": "system", "content": system},
            {"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": tokens,
            "response_format": schema}
        record = {"role": role, "max_tokens": tokens, "request_sha256": digest(kwargs)}
        trace.setdefault("calls", []).append(record)
        trace.setdefault("model_call_events", []).append({"role": role, "max_tokens": tokens})
        response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""
        if not isinstance(raw, str): raise ResponseFormatError("nontext completion")
        record["raw"] = raw
        return raw

    def _nli(self, client, left, right, trace, role):
        prompt = json.dumps({"premise": left, "hypothesis": right}, ensure_ascii=False)
        raw = self._call(client, NLI_SYSTEM, prompt, NLI_FORMAT, NLI_TOKENS, 0.0, trace, role)
        value = core.legacy.extract_json(raw)
        if (not isinstance(value, dict) or set(value) != {"label", "reason"}
                or not isinstance(value["label"], str) or value["label"] not in core.LABEL_IDS
                or not isinstance(value["reason"], str)):
            raise ResponseFormatError("invalid single directional NLI verdict")
        return value["label"]

    def _judge(self, client, source, operation, proposal_list, trace, trajectory=None):
        if len(proposal_list) != 1: raise ValueError("exactly one prospective candidate required")
        p = proposal_list[0]; text = p["hypothesis"]
        invariant = core.OPERATION_SPECS[operation]["family"] == "inv"
        if invariant:
            directions = {"source_to_candidate": self._nli(client, source["hypothesis"], text, trace, "H_to_candidate"),
                          "candidate_to_source": self._nli(client, text, source["hypothesis"], trace, "candidate_to_H")}
        else:
            directions = {"source_label": self._nli(client, source["premise"], source["hypothesis"], trace, "P_to_H"),
                "label": self._nli(client, source["premise"], text, trace, "P_to_candidate"),
                "premise_to_added": self._nli(client, source["premise"], p["added_clause"], trace, "P_to_added"),
                "source_to_added": self._nli(client, source["hypothesis"], p["added_clause"], trace, "H_to_added")}
        trace["independent_relations"] = directions
        prompt, schema, checks = self.edit_request(source, operation, p, trajectory)
        raw = self._call(client, EDIT_SYSTEM, prompt, schema, EDIT_TOKENS, 0.0, trace, "edit_checks")
        value = core.legacy.extract_json(raw); trace["edit_checks"] = value
        value = self.normalize_edit(value, checks)
        if (not isinstance(value, dict) or set(value) != set(checks) | {"reason"}
                or not isinstance(value.get("reason"), str)
                or any(type(value.get(k)) is not bool for k in checks)):
            raise ResponseFormatError("edit checks missing/invalid boolean")
        if not all(value[k] for k in checks): return {}
        if not invariant:
            expected = "contradiction" if operation == "adding_contradiction" else "neutral"
            if directions["source_to_added"] != "neutral" or directions["premise_to_added"] != expected:
                return {}
        verdict = {**directions, "grammatical": True, "operation_valid": True,
            "source_equivalent": invariant and all(v == "entailment" for v in directions.values()),
            "reason": str(value.get("reason", "checked"))}
        trace["judgments"] = {0: verdict}
        return {0: verdict}

    def _attempt(self, client, source, operation, round_index, seed):
        trace = {"round": round_index}
        try:
            system = self.generation_system if operation in parent.ADDITIONS else core.INVARIANT_SYSTEM
            raw = self._call(client, system, self.generation_prompt(source, operation, round_index, seed),
                self.generation_schema(source, operation, round_index, seed), GEN_TOKENS, 0.7, trace, "generation")
            trace["raw_generation"] = raw
            core._record_generation_normalization(trace, raw)
            ps = self.parse_proposals(raw, source, operation, round_index, seed)
            trace.update(proposals=ps, candidates=[p["hypothesis"] for p in ps])
            if ps:
                judgments = self._judge(client, source, operation, ps, trace)
                chosen = core.choose_current(trace["candidates"], judgments, source, operation)
                if chosen:
                    trace["status"] = "accepted"
                    return chosen[0], trace
            trace["status"] = "rejected"
        except Exception as exc:
            trace.update(exception_fields(exc))
        return None, trace

    def generate_atomic(self, client, source, operation, seed=42):
        log = {"revision": self.revision, "contract_sha256": self.fingerprint(), "operation": operation,
               "source_uid": source_uid(source), "attempts": []}
        for i in range(2):
            text, trace = self._attempt(client, source, operation, i, seed)
            log["attempts"].append(trace)
            if text:
                log.update(status="accepted", selected_hypothesis=text)
                return text, log
        log["status"] = "rejected"
        return None, log

    def generate_one(self, client, source, operation, seed=42):
        if not self.eligible(source, operation): return None, {"operation": operation, "status": "ineligible"}
        if operation not in core.COMPOSITE_STAGE_PLANS:
            return self.generate_atomic(client, source, operation, seed)
        plans = self.available_plans(source, operation)
        steps = plans[int(digest([source_uid(source), operation, seed])[:8], 16) % len(plans)]
        working = dict(source); trajectory = []
        log = {"revision": self.revision, "contract_sha256": self.fingerprint(), "operation": operation,
            "source_uid": source_uid(source), "applied_steps": list(steps), "stage_logs": []}
        for stage in steps:
            if not self.eligible(working, stage): log["status"] = "ineligible_stage"; return None, log
            text, stage_log = self.generate_atomic(client, working, stage, seed)
            log["stage_logs"].append(stage_log)
            if not text: log["status"] = "rejected"; return None, log
            trajectory.append({"operation": stage, "source_hypothesis": working["hypothesis"], "hypothesis": text})
            working.update(hypothesis=text, label=core.LABEL_IDS[core.target_label(working, stage)])
        trace = log.setdefault("final_judge_trace", {})
        try:
            if normalize(working["hypothesis"]) == normalize(source["hypothesis"]):
                log["status"] = "no_effect"; return None, log
            ps = [{"hypothesis": working["hypothesis"]}]
            judgments = self._judge(client, source, operation, ps, trace, trajectory)
            if core.choose_current([working["hypothesis"]], judgments, source, operation):
                log.update(status="accepted", selected_hypothesis=working["hypothesis"])
                return working["hypothesis"], log
        except Exception as exc: log.update(exception_fields(exc))
        log["status"] = "rejected"
        return None, log

    def replay_one(self, source, operation, trace, seed=42):
        if (trace.get("contract_sha256") != self.fingerprint() or trace.get("source_uid") != source_uid(source)
                or trace.get("operation") != operation or trace.get("revision") != self.revision):
            raise ValueError("refinement trace identity changed")
        if trace.get("status") != "accepted": return None
        if operation in core.COMPOSITE_STAGE_PLANS:
            steps = tuple(trace.get("applied_steps", [])); logs = trace.get("stage_logs", [])
            if steps not in self.available_plans(source, operation) or len(logs) != len(steps):
                raise ValueError("ordered refinement trajectory changed")
            working = dict(source); trajectory = []
            for stage, saved in zip(steps, logs):
                text = self.replay_one(working, stage, saved, seed)
                if text is None: raise ValueError("unaccepted composite stage")
                trajectory.append({"operation": stage, "source_hypothesis": working["hypothesis"], "hypothesis": text})
                working.update(hypothesis=text, label=core.LABEL_IDS[core.target_label(working, stage)])
            saved = trace["final_judge_trace"]; client = SavedResponses(saved["calls"]); replay = {}
            verdict = self._judge(client, source, operation, [{"hypothesis": working["hypothesis"]}], replay, trajectory)
            chosen = core.choose_current([working["hypothesis"]], verdict, source, operation)
            if not chosen or client.position != len(client.records) or json.loads(json.dumps(replay)) != json.loads(json.dumps(saved)):
                raise ValueError("final refinement replay failed")
            text = chosen[0]
        else:
            attempts = trace.get("attempts", [])
            if not 1 <= len(attempts) <= 2 or [a["round"] for a in attempts] != list(range(len(attempts))):
                raise ValueError("refinement round budget changed")
            accepted = [a for a in attempts if a.get("status") == "accepted"]
            if len(accepted) != 1 or accepted[0] is not attempts[-1]: raise ValueError("ambiguous accepted attempt")
            saved = accepted[0]; client = SavedResponses(saved["calls"])
            text, replay = self._attempt(client, source, operation, saved["round"], seed)
            if not text or client.position != len(client.records) or json.loads(json.dumps(replay)) != json.loads(json.dumps(saved)):
                raise ValueError("refinement raw replay failed")
        if text != trace.get("selected_hypothesis"): raise ValueError("selected refinement text changed")
        return text
