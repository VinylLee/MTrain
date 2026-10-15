"""Post-review round-4 design; explicit opt-in, never replaces the round-3 engine.

Only the existing Gemma editor/judge is callable here. External NLI/probe scores
and corpus identities have no input to the proposal or acceptance functions.
"""
import json
import re

from . import core
from .data import digest, normalize, source_uid
from .invariants import WORD, FIXED_WORDS, FIRST_COMMON_WORDS, AGE_ALIASES, fixed_span_positions
from .protocol import (generator_json, parse_judgments, parse_equivalence_judgments,
                       exception_fields)
from .realization import realize_added_clause
from .schemas import FormattedClient, invariant_generation_schema

REVISION = "round4_shared_addition_independent_fact_checks_draft1"
ADDITIONS = frozenset(("adding_contradiction", "uninformative"))
DOMAINS = ("location", "position_or_posture", "activity", "possession", "material",
           "shape_or_size", "colour", "clothing", "count", "time_or_duration")
GENERATOR_SYSTEM = """Propose conservative English sentence edits. A source must be a
complete grammatical declarative statement; a fragment, question or command is not
an assertion to extend. Never silently repair a source or supply a missing subject.
Hypothetical additions need not be true of the premise: do not assert or assume
their truth. Only the requested MR may change the original proposition.
Return [] when safe realization is unavailable.
For additions generate from the supplied detail schedule without anticipating a
requested NLI label or MR. Use the same vocabulary and constructions for known and
unknown details. All proposals are untrusted until independently judged. JSON only."""
ADDITION_RULES = """Copy the original hypothesis unchanged, then append exactly one
complete clause about an explicitly anchored entity already mentioned in it.
Do not change its event, invent a person, or attach a clause to a fragment/command.
Use exactly one supplied domain/polarity slot; never switch to a clothing/colour
fallback just because the scheduled detail is difficult. Empty result is allowed.
Each slot may be used at most once in the returned array.
The anchor must be an exact entity span of the source hypothesis and unambiguous
there. For an anchor beginning A/An, start the added clause with The plus the same
remaining noun phrase, referring back to that entity; never introduce a second A/An
entity. Otherwise repeat the anchor (case may differ). Do not begin with and/but/
while or another connector. Use a short finite declarative clause, not a list
or multiple coordinated facts. Do not repeat the source hypothesis in the clause.
If the premise directly resolves this detail, quote its evidence_fact verbatim;
otherwise evidence_fact is empty. A quote is not proof of what it implies.
Concrete alternative values or negations of premise details are permitted when
they remain compatible with H. Do not restrict proposals to facts already true of P.
Negative polarity uses an explicit not/no/never; affirmative has no overt negator.
Affirmative/negative are grammatical polarities, never NLI labels. A positive
detail can be incompatible and a negative detail can be unresolved.
There is no target label. Do not prioritize which proposal might later be used.
The original hypothesis must leave the added detail unresolved: repeating it or
denying it is not an independent addition. No safe scheduled detail => []."""
JUDGE_SYSTEM = """Classify supplied English NLI comparisons independently from their
actual text, then check the edit. A generator claim and a requested operation are
not evidence. Guaranteed true => entailment; guaranteed false => contradiction;
otherwise neutral. Missing evidence is not contradiction. Do not infer a person's
identity, a missing fact or a finite verb to rescue an edit. A copied fragment or
bad subject/verb agreement still fails grammar. Evaluate full sentences, not only
the newly added words. Output only JSON with explicit labels and boolean checks."""
COMMON_CHECKS = ("source_is_complete_declarative", "candidate_is_complete_declarative",
                 "internally_consistent")
ADDITION_CHECKS = ("source_preserved", "one_fact_added", "anchor_unambiguous", "same_entity", "scheduled_domain_followed")


def schedule(hypothesis, round_index=0, seed=42):
    """Shared domain and polarity preferences, independent of labels and MR names."""
    ranked = sorted(DOMAINS, key=lambda d: digest([hypothesis, seed, round_index, d, REVISION]))
    return [{"slot":i,"domain":domain,"polarity":("affirmative" if
        int(digest([hypothesis,seed,round_index,domain,"polarity"])[0],16)%2 else "negative")}
        for i, domain in enumerate(ranked[:3])]


def editable_slots(hypothesis):
    words=list(WORD.finditer(hypothesis)); protected=fixed_span_positions([w.group() for w in words])
    ages=set().union(*AGE_ALIASES)
    result=[]
    for index, match in enumerate(words):
        word=match.group(); start,end=match.span()
        if index in protected or any(c.isdigit() for c in word):continue
        if not core.legacy.content_tokens(word) or (word.lower() in FIXED_WORDS and word.lower() not in ages):continue
        if word[0].isupper() and not (index==0 and word.lower() in FIRST_COMMON_WORDS):continue
        if start and hypothesis[start-1] in '\"“‘' and end<len(hypothesis) and hypothesis[end] in '\"”’':continue
        result.append({"word_index":index,"word":word})
    return result


def generation_schema(source, operation):
    if operation in ADDITIONS:
        item={"type":"object","properties":{
            "slot":{"type":"integer","enum":[0,1,2]},
            "source_is_complete_declarative":{"type":"boolean"},
            "entity_anchor":{"type":"string"},"evidence_fact":{"type":"string"},
            "edit_plan":{"type":"object","properties":{
                "connector":{"type":"string","enum":sorted(core.CONNECTORS)},
                "added_clause":{"type":"string"}},
                "required":["connector","added_clause"],"additionalProperties":False}},
            "required":["slot","source_is_complete_declarative","entity_anchor","evidence_fact","edit_plan"],
            "additionalProperties":False}
        return {"type":"json_schema","json_schema":{"name":"shared_addition_proposals","strict":True,
            "schema":{"type":"array","minItems":0,"maxItems":3,"items":item}}}
    result=invariant_generation_schema(source,operation)
    if operation=="synonym_replacement":
        allowed={s["word_index"] for s in editable_slots(source["hypothesis"])}
        schema=result["json_schema"]["schema"]
        if not allowed:
            result["json_schema"]["schema"]={"type":"array","maxItems":0}
        else:
            choices=schema["items"]["properties"]["edit_plan"]["properties"]["edits"]["items"]["oneOf"]
            choices[:]=[c for c in choices if c["properties"]["word_index"]["enum"][0] in allowed]
    return result


def build_generation_prompt(source, operation, round_index=0, seed=42):
    if operation in ADDITIONS:
        payload={"premise":source["premise"],"source_hypothesis":source["hypothesis"],
                 "detail_schedule":schedule(source["hypothesis"],round_index,seed)}
        example={"slot":0,"source_is_complete_declarative":True,"entity_anchor":"<exact source entity span>",
            "evidence_fact":"<exact premise quote, or empty string>",
            "edit_plan":{"connector":" and ","added_clause":"<one complete anchored claim>"}}
        return json.dumps(payload,ensure_ascii=False)+"\n"+ADDITION_RULES+"\nUp to 3 proposals: "+json.dumps([example])
    prompt=core.build_generation_prompt(source,operation,round_index,seed)
    if operation=="synonym_replacement":
        # Retain original indices, rather than renumbering the editable subset.
        first,rest=prompt.split("\n",1);payload=json.loads(first)
        payload["word_slots"]=editable_slots(source["hypothesis"])
        prompt=json.dumps(payload,ensure_ascii=False)+"\n"+rest
        prompt+="\nOnly these editable slots may change. Preserve grammatical inflection and every fixed span."
    if core.OPERATION_SPECS[operation]["family"]=="inv":
        prompt+="\nIf the original is a fragment, question or command, abstain; do not complete or repair it."
    return prompt


def surface_failure(hypothesis, clause):
    """Mechanical admissibility subset, not a complete grammar detector."""
    if hypothesis.strip().endswith(("?","!")):return "nondeclarative_punctuation"
    if re.match(r"(?i)^\s*(and|or|but|while|although|yet|even though|only if)\b",clause):return "leading_connector"
    if re.search(r"[;.!?]",clause.strip().rstrip(".")):return "unsupported_clause_punctuation"
    return None


def reference_phrase(anchor):
    """Anaphoric definite reference; the source hypothesis itself is untouched."""
    return re.sub(r"(?i)^(a|an)\s+", "the ", anchor)


def parse_proposals(raw, source, operation, round_index=0, seed=42):
    if operation not in ADDITIONS:
        if operation=="synonym_replacement":
            value=generator_json(raw)
            if isinstance(value,dict):value=[value]
            if not isinstance(value,list):raise ValueError("expected invariant candidate array")
            allowed={s["word_index"] for s in editable_slots(source["hypothesis"])}
            value=[v for v in value[:3] if isinstance(v,dict) and isinstance(v.get("edit_plan"),dict)
                   and isinstance(v["edit_plan"].get("edits"),list)
                   and all(isinstance(e,dict) and type(e.get("word_index")) is int and e["word_index"] in allowed
                           for e in v["edit_plan"]["edits"])]
            raw=json.dumps(value)
        return [{"hypothesis":text} for text in core.parse_candidates(raw,source,operation)]
    value=generator_json(raw)
    if not isinstance(value,list):raise ValueError("expected shared proposal array")
    result=[]; preferences=schedule(source["hypothesis"],round_index,seed);used_slots=set()
    for item in value[:3]:
        if not isinstance(item,dict) or item.get("source_is_complete_declarative") is not True:continue
        slot=item.get("slot");anchor=item.get("entity_anchor");evidence=item.get("evidence_fact");plan=item.get("edit_plan")
        if type(slot) is not int or not 0<=slot<3 or slot in used_slots or not isinstance(plan,dict):continue
        if not isinstance(anchor,str) or not anchor.strip() or anchor!=anchor.strip() or anchor not in source["hypothesis"]:continue
        # Require word boundaries for the quoted anchor, not a substring of another name.
        if not re.search(r"(?<!\w)"+re.escape(anchor)+r"(?!\w)",source["hypothesis"]):continue
        added=plan.get("added_clause");connector=plan.get("connector")
        if not isinstance(added,str) or connector not in core.CONNECTORS or surface_failure(source["hypothesis"],added):continue
        reference=reference_phrase(anchor)
        if not re.match(r"(?i)^\s*"+re.escape(reference)+r"(?!\w)",added):continue
        if not isinstance(evidence,str) or (evidence and evidence not in source["premise"]):continue
        if core.legacy.has_negation(added)!=(preferences[slot]["polarity"]=="negative"):continue
        if operation=="adding_contradiction":
            if not evidence.strip() or normalize(evidence) in normalize(source["hypothesis"]):continue
        text=realize_added_clause(source["hypothesis"],connector,added)
        if normalize(text) in {normalize(source["hypothesis"]),normalize(source["premise"])}:continue
        if not 3<=len(text.split())<=max(45,len(source["hypothesis"].split())*2+16):continue
        if re.search(r"(?i)(premise:|hypothesis:|target_label|source_label|```|<think>)",text):continue
        if text not in [p["hypothesis"] for p in result]:
            used_slots.add(slot)
            result.append({"hypothesis":text,"added_clause":added,"entity_anchor":anchor,
                "reference_phrase":reference,"evidence_fact":evidence,"slot":slot,"detail_preference":preferences[slot]})
    return result


def build_judge_prompt(source, operation, proposals, trajectory=None):
    candidates=[{"index":i,**proposal} for i,proposal in enumerate(proposals)]
    if core.OPERATION_SPECS[operation]["family"]=="inv":
        prompt=core.build_equivalence_prompt(source,operation,[p["hypothesis"] for p in proposals])
        prompt=prompt.replace("Keep a grammatical original fragment as a fragment rather than inventing facts.",
                              "Abstain on fragments, questions or commands; do not silently repair them.")
        prompt+="\nAlso return source_is_complete_declarative and candidate_is_complete_declarative as booleans. Both must be genuine complete grammatical assertions."
    else:
        payload={"premise":source["premise"],"source_hypothesis":source["hypothesis"],
                 "requested_operation":operation,"operation_requirement":core.OPERATION_SPECS[operation]["instruction"],
                 "candidates":candidates}
        prompt=json.dumps(payload,ensure_ascii=False)+"\nClassify P->source and P->candidate without gold/target labels."
        prompt+="\nReturn {evaluations:[...]} with index, source_label, label, grammatical, operation_valid, source_equivalent=false, reason (short tag)."
        prompt+="\nEach entry also has boolean source_is_complete_declarative, candidate_is_complete_declarative, internally_consistent. Reject fragments, commands, agreement errors and contradictory conjunctions."
        if operation in ADDITIONS:
            prompt+="""\nFor each candidate independently classify premise_to_added and source_to_added,
using only the supplied texts; each is entailment, neutral or contradiction.
Also return booleans source_preserved, one_fact_added, anchor_unambiguous, same_entity,
scheduled_domain_followed. Check the actual clause against its detail_preference;
a colour/clothing fallback under another scheduled domain must fail this check.
The quote is untrusted: verify it actually determines the added proposition, in
the same world, entity, tense and scope; a crash not mentioned in P is not denied.
For adding_contradiction require P->added=contradiction but H->added=neutral.
H->added=contradiction means an internally inconsistent conjunction and MUST fail
even when P->candidate is contradiction. H->added=entailment is redundant, not an
independent contradicted fact. The new fact must be different from what H says.
For uninformative require both P->added and H->added neutral, P->candidate neutral.
One complete anchored clause is required, with no invented referent. Reject any
claim without an unambiguous anchor and each clause with multiple added facts."""
    if trajectory is not None:
        # Only hypotheses and MR names: no factual P or labels enter an INV jury.
        prompt+="\nOrdered intermediate edits: "+json.dumps(trajectory,ensure_ascii=False)
        prompt+="\nAlso return ordered_components_valid=true only if every distinct requested step is valid in this order."
    return prompt


def parse_judgments_checked(raw, source, operation, proposals, composite=False):
    invariant=core.OPERATION_SPECS[operation]["family"]=="inv"
    base=(parse_equivalence_judgments if invariant else parse_judgments)(raw,len(proposals))
    value=core.legacy.extract_json(raw)
    entries=value if isinstance(value,list) else value.get("evaluations",[])
    extra={v["index"]:v for v in entries if isinstance(v,dict) and type(v.get("index")) is int}
    result={}
    for index,verdict in base.items():
        entry=extra[index]
        required=COMMON_CHECKS[:2] if invariant else COMMON_CHECKS
        if operation in ADDITIONS:required+=ADDITION_CHECKS
        if composite:required+=("ordered_components_valid",)
        if any(entry.get(k) is not True for k in required):continue
        if operation in ADDITIONS:
            expected="contradiction" if operation=="adding_contradiction" else "neutral"
            if entry.get("premise_to_added")!=expected or entry.get("source_to_added")!="neutral":continue
        result[index]=verdict
    return result


def contract():
    return {"revision":REVISION,"parent_round3_contract":core.fingerprint(),"status":"draft_not_frozen",
        "generator_system":GENERATOR_SYSTEM,"addition_rules":ADDITION_RULES,"judge_system":JUDGE_SYSTEM,
        "domain_schedule":list(DOMAINS),"generation_tokens":1000,"judge_tokens":1400,
        "candidate_count":3,"atomic_rounds":2,"atomic_call_ceiling":4,"composite_call_ceiling":9,
        "addition_generation":"identical_prompt_for_C_and_N_no_labels_MR_or_role",
        "acceptance":"P_to_added_target_H_to_added_neutral_explicit_grammar_anchor_and_single_fact",
        "required_common_checks":list(COMMON_CHECKS),"required_addition_checks":list(ADDITION_CHECKS),
        "schema":"pruned_original_index_synonym_slots_and_shared_typed_additions_v1",
        "invariant_rule":"H_only_generation_and_judgment_inherit_unchanged_gold",
        "source_scope":"shared_exact_H_entity_anchor_conservative_subset_no_source_rows_removed",
        "external_nli_in_pipeline":False,"source_labels_modified":False,
        "complete_sources_retained":True,"minimum_inv_fraction":core.MINIMUM_INV_FRACTION}


class Round4Engine:
    """Budgeted experimental engine; caller owns durable ledgers and run identities."""
    contract=staticmethod(contract)
    fingerprint=staticmethod(lambda:digest(contract()))
    target_label=staticmethod(core.target_label)
    OPERATION_SPECS=core.OPERATION_SPECS
    COMPOSITE_STAGE_PLANS=core.COMPOSITE_STAGE_PLANS
    LABEL_IDS=core.LABEL_IDS

    def eligible(self,source,operation):
        if operation=="synonym_replacement":return core.eligible(source,operation) and bool(editable_slots(source["hypothesis"]))
        if operation in core.COMPOSITE_STAGE_PLANS:return bool(self.available_plans(source,operation))
        return core.eligible(source,operation)

    def available_plans(self,source,operation):
        return [p for p in core.available_plans(source,operation) if self.eligible(source,p[0])]

    def _judge(self,client,source,operation,proposals,trace,trajectory=None):
        prompt=build_judge_prompt(source,operation,proposals,trajectory)
        system=core.EQUIVALENCE_SYSTEM if core.OPERATION_SPECS[operation]["family"]=="inv" else JUDGE_SYSTEM
        trace.setdefault("model_call_events",[]).append({"role":"round4_quality_judge","max_tokens":1400})
        raw=core._call(client,system,prompt,0.0,1400)
        trace.update(raw_judgment=raw,judge_prompt_sha256=digest([system,prompt]))
        judgments=parse_judgments_checked(raw,source,operation,proposals,trajectory is not None)
        trace["judgments"]=judgments
        return judgments

    def generate_atomic(self,client,source,operation,seed=42):
        log={"revision":REVISION,"contract_sha256":self.fingerprint(),"operation":operation,"source_uid":source_uid(source),"attempts":[]}
        for round_index in range(2):
            prompt=build_generation_prompt(source,operation,round_index,seed)
            invariant=core.OPERATION_SPECS[operation]["family"]=="inv"
            system=core.INVARIANT_SYSTEM if invariant else GENERATOR_SYSTEM
            attempt={"round":round_index,"prompt_sha256":digest([system,prompt]),"model_call_events":[{"role":"generation","max_tokens":1000}]}
            try:
                formatted=client
                if invariant or operation in ADDITIONS:
                    schema=generation_schema(source,operation);formatted=FormattedClient(client,schema)
                    attempt["generation_response_format_sha256"]=digest(schema)
                raw=core._call(formatted,system,prompt,0.7,1000);attempt["raw_generation"]=raw
                core._record_generation_normalization(attempt,raw)
                proposals=parse_proposals(raw,source,operation,round_index,seed)
                attempt.update(proposals=proposals,candidates=[p["hypothesis"] for p in proposals])
                if proposals:
                    judgments=self._judge(client,source,operation,proposals,attempt)
                    chosen=core.choose_current(attempt["candidates"],judgments,source,operation)
                    if chosen:
                        attempt["status"]="accepted";log["attempts"].append(attempt)
                        log.update(status="accepted",selected_hypothesis=chosen[0]);return chosen[0],log
                attempt["status"]="rejected"
            except Exception as exc:attempt.update(exception_fields(exc))
            log["attempts"].append(attempt)
        log["status"]="rejected";return None,log

    def generate_one(self,client,source,operation,seed=42):
        if not self.eligible(source,operation):return None,{"operation":operation,"status":"ineligible"}
        if operation not in core.COMPOSITE_STAGE_PLANS:return self.generate_atomic(client,source,operation,seed)
        plans=self.available_plans(source,operation)
        plan=plans[int(digest([source_uid(source),operation,seed])[:8],16)%len(plans)]
        working=dict(source);log={"operation":operation,"revision":REVISION,"contract_sha256":self.fingerprint(),
                                 "source_uid":source_uid(source),"applied_steps":list(plan),"stage_logs":[]}
        trajectory=[]
        for stage in plan:
            if not self.eligible(working,stage):log["status"]="ineligible_stage";return None,log
            text,stage_log=self.generate_atomic(client,working,stage,seed);log["stage_logs"].append(stage_log)
            if text is None:log["status"]="rejected";return None,log
            trajectory.append({"operation":stage,"source_hypothesis":working["hypothesis"],"hypothesis":text})
            working["label"]=core.LABEL_IDS[core.target_label(working,stage)];working["hypothesis"]=text
        try:
            text=working["hypothesis"];trace=log.setdefault("final_judge_trace",{})
            if normalize(text)==normalize(source["hypothesis"]):log["status"]="no_effect";return None,log
            verdict=self._judge(client,source,operation,[{"hypothesis":text}],trace,trajectory)
            chosen=core.choose_current([text],verdict,source,operation)
            if chosen:log.update(status="accepted",selected_hypothesis=text);return text,log
        except Exception as exc:log.update(exception_fields(exc))
        log["status"]="rejected";return None,log

    def replay_one(self,source,operation,trace,seed=42):
        """Verify accepted raw responses under this contract, with no client/API."""
        if (trace.get("contract_sha256")!=self.fingerprint() or trace.get("source_uid")!=source_uid(source)
                or trace.get("operation")!=operation):
            raise ValueError("saved round-4 contract/source/operation mismatch")
        if trace.get("status")!="accepted":return None
        if operation in core.COMPOSITE_STAGE_PLANS:
            steps=tuple(trace.get("applied_steps",[])); logs=trace.get("stage_logs",[])
            if steps not in self.available_plans(source,operation) or len(logs)!=len(steps):
                raise ValueError("invalid saved composite trajectory")
            working=dict(source);trajectory=[]
            for stage,stage_log in zip(steps,logs):
                text=self.replay_one(working,stage,stage_log,seed)
                if text is None:raise ValueError("composite contains an unaccepted stage")
                trajectory.append({"operation":stage,"source_hypothesis":working["hypothesis"],"hypothesis":text})
                working["label"]=core.LABEL_IDS[core.target_label(working,stage)];working["hypothesis"]=text
            proposals=[{"hypothesis":working["hypothesis"]}];final=trace["final_judge_trace"]
            invariant=core.OPERATION_SPECS[operation]["family"]=="inv"
            system=core.EQUIVALENCE_SYSTEM if invariant else JUDGE_SYSTEM
            if final["judge_prompt_sha256"]!=digest([system,build_judge_prompt(source,operation,proposals,trajectory)]):
                raise ValueError("saved final judge prompt changed")
            judgments=parse_judgments_checked(final["raw_judgment"],source,operation,proposals,True)
            chosen=core.choose_current([p["hypothesis"] for p in proposals],judgments,source,operation)
            if not chosen or chosen[0]!=trace.get("selected_hypothesis"):raise ValueError("final replay rejected")
            return chosen[0]
        attempts=trace.get("attempts",[])
        if len(attempts)>2:raise ValueError("saved request exceeds two rounds")
        for attempt in attempts:
            if attempt.get("status")!="accepted":continue
            i=attempt["round"];proposals=parse_proposals(attempt["raw_generation"],source,operation,i,seed)
            if proposals!=attempt["proposals"] or [p["hypothesis"] for p in proposals]!=attempt["candidates"]:
                raise ValueError("saved proposals differ from raw generation")
            invariant=core.OPERATION_SPECS[operation]["family"]=="inv"
            system=core.INVARIANT_SYSTEM if invariant else GENERATOR_SYSTEM
            if attempt["prompt_sha256"]!=digest([system,build_generation_prompt(source,operation,i,seed)]):
                raise ValueError("saved generation prompt changed")
            judge_system=core.EQUIVALENCE_SYSTEM if invariant else JUDGE_SYSTEM
            if attempt["judge_prompt_sha256"]!=digest([judge_system,build_judge_prompt(source,operation,proposals)]):
                raise ValueError("saved judge prompt changed")
            judgments=parse_judgments_checked(attempt["raw_judgment"],source,operation,proposals)
            chosen=core.choose_current(attempt["candidates"],judgments,source,operation)
            if chosen and chosen[0]==trace.get("selected_hypothesis"):return chosen[0]
        raise ValueError("accepted trace cannot be replayed")
