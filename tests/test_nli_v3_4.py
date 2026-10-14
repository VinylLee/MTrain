"""Research invariants: isolation, source preservation, quotas and fail-closed gates."""
import json
from pathlib import Path

import pytest

pytest.importorskip("openai")
from nli_v3_4 import core, workflow
from nli_v3_4.data import canonical_source, digest, load_dataset, quotas, write_json, write_jsonl
from analysis.design_nli_quality import development_rows
from nli_v3_4.protocol import parse_judgments, ResponseFormatError, has_api_failure, exception_fields

SOURCE = {"premise": "A man in a red shirt is carrying a box outdoors.",
          "hypothesis": "A man carries a box.", "label": 0, "pair_id": 7}


VERDICT = {"index":0,"label":"entailment","source_label":"entailment",
           "grammatical":True,"operation_valid":True,"source_equivalent":True,"reason":"ok",
           "source_to_candidate":"entailment","candidate_to_source":"entailment"}


def synonym_object(source,text):
    before=core.WORD.findall(source["hypothesis"]);after=core.WORD.findall(text)
    edits=[{"word_index":i,"old_word":a,"replacement":b} for i,(a,b) in enumerate(zip(before,after)) if a!=b]
    return {"label_rule":"inherit_source","edit_plan":{"edits":edits}}


def test_inv_cannot_pass_without_both_independent_entailment_directions():
    for changed in ({"candidate_to_source":"neutral"}, {"source_to_candidate":"contradiction"}):
        verdict=parse_judgments(json.dumps([{**VERDICT,**changed}]),1,require_bidirectional=True)[0]
        assert not verdict["source_equivalent"]
    value={k:v for k,v in VERDICT.items() if k!="candidate_to_source"}
    assert parse_judgments(json.dumps([value]),1,require_bidirectional=True)=={}


def test_synonym_permits_two_words_but_preserves_spelled_numbers_and_qualifiers():
    source={**SOURCE,"hypothesis":"The men carry a box."}
    def parsed(text):
        if len(core.WORD.findall(text))!=len(core.WORD.findall(source["hypothesis"])):return []
        return core.parse_candidates(json.dumps([synonym_object(source,text)]),source,"synonym_replacement")
    assert parsed("The men transport a carton.")
    assert parsed("The men transport a carton")
    source["hypothesis"]="The men carry a box"
    assert parsed("The men transport a carton.")
    source["hypothesis"]="It was one hundred ten degrees in West Texas."
    assert not parsed("It was one-ten degrees in West Texas.")
    source["hypothesis"]="Professor Rogers has always been a clerk to him."
    assert not parsed("Professor Rogers has served him as a clerk.")
    assert not parsed("Professor Smith has always been a clerk to him.")


def test_pronoun_requires_removed_np_and_handles_repeated_np_occurrences():
    source={**SOURCE,"hypothesis":"The instructors help, and are always available."}
    text="The instructors help, and they are always available."
    assert not core.parse_candidates(json.dumps([{"hypothesis":text,"label_rule":"inherit_source"}]),source,"pronoun_substitution")
    source["hypothesis"]="The men carry a box and the men walk."
    text="The men carry a box and they walk."
    assert core.parse_candidates(json.dumps([{"hypothesis":text,"label_rule":"inherit_source"}]),source,"pronoun_substitution")
    source["hypothesis"]="A man carries a box."
    text="He carries a box."
    assert not core.parse_candidates(json.dumps([{"hypothesis":text,"label_rule":"inherit_source"}]),source,"pronoun_substitution")
    source["hypothesis"]="A man watches another man at the table."
    text="He watches another man at the table."
    assert not core.parse_candidates(json.dumps([{"hypothesis":text,"label_rule":"inherit_source"}]),source,"pronoun_substitution")
    assert not core.eligible(source,"pronoun_substitution")
    source["hypothesis"]="The woman rests and the woman sleeps."
    for text in ("The woman rests and he sleeps.", "She rests and the woman sleeps."):
        assert not core.parse_candidates(json.dumps([{"hypothesis":text,"label_rule":"inherit_source"}]),source,"pronoun_substitution")
    assert core.parse_candidates(json.dumps([{"hypothesis":"The woman rests and she sleeps.","label_rule":"inherit_source"}]),source,"pronoun_substitution")


def test_judge_array_and_object_have_identical_strict_verdicts():
    array = parse_judgments(json.dumps([VERDICT]),1)
    wrapped = parse_judgments(json.dumps({"evaluations":[VERDICT]}),1)
    assert array == wrapped


@pytest.mark.parametrize("changed", [{"label":"the premise seems plausible"},
    {"label":0},{"grammatical":"true"},{"source_equivalent":None}])
def test_judge_does_not_coerce_nonlabels_or_missing_booleans(changed):
    assert parse_judgments(json.dumps([{**VERDICT,**changed}]),1) == {}


def test_duplicate_or_shifted_indices_are_ambiguous_and_rejected():
    for value in ([VERDICT,VERDICT],[{**VERDICT,"index":1}],[{**VERDICT,"index":True}]):
        with pytest.raises(ResponseFormatError):parse_judgments(json.dumps(value),1)


def test_format_error_is_not_a_service_failure():
    fields=exception_fields(ResponseFormatError("missing evaluations"))
    assert fields["error_kind"] == "response_format"
    assert not has_api_failure({"attempts":[fields]})


def test_cached_judgment_replay_does_not_spend_new_model_budget():
    from nli_v3_4.recovery import OfflineClient
    candidate="A man transports a box."
    raw=json.dumps([synonym_object(SOURCE,candidate)])
    prior={"operation":"synonym_replacement","status":"rejected","attempts":[
        {"raw_generation":raw,"candidates":[candidate],"raw_judgment":json.dumps([VERDICT]),
         "raw_equivalence_judgment":json.dumps([VERDICT]),
         "status":"error","error":"ValueError: judge response lacks evaluations"}]}
    text,trace=core.generate_atomic(OfflineClient(),SOURCE,"synonym_replacement",prior_trace=prior)
    assert text == candidate and len(trace["attempts"]) == 1
    assert trace["protocol_replay"]["original_status"] == "rejected"


def test_api_errors_are_distinct_from_response_errors():
    import httpx
    from openai import APIConnectionError
    fields=exception_fields(APIConnectionError(message="test transport failure",
        request=httpx.Request("POST","http://localhost/v1/chat/completions")))
    assert has_api_failure({"stage_logs":[fields]})


def test_pending_journal_update_keeps_later_appends_and_request_identity(tmp_path,monkeypatch):
    import openai
    from nli_v3_4.data import source_uid
    monkeypatch.setattr(openai,"OpenAI",lambda **kwargs:object())
    monkeypatch.setattr(core.legacy,"_completion",lambda *args,**kwargs:"OK")
    manifest=tiny_manifest(tmp_path);output=tmp_path/"resumed.jsonl"
    workflow.run_generation(manifest,"train",output,target=3,limit_requests=0)
    source=load_dataset(tmp_path/"original.jsonl")[1][0]
    operation="composite_inv";key=digest([source_uid(source),operation,0,42,"v3.4"])
    entry={"request_id":key,"pair_id":0,"source_uid":source_uid(source),"operation":operation,
        "refill_round":0,"target_label":0,"pending_continuation":True,"row":None,
        "trace":{"status":"rejected","applied_steps":list(core.COMPOSITE_STAGE_PLANS[operation][0])}}
    write_jsonl(output.with_suffix(".generation_log.jsonl"),[entry])
    prior_calls=[]
    def generate(client,source,op,seed,prior_trace=None):
        prior_calls.append(prior_trace)
        text=source["hypothesis"].rstrip(".")+" transformed "+op
        trace={"status":"accepted","selected_hypothesis":text}
        if op in core.COMPOSITE_STAGE_PLANS:trace["applied_steps"]=list(core.COMPOSITE_STAGE_PLANS[op][0])
        return text,trace
    monkeypatch.setattr(core,"generate_one",generate)
    result=workflow.run_generation(manifest,"train",output,target=3,resume=True)
    journal=[json.loads(s) for s in output.with_suffix(".generation_log.jsonl").read_text().splitlines()]
    assert prior_calls[0]==entry["trace"]
    assert len(journal)==3 and len({r["request_id"] for r in journal})==3
    assert all(r.get("row") for r in journal) and journal[0]["request_id"]==key
    assert result["generation_complete"]


def test_repeated_format_errors_do_not_abort_generation(tmp_path,monkeypatch):
    import openai
    monkeypatch.setattr(openai,"OpenAI",lambda **kwargs:object())
    monkeypatch.setattr(core.legacy,"_completion",lambda *args,**kwargs:"OK")
    calls=[]
    def generate(client,source,op,seed):
        calls.append(op)
        if len(calls)<=4:return None,{"status":"rejected","attempts":[exception_fields(ResponseFormatError("test"))]}
        trace={"status":"accepted"}
        if op in core.COMPOSITE_STAGE_PLANS:trace["applied_steps"]=list(core.COMPOSITE_STAGE_PLANS[op][0])
        return source["hypothesis"]+" transformed "+op,trace
    monkeypatch.setattr(core,"generate_one",generate)
    result=workflow.run_generation(tiny_manifest(tmp_path),"train",tmp_path/"generated.jsonl",target=3)
    assert result["generation_complete"] and len(calls)>=7


def test_composite_resume_only_calls_unused_final_judge(monkeypatch):
    from nli_v3_4.data import source_uid
    source={"premise":"The men carry a box and the men walk.","hypothesis":"The men carry a box and the men walk.","label":0,"pair_id":7}
    intermediate="The men carry a box and they walk."
    final="The men transport a box and they walk."
    stages=[];working=dict(source)
    for op,text in (("pronoun_substitution",intermediate),("synonym_replacement",final)):
        stages.append({"operation":op,"source_uid":source_uid(working),"status":"rejected","attempts":[
            {"raw_generation":json.dumps([synonym_object(working,text) if op=="synonym_replacement" else {"hypothesis":text,"label_rule":"inherit_source"}]),
             "candidates":[text],"raw_judgment":json.dumps([VERDICT])}]})
        stages[-1]["attempts"][0]["raw_equivalence_judgment"]=json.dumps([VERDICT])
        working["hypothesis"]=text
    prior={"operation":"composite_inv","status":"rejected",
           "applied_steps":["pronoun_substitution","synonym_replacement"],"stage_logs":stages}
    calls=[]
    def call(client,system,prompt,temperature,tokens):
        calls.append(prompt);return json.dumps([VERDICT])
    monkeypatch.setattr(core,"_call",call)
    text,trace=core.generate_one(object(),source,"composite_inv",prior_trace=prior)
    assert text==final and len(calls)==1
    assert trace["applied_steps"]==prior["applied_steps"]


def test_equivalence_prompt_has_no_original_premise_or_gold_label():
    source={**SOURCE,"premise":"UNIQUE FACTUAL PREMISE SENTINEL", "label":2}
    prompt=core.build_equivalence_prompt(source,"synonym_replacement",["A man transports a box."])
    assert source["premise"] not in prompt
    assert '"source_label"' not in prompt and '"target_label"' not in prompt


def test_inv_preserves_gold_only_with_valid_direction_labels_and_stays_in_family():
    from nli_v3_4.protocol import choose_invariant_candidate
    source={**SOURCE,"label":1}
    chosen=choose_invariant_candidate(["A man transports a box."],{0:VERDICT},source,"synonym_replacement")
    assert chosen[1]["output_label"]=="neutral" and source["label"]==1
    assert choose_invariant_candidate(["A man transports a box."],
        {0:{**VERDICT,"candidate_to_source":"neutral"}},source,"synonym_replacement") is None
    with pytest.raises(ValueError):
        choose_invariant_candidate(["A man transports a box."],{0:VERDICT},source,"negation_flip")


def plan(added="the man is not a doctor", connector=" and ", label="neutral"):
    return {"target_label": label, "edit_plan": {"added_clause": added, "connector": connector}}


def test_roles_and_corpus_metadata_cannot_change_prompt_or_style():
    train = {**SOURCE, "is_syn": False, "corpus": "training", "mr_id": "none"}
    test = {**SOURCE, "is_source": True, "corpus": "test", "idx": 100}
    for op in core.OPERATION_SPECS:
        assert core.build_generation_prompt(train, op) == core.build_generation_prompt(test, op)
        assert core.style_for(train, op, 1) == core.style_for(test, op, 1)


def test_invariant_editor_is_blind_to_premise_gold_id_and_role():
    changed={**SOURCE,"premise":"SECRET FACTUAL WORLD","label":2,"pair_id":999,"corpus":"test"}
    for op in ("synonym_replacement","voice_switch","pronoun_substitution"):
        first=core.build_generation_prompt(SOURCE,op)
        assert first==core.build_generation_prompt(changed,op)
        assert SOURCE["premise"] not in first and '"source_label"' not in first and '"target_label"' not in first


def test_synonym_plan_copies_text_and_rejects_changed_quantity_name_and_added_qualifier():
    from nli_v3_4.invariants import realize_synonym_plan
    text="The children buy two books."
    plan={"edits":[{"word_index":1,"old_word":"children","replacement":"kids"},
                   {"word_index":2,"old_word":"buy","replacement":"purchase"}]}
    assert realize_synonym_plan(text,plan)=="The kids purchase two books."
    assert realize_synonym_plan(text,{"edits":[{"word_index":3,"old_word":"two","replacement":"three"}]}) is None
    assert realize_synonym_plan("John buys books.",{"edits":[{"word_index":0,"old_word":"John","replacement":"Joe"}]}) is None
    assert realize_synonym_plan("He walks.",{"edits":[{"word_index":1,"old_word":"walks","replacement":"walks happily"}]}) is None
    assert realize_synonym_plan(text,{"edits":[{"word_index":True,"old_word":"children","replacement":"kids"}]}) is None
    assert realize_synonym_plan("They await help.",{"edits":[{"word_index":1,"old_word":"await","replacement":"wait for"}]})=="They wait for help."


def test_voice_accepts_nonvisual_verb_and_preexisting_locative_by_but_keeps_aspect():
    source={**SOURCE,"hypothesis":"The manager rejected the proposal by the river."}
    assert core.eligible(source,"voice_switch")
    def parsed(text):return core.parse_candidates(json.dumps([{"label_rule":"inherit_source","hypothesis":text}]),source,"voice_switch")
    assert parsed("The proposal was rejected by the manager by the river.")
    source["hypothesis"]="The men are carrying a box."
    assert not parsed("A box has been carried by the men.")
    assert parsed("A box is being carried by the men.")


def test_noninvariant_outputs_cannot_fill_missing_invariant_quota(tmp_path,monkeypatch):
    import openai
    monkeypatch.setattr(openai,"OpenAI",lambda **kwargs:object())
    monkeypatch.setattr(core.legacy,"_completion",lambda *args,**kwargs:"OK")
    def generate(client,source,op,seed):
        if core.OPERATION_SPECS[op]["family"]=="inv":return None,{"status":"rejected"}
        trace={"status":"accepted"}
        if op in core.COMPOSITE_STAGE_PLANS:trace["applied_steps"]=list(core.available_plans(source,op)[0])
        return source["hypothesis"]+f" {op} {seed}",trace
    monkeypatch.setattr(core,"generate_one",generate)
    report=workflow.run_generation(tiny_manifest(tmp_path),"train",tmp_path/"quota.jsonl",target=30)
    assert not report["generation_complete"]
    assert all(count<=2 for count in report["label_counts"].values())
    assert report["invariant_deficits"]=={"0":10,"1":8,"2":8}


def test_added_negative_unknown_fact_is_allowed_and_source_is_realized_by_code():
    value = plan()
    # An ignored hallucinated source field cannot replace the actual source proposition.
    value["edit_plan"]["source_clause"] = "An unrelated premise."
    text = core.parse_candidates(json.dumps([value]), SOURCE, "uninformative")
    assert text == ["A man carries a box and the man is not a doctor."]


def test_affirmative_contradiction_uses_shared_connector_pool():
    value = plan("the man is indoors", " and ", "contradiction")
    value["evidence_fact"] = "outdoors"
    assert core.parse_candidates(json.dumps([value]), SOURCE, "adding_contradiction") == [
        "A man carries a box and the man is indoors."]


@pytest.mark.parametrize("connector", [" if ", " provided that ", " when "])
def test_sufficient_or_ambiguous_conditions_are_rejected(connector):
    value = plan(connector=connector)
    value["edit_plan"]["condition_direction"] = "H_implies_C"
    assert core.parse_candidates(json.dumps([value]), SOURCE, "conditional_clause") == []


def test_necessary_condition_direction_is_verified():
    value = plan(connector=" only when ")
    value["edit_plan"]["condition_direction"] = "C_implies_H"
    assert core.parse_candidates(json.dumps([value]), SOURCE, "conditional_clause") == []
    value["edit_plan"]["condition_direction"] = "H_implies_C"
    assert core.parse_candidates(json.dumps([value]), SOURCE, "conditional_clause")


def test_punctuation_only_synonym_is_rejected():
    raw = json.dumps([{"hypothesis": "A man carries a box!", "target_label": "entailment"}])
    assert core.parse_candidates(raw, SOURCE, "synonym_replacement") == []


def test_inv_does_not_turn_a_neutral_source_into_entailment():
    source = {**SOURCE, "label": 1}
    assert core.target_label(source, "synonym_replacement") == "neutral"
    assert not core.eligible(source, "negation_flip")
    assert not core.eligible(source, "conditional_clause")


def test_composite_preserves_order_and_rejects_cancelling_edits(monkeypatch):
    calls = []
    def generate(client, source, operation, seed):
        calls.append(operation)
        text = SOURCE["hypothesis"] if len(calls) == 2 else "The man carries a box."
        return text, {"status": "accepted", "selected_hypothesis": text}
    monkeypatch.setattr(core, "generate_atomic", generate)
    text, log = core.generate_one(object(), SOURCE, "composite_inv")
    assert text is None and log["status"] == "no_effect"
    assert calls == log["applied_steps"]
    assert tuple(calls) in core.COMPOSITE_STAGE_PLANS["composite_inv"]


def tiny_manifest(tmp_path):
    sources = [{**SOURCE, "label": l, "pair_id": l,
                "hypothesis": f"The person carries box number {l}."} for l in range(3)]
    original = tmp_path/"original.jsonl"
    write_jsonl(original, sources)
    from nli_v3_4.data import sha256
    manifest = {"stage": "development", "datasets": {"train": {"input": str(original),
        "sha256": sha256(original), "schema": "train", "target_followups": 3,
        "development_pair_ids": [0,1,2], "development_original_followup_labels": {"0":1,"1":1,"2":1}}}}
    manifest["manifest_fingerprint"] = digest(manifest)
    path = tmp_path/"manifest.json";write_json(path, manifest)
    return path


def test_journalled_export_and_resume_do_not_regenerate_or_change_originals(tmp_path, monkeypatch):
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(core.legacy, "_completion", lambda *args, **kwargs: "OK")
    def generate(client, source, operation, seed):
        text = source["hypothesis"].rstrip(".") + " safely " + operation
        log = {"status": "accepted", "selected_hypothesis": text}
        if operation in core.COMPOSITE_STAGE_PLANS:
            log["applied_steps"] = list(core.COMPOSITE_STAGE_PLANS[operation][0])
        return text, log
    monkeypatch.setattr(core, "generate_one", generate)
    manifest = tiny_manifest(tmp_path)
    output = tmp_path/"generated.jsonl"
    report = workflow.run_generation(manifest, "train", output, target=3)
    rows, sources = load_dataset(output)
    assert report["generation_complete"] and report["overall_complete"] is False
    assert report["label_counts"] == {"0":1,"1":1,"2":1}
    assert report["external_nli_in_pipeline"] is False
    before = output.read_bytes()
    monkeypatch.setattr(core, "generate_one", lambda *args: pytest.fail("completed run should not regenerate"))
    workflow.run_generation(manifest, "train", output, target=3, resume=True)
    assert output.read_bytes() == before
    assert [canonical_source(r) for r in sources] == [canonical_source(r) for r in load_dataset(tmp_path/"original.jsonl")[1]]
    with pytest.raises(ValueError, match="configuration changed"):
        workflow.run_generation(manifest, "train", output, target=6, resume=True)


def test_entailment_inv_attempts_do_not_starve_neutral_inv_requests(tmp_path,monkeypatch):
    import openai
    from nli_v3_4.data import source_uid,export_followup
    monkeypatch.setattr(openai,"OpenAI",lambda **kwargs:object())
    monkeypatch.setattr(core.legacy,"_completion",lambda *args,**kwargs:"OK")
    manifest=tiny_manifest(tmp_path);output=tmp_path/"run.jsonl"
    workflow.run_generation(manifest,"train",output,target=6,limit_requests=0)
    source=load_dataset(tmp_path/"original.jsonl")[1][0];logs=[]
    for operation in ("synonym_replacement","voice_switch","composite_inv"):
        for refill in range(3):
            row=export_followup(source,operation,source["hypothesis"]+f" edit {refill}",0,"inv",None,"train") if operation=="synonym_replacement" and refill<2 else None
            logs.append({"request_id":digest([source_uid(source),operation,refill,42,"v3.4"]),
                "source_uid":source_uid(source),"pair_id":source["pair_id"],"operation":operation,
                "refill_round":refill,"target_label":0,"row":row,"trace":{"status":"accepted" if row else "rejected"}})
    write_jsonl(output.with_suffix(".generation_log.jsonl"),logs)
    calls=[]
    def generate(client,source,operation,seed):
        calls.append((source["label"],operation));return None,{"status":"rejected"}
    monkeypatch.setattr(core,"generate_one",generate)
    workflow.run_generation(manifest,"train",output,target=6,resume=True,limit_requests=1)
    assert calls[0][0]==1 and core.OPERATION_SPECS[calls[0][1]]["family"]=="inv"


def test_production_requires_real_frozen_gates(tmp_path):
    manifest = tiny_manifest(tmp_path)
    with pytest.raises(ValueError, match="requires frozen"):
        workflow.run_generation(manifest, "train", tmp_path/"prod.jsonl", stage="production")
    assert not (tmp_path/"prod.jsonl").exists()


def test_design_scorer_refuses_production_before_importing_or_running_models(tmp_path):
    output = tmp_path/"prod.jsonl"
    write_json(output.with_suffix(".run.json"), {"stage":"production", "dataset":"train"})
    manifest = tmp_path/"manifest.json";write_json(manifest,{"datasets":{}})
    with pytest.raises(ValueError, match="restricted to development"):
        list(development_rows(manifest,"train",output))


def test_quota_remainders_are_deterministic():
    assert quotas(3307) == {0:1103,1:1102,2:1102}
    assert sum(quotas(10425).values()) == 10425


def test_tfidf_never_fits_test_vocabulary_or_metadata():
    pytest.importorskip("sklearn")
    from analysis.hypothesis_only_logreg import fit_probe
    rows=[]
    for label,word in enumerate(("walking", "reading", "sleeping")):
        rows.extend([{"hypothesis": f"A person is {word}.", "label":label,
                      "premise":"SECRET_PREMISE", "mr_id":"SECRET_MR", "pair_id":"SECRET_PAIR"}]*3)
    probe,_=fit_probe(rows)
    probe.predict(["SECRET_TEST_WORD person is walking"])
    vocabulary=probe.named_steps["tfidf"].vocabulary_
    assert all(not w.startswith("secret") for w in vocabulary)
    mutated=[{**r,"premise":"different", "mr_id":"other", "pair_id":77} for r in rows]
    second,_=fit_probe(mutated)
    assert (probe.named_steps["lr"].coef_ == second.named_steps["lr"].coef_).all()


def test_human_review_includes_new_only_outputs_and_rejects_incomplete_answers(tmp_path):
    pytest.importorskip("numpy")
    from analysis.nli_v3_4_review import package, assess
    from nli_v3_4.data import DATASETS, export_source, export_followup
    old_root=tmp_path/"old";new_root=tmp_path/"new"
    for dataset in ("train",*DATASETS):
        schema="train" if dataset=="train" else "test"
        source=export_source(SOURCE,schema)
        common=export_followup(SOURCE,"synonym_replacement","A man transports a box.",0,"inv",None,schema)
        unmatched=export_followup(SOURCE,"synonym_replacement","A man carries a carton.",0,"inv",None,schema)
        for root,rows in ((old_root,[source,common]),(new_root,[source,common,unmatched])):
            path=root/f"{dataset}.jsonl";write_jsonl(path,rows)
            write_json(path.with_suffix(".run.json"),{"stage":"development",
                "source_sha256":digest([canonical_source(SOURCE)]),"contract_sha256":core.fingerprint()})
    review=tmp_path/"review";package(old_root,new_root,review)
    meta=json.loads((review/"manifest.json").read_text())
    assert meta["new_only_units"]==5
    answers=json.loads((review/"answers.template.json").read_text())
    write_json(tmp_path/"answers.json",answers)
    with pytest.raises(ValueError,match="named human"):
        assess(review,tmp_path/"answers.json",tmp_path/"quality.json")
    answers.update(reviewer="synthetic test fixture",human_attestation=True)
    for answer in answers["answers"]:
        answer.update(label=0,grammatical=True,operation_valid=True,source_equivalent=True)
    private=[json.loads(s) for s in (review/"private_key.jsonl").read_text().splitlines()]
    new_only=next(k["review_id"] for k in private if not k["paired"])
    next(a for a in answers["answers"] if a["review_id"]==new_only)["label"]=2
    write_json(tmp_path/"answers.json",answers)
    assess(review,tmp_path/"answers.json",tmp_path/"quality.json")
    quality=json.loads((tmp_path/"quality.json").read_text())
    assert quality["criteria"]["label"]["v3_4"]==.9
    assert quality["quality_passed"] is False
