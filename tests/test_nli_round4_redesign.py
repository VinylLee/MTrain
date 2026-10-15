"""Synthetic protocol fixtures only; no real model/human quality is inferred."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from nli_v3_4 import core
from nli_v3_4 import redesign as r4
from nli_v3_4.calls import DurableRequestClient
from nli_v3_4.data import digest

SOURCE={"premise":"A man carries a box outdoors.","hypothesis":"A man carries a box.","label":0,"pair_id":7}
ENGINE=r4.Round4Engine()


def addition(source=SOURCE,operation="uninformative",clause=None,round_index=0):
    negative=r4.schedule(source["hypothesis"],round_index)[0]["polarity"]=="negative"
    clause=clause or ("The man does not own a car." if negative else "The man owns a car.")
    return {"slot":0,"source_is_complete_declarative":True,"entity_anchor":"A man",
            "evidence_fact":"","edit_plan":{"connector":" and ","added_clause":clause}}


def verdict(operation="uninformative",**changed):
    value={"index":0,"source_label":"entailment","label":"contradiction" if operation=="adding_contradiction" else "neutral",
        "source_equivalent":False,"grammatical":True,"operation_valid":True,
        "source_is_complete_declarative":True,"candidate_is_complete_declarative":True,
        "internally_consistent":True,"source_preserved":True,"one_fact_added":True,
        "anchor_unambiguous":True,"same_entity":True,"scheduled_domain_followed":True,
        "premise_to_added":"contradiction" if operation=="adding_contradiction" else "neutral",
        "source_to_added":"neutral","reason":"synthetic fixture"}
    return {**value,**changed}


def backend(outputs,calls):
    iterator=iter(outputs)
    def create(**kwargs):
        calls.append(kwargs);content=next(iterator)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_addition_proposals_are_blind_to_gold_mr_id_role_and_corpus():
    changed={**SOURCE,"label":2,"pair_id":900,"corpus":"test","idx":111,"is_source":True}
    for i in range(2):
        for seed in (42,43,44):
            expected=r4.build_generation_prompt(SOURCE,"uninformative",i,seed)
            assert expected==r4.build_generation_prompt(changed,"adding_contradiction",i,seed)
            assert '"target_label"' not in expected and '"source_label"' not in expected
            assert '"requested_operation"' not in expected and '"pair_id"' not in expected
    assert r4.generation_schema(SOURCE,"uninformative")==r4.generation_schema(SOURCE,"adding_contradiction")


def test_invariant_proposals_and_jury_do_not_read_premise_gold_or_identity():
    changed={**SOURCE,"premise":"SECRET_CHANGED_WORLD","label":2,"pair_id":99,"corpus":"test"}
    for op in ("synonym_replacement","voice_switch","pronoun_substitution"):
        assert r4.build_generation_prompt(SOURCE,op)==r4.build_generation_prompt(changed,op)
        assert r4.build_judge_prompt(SOURCE,op,[{"hypothesis":"A box is carried by a man."}])==r4.build_judge_prompt(changed,op,[{"hypothesis":"A box is carried by a man."}])
        assert "grammatical original fragment" not in r4.build_judge_prompt(SOURCE,op,[])


def test_synonym_schema_preserves_original_indices_and_protects_fixed_words():
    s={**SOURCE,"hypothesis":"The children buy two books in front of John."}
    slots=r4.editable_slots(s["hypothesis"])
    assert slots==[{"word_index":1,"word":"children"},{"word_index":2,"word":"buy"},{"word_index":4,"word":"books"}]
    choices=r4.generation_schema(s,"synonym_replacement")["json_schema"]["schema"]["items"]["properties"]["edit_plan"]["properties"]["edits"]["items"]["oneOf"]
    assert [c["properties"]["word_index"]["enum"][0] for c in choices]==[1,2,4]
    bad=[{"label_rule":"inherit_source","edit_plan":{"edits":[{"word_index":0,"old_word":"The","replacement":"A"}]}}]
    assert r4.parse_proposals(json.dumps(bad),s,"synonym_replacement")==[]
    good=[{"label_rule":"inherit_source","edit_plan":{"edits":[{"word_index":2,"old_word":"buy","replacement":"purchase"}]}}]
    assert r4.parse_proposals(json.dumps(good),s,"synonym_replacement")==[{"hypothesis":"The children purchase two books in front of John."}]


def test_fully_protected_source_can_abstain_without_requesting_a_forbidden_slot():
    s={**SOURCE,"hypothesis":"John is two."}
    assert not ENGINE.eligible(s,"synonym_replacement")
    assert r4.generation_schema(s,"synonym_replacement")["json_schema"]["schema"]["maxItems"]==0


@pytest.mark.parametrize("field",r4.COMMON_CHECKS+r4.ADDITION_CHECKS)
def test_boolean_claim_cannot_be_missing_false_or_a_string(field):
    for value in (False,"true",None):
        bad=verdict(**{field:value})
        assert not r4.parse_judgments_checked(json.dumps([bad]),SOURCE,"uninformative",[{"hypothesis":"candidate"}])


def test_nli_contradiction_is_not_enough_for_an_independent_contradiction_mr():
    for relation in ("contradiction","entailment"):
        bad=verdict("adding_contradiction",source_to_added=relation)
        assert not r4.parse_judgments_checked(json.dumps([bad]),SOURCE,"adding_contradiction",[{"hypothesis":"candidate"}])
    missing_fact=verdict("adding_contradiction",premise_to_added="neutral")
    assert not r4.parse_judgments_checked(json.dumps([missing_fact]),SOURCE,"adding_contradiction",[{"hypothesis":"candidate"}])
    assert r4.parse_judgments_checked(json.dumps([verdict("adding_contradiction")]),SOURCE,"adding_contradiction",[{"hypothesis":"candidate"}])


def test_common_surface_contract_rejects_duplicate_connector_unanchored_and_multiple_sentences():
    value=addition()
    for clause in ("and The man owns a car.","A stranger owns a car.","The man owns a car. The car is new."):
        value["edit_plan"]["added_clause"]=clause
        assert not r4.parse_proposals(json.dumps([value]),SOURCE,"uninformative")
    value=addition();assert len(r4.parse_proposals(json.dumps([value,value]),SOURCE,"uninformative"))==1


def test_judge_and_export_use_the_realized_text_and_replay_rechecks_proof():
    proposal=addition();calls=[]
    client=backend([json.dumps([proposal]),json.dumps([verdict()])],calls)
    original=deepcopy(SOURCE)
    text,trace=ENGINE.generate_one(client,SOURCE,"uninformative")
    assert text and SOURCE==original and len(calls)==2
    assert trace["attempts"][0]["proposals"][0]["hypothesis"]==text
    assert text in calls[1]["messages"][1]["content"]
    assert ENGINE.replay_one(SOURCE,"uninformative",trace)==text
    changed=deepcopy(trace)
    changed["attempts"][0]["raw_judgment"]=json.dumps([verdict(source_to_added="contradiction")])
    with pytest.raises(ValueError,match="cannot be replayed"):
        ENGINE.replay_one(SOURCE,"uninformative",changed)
    changed=deepcopy(trace);changed["contract_sha256"]=core.fingerprint()
    with pytest.raises(ValueError,match="contract/source/operation"):
        ENGINE.replay_one(SOURCE,"uninformative",changed)


def test_two_round_budget_and_durable_replay_do_not_reissue_calls(tmp_path):
    proposal=json.dumps([addition()]);second=json.dumps([addition(round_index=1)])
    bad=json.dumps([verdict(source_is_complete_declarative=False)])
    sent=[];api=backend([proposal,bad,second,bad],sent)
    for _ in range(2):
        text,trace=ENGINE.generate_one(DurableRequestClient(api,tmp_path/"calls"),SOURCE,"uninformative")
        assert text is None and len(trace["attempts"])==2
    assert len(sent)==4 and len(list((tmp_path/"calls").glob("*.json")))==4
    assert [c["max_tokens"] for c in sent]==[1000,1400,1000,1400]


def test_reference_points_back_to_the_existing_entity_without_editing_source():
    assert r4.reference_phrase("A man")=="the man"
    assert r4.reference_phrase("An actor")=="the actor"
    assert r4.reference_phrase("John")=="John"
    value=addition();value["edit_plan"]["added_clause"]=value["edit_plan"]["added_clause"].replace("The man","A man")
    assert not r4.parse_proposals(json.dumps([value]),SOURCE,"uninformative")


def test_composite_logs_ordered_intermediate_edits_and_replays_without_inference():
    source={**SOURCE,"hypothesis":"The man buys the book.","premise":"The man buys the book."}
    voice=[{"label_rule":"inherit_source","hypothesis":"The book is bought by the man."}]
    synonym=[{"label_rule":"inherit_source","edit_plan":{"edits":[{"word_index":3,"old_word":"bought","replacement":"purchased"}]}}]
    inv={"index":0,"source_to_candidate":"entailment","candidate_to_source":"entailment",
         "source_equivalent":True,"grammatical":True,"operation_valid":True,
         "source_is_complete_declarative":True,"candidate_is_complete_declarative":True}
    calls=[];api=backend(list(map(json.dumps,[voice,[inv],synonym,[inv],[{**inv,"ordered_components_valid":True}]])),calls)
    text,trace=ENGINE.generate_one(api,source,"composite_inv")
    assert text=="The book is purchased by the man." and len(calls)==5
    assert trace["applied_steps"]==["voice_switch","synonym_replacement"]
    assert ENGINE.replay_one(source,"composite_inv",trace)==text
    assert core.fingerprint()=="32fb80bff5d54589772782a2b145047a07e80ff8efefd6c2d0b1e99f542a807f"
    assert core.DEVELOPMENT_REVISION=="round3_hypothesis_edit_plan"


@pytest.fixture
def round4_plan(tmp_path,monkeypatch):
    import run_nli_v3_4_redesign as cli
    from nli_v3_4.data import write_jsonl
    sources={d:[{**SOURCE,"pair_id":i,"hypothesis":f"The man carries {i} boxes."} for i in range(14)] for d in cli.previous.ORIGINS}
    monkeypatch.setattr(cli.previous,"frozen_sources",lambda path:({},sources))
    manifest=tmp_path/"manifest.json";manifest.write_text('{}')
    keys=[]
    for i in (0,1):
        path=tmp_path/f"key{i}.jsonl";write_jsonl(path,[{"dataset":d,"pair_id":i} for d in cli.previous.ORIGINS]);keys.append(path)
    path=tmp_path/"plan.json";cli.prepare(manifest,keys,path)
    return cli,path,sources


def test_new_plan_combines_all_previous_review_exposure_and_declares_token_budget(round4_plan):
    cli,path,_=round4_plan;plan,_=cli.validate(path)
    assert len(plan["requests"])==50 and sum(v["max_calls"] for v in plan["requests"])==250
    assert all(v["pair_id"] not in {0,1} for v in plan["requests"])
    assert plan["max_output_tokens"]==40*(2*1000+2*1400)+10*(4*1000+5*1400)
    assert not __import__('pathlib').Path(plan["run_output"]).exists()


def test_finished_new_plan_does_not_reset_budget_or_construct_client_on_resume(round4_plan,monkeypatch):
    import openai
    from pathlib import Path
    cli,path,sources=round4_plan;executed=[]
    monkeypatch.setattr(openai,"OpenAI",lambda **kw:object())
    monkeypatch.setattr(cli.ENGINE,"generate_one",lambda client,source,op,seed:executed.append((source,op)) or (None,{"status":"rejected"}))
    cli.run(path)
    plan=json.loads(path.read_text());out=Path(plan["run_output"])
    assert len(executed)==50
    status=json.loads((out/"status.json").read_text())
    assert status["requests_complete"] and not status["quality_passed"] and not status["production_allowed"]
    monkeypatch.setattr(openai,"OpenAI",lambda **kw:pytest.fail("completed requests must not dispatch"))
    cli.run(path,resume=True)
    assert len(executed)==50


def test_new_budget_identity_changes_fail_before_network(round4_plan,tmp_path,monkeypatch):
    import openai
    cli,path,_=round4_plan
    monkeypatch.setattr(openai,"OpenAI",lambda **kw:pytest.fail("no network before validation"))
    plan=json.loads(path.read_text());plan.pop("plan_fingerprint");plan["max_model_calls"]=251
    plan["plan_fingerprint"]=digest(plan);path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match="scope/budget changed"):cli.run(path)


def test_interrupted_new_jury_consumes_slot_and_uses_only_the_second_round(tmp_path):
    sent=[]
    class StopOnce:
        def __init__(self):self.chat=self.completions=self
        def create(self,**kw):
            sent.append(kw)
            if len(sent)==2:raise KeyboardInterrupt("synthetic crash")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps([addition()])))])
    with pytest.raises(KeyboardInterrupt):
        ENGINE.generate_one(DurableRequestClient(StopOnce(),tmp_path/"calls"),SOURCE,"uninformative")
    resumed=backend([json.dumps([addition(round_index=1)]),json.dumps([verdict()])],sent)
    client=DurableRequestClient(resumed,tmp_path/"calls")
    text,trace=ENGINE.generate_one(client,SOURCE,"uninformative")
    assert text and len(sent)==4 and len(trace["attempts"])==2
    assert trace["attempts"][0]["error_kind"]=="interrupted_call"
    assert (client.new_calls,client.replayed_calls,client.uncertain_calls)==(2,1,1)


def test_round4_review_isolated_from_final_freeze_and_records_its_contract(tmp_path):
    from analysis.nli_v3_4_review import package,HISTORICAL_DEVELOPMENT
    from nli_v3_4.data import export_source,export_followup,write_jsonl,write_json
    old,new=tmp_path/"old",tmp_path/"new"
    for origin in ("train","snli","mnlim","mnlimm","sick"):
        schema="train" if origin=="train" else "test"
        s=export_source(SOURCE,schema)
        f=export_followup(SOURCE,"uninformative","A man carries a box and the man owns a car.",1,"neutral",None,schema)
        for root in (old,new):write_jsonl(root/f"{origin}.jsonl",[s,f])
        write_json(new/f"{origin}.run.json",{"stage":"development","engine":"round4",
            "contract_sha256":ENGINE.fingerprint(),"source_sha256":digest([SOURCE])})
    package(old,new,tmp_path/"review",stage=HISTORICAL_DEVELOPMENT)
    manifest=json.loads((tmp_path/"review/manifest.json").read_text())
    assert manifest["contract_sha256"]==ENGINE.fingerprint()!=core.fingerprint()
    assert manifest["freeze_eligible"] is False
    # Even giving the baseline generated metadata must not make the draft a final gate.
    for origin in ("train","snli","mnlim","mnlimm","sick"):
        write_json(old/f"{origin}.run.json",{"stage":"development","contract_sha256":core.fingerprint(),"source_sha256":digest([SOURCE])})
    with pytest.raises(ValueError,match="draft round-4 output"):
        package(old,new,tmp_path/"wrong_scope",stage="development")


def test_nli_fact_diagnostics_bind_raw_journal_and_never_invent_an_added_clause(tmp_path):
    from analysis.design_nli_quality import addition_contexts
    from nli_v3_4.data import export_followup,write_json
    calls=[];client=backend([json.dumps([addition()]),json.dumps([verdict()])],calls)
    text,trace=ENGINE.generate_one(client,SOURCE,"uninformative")
    row=export_followup(SOURCE,"uninformative",text,1,"neutral",None,"train")
    generated=tmp_path/"train.jsonl"
    write_json(generated.with_suffix(".run.json"),{"stage":"development","engine":"round4","contract_sha256":ENGINE.fingerprint()})
    journal=tmp_path/"entries.json"
    entry={"request":{"dataset":"train","operation":"uninformative","seed":42},"row":row,"trace":trace}
    write_json(journal,[entry])
    contexts=addition_contexts("train",generated,journal,[(row,SOURCE)])
    assert list(contexts.values())==[addition()["edit_plan"]["added_clause"]]
    assert len(calls)==2  # Offline journal replay issued no NLI or generator calls.
    entry["trace"]["attempts"][0]["proposals"][0]["added_clause"]="Unbound claim."
    write_json(journal,[entry])
    with pytest.raises(ValueError,match="proposals differ"):
        addition_contexts("train",generated,journal,[(row,SOURCE)])
