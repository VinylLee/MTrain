"""Explicit kind/grammar and typed candidate tests; no semantic population claims."""
from copy import deepcopy
import json
import pytest
from nli_v3_4 import refinement as base, refinement_v2 as r
from test_nli_refinement import SOURCE,SYN,VALUES,backend,nli

ENGINE=r.DirectTypedEngine()


def edit(add=False,**changes):
    other=set(base.COMMON+(base.ADDITION if add else ()))-{'grammatical','source_is_complete_declarative','candidate_is_complete_declarative'}
    return {'source':{'kind':'statement','grammatical':True},'candidate':{'kind':'statement','grammatical':True},
        **{k:True for k in other},'reason':'synthetic',**changes}


def typed(i=0):return {'detail':VALUES[base.preference(SOURCE,i)['domain']]}


def test_v2_shared_generation_has_no_abstention_or_quote_proxy_and_controls_polarity():
    changed={**SOURCE,'label':2,'pair_id':999,'corpus':'TEST'}
    for i in range(2):
        assert r.generation_prompt(SOURCE,'adding_contradiction',i)==r.generation_prompt(changed,'uninformative',i)
        assert r.generation_schema(SOURCE,'adding_contradiction',i)==r.generation_schema(changed,'uninformative',i)
        c=r.parse_proposals(json.dumps(typed(i)),SOURCE,'adding_contradiction',i)
        n=r.parse_proposals(json.dumps(typed(i)),SOURCE,'uninformative',i)
        assert c==n and c and c[0]['evidence_fact']==''


def test_complete_grammar_is_not_invented_from_sentence_kind():
    checks=base.COMMON
    valid=ENGINE.normalize_edit(edit(),checks)
    assert valid['grammatical'] and valid['source_is_complete_declarative']
    invalid=ENGINE.normalize_edit(edit(source={'kind':'statement','grammatical':False}),checks)
    assert not invalid['grammatical'] and invalid['source_is_complete_declarative']
    fragment=ENGINE.normalize_edit(edit(source={'kind':'fragment','grammatical':True}),checks)
    assert not fragment['source_is_complete_declarative']
    with pytest.raises(ValueError):ENGINE.normalize_edit(edit(source={'kind':'statement','grammatical':'true'}),checks)


def test_v2_true_two_way_edit_replays_and_agreement_failure_still_rejects():
    calls=[];text,trace=ENGINE.generate_one(backend([SYN,nli('entailment'),nli('entailment'),edit()],calls),SOURCE,'synonym_replacement')
    assert text and trace['revision']==r.REVISION and ENGINE.replay_one(SOURCE,'synonym_replacement',trace)==text
    calls=[];bad=edit(candidate={'kind':'statement','grammatical':False})
    text,trace=ENGINE.generate_one(backend([SYN,nli('entailment'),nli('entailment'),bad]*2,calls),SOURCE,'synonym_replacement')
    assert text is None and len(calls)==8


def test_v2_realizes_candidates_but_does_not_force_jury_acceptance():
    calls=[];rounds=[]
    for i in range(2):rounds.extend([typed(i),nli('entailment'),nli('contradiction'),nli('contradiction'),nli('contradiction'),edit(add=True)])
    text,trace=ENGINE.generate_one(backend(rounds,calls),SOURCE,'adding_contradiction')
    assert text is None and len(calls)==12  # H contradicts A; the MR remains invalid.


def test_voice_is_an_explicit_subset_not_a_new_definition():
    assert ENGINE.eligible(SOURCE,'voice_switch')
    assert ENGINE.eligible({**SOURCE,'hypothesis':'Two men are playing frisbee at the park.'},'voice_switch')
    for h in ('The dog was hungry.','A man is outside.','The dog runs fast.'):
        assert not ENGINE.eligible({**SOURCE,'hypothesis':h},'voice_switch')
    assert base.RefinementEngine().fingerprint()=='c96af4172015d883e0370bb7cc5c0a8440bd0564f8b01cceda99d09c38d3b26c'


def test_profile_selection_does_not_happen_at_import_and_preserves_caps(monkeypatch,tmp_path):
    import run_nli_v3_4_refinement as cli
    from nli_v3_4.data import write_jsonl
    try:
        cli.configure('v2');assert cli.ENGINE_NAME=='refinement_v2' and cli.REVISION==r.REVISION
        sources={d:[{**SOURCE,'pair_id':i,'hypothesis':f'The man buys {i} books.'} for i in range(12)] for d in cli.previous.ORIGINS}
        monkeypatch.setattr(cli.previous,'frozen_sources',lambda path:({},sources))
        manifest=tmp_path/'manifest.json';manifest.write_text('{}');keys=tmp_path/'keys.jsonl';write_jsonl(keys,[])
        plan=cli.prepare(manifest,[keys],tmp_path/'plan.json');cli.validate(tmp_path/'plan.json')
        assert plan['engine']=='refinement_v2' and plan['max_model_calls']==295 and plan['max_output_tokens']==144640
    finally:cli.configure('v1')
