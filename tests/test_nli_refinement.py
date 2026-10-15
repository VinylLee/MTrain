"""Protocol/identity tests, not evidence of real semantic or human quality."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from nli_v3_4 import core, refinement as r
from nli_v3_4.calls import DurableRequestClient
from nli_v3_4.data import write_jsonl

SOURCE = {"premise": "The man buys the book outdoors.", "hypothesis": "The man buys the book.", "label": 0, "pair_id": 7}
ENGINE = r.RefinementEngine()
VALUES = {"location": {"preposition": "in", "place": "a library"},
    "position_or_posture": {"posture": "sitting"}, "activity": {"action_ing": "reading a newspaper"},
    "possession": {"object_np": "a car"}, "material": {"material_np": "wood"},
    "shape_or_size": {"attribute": "tall"}, "colour": {"colour": "blue"},
    "clothing": {"garment_np": "a hat"}, "count": {"count": 2, "objects_np": "dogs"},
    "time_or_duration": {"age_years": 30}}
SYN = [{"label_rule": "inherit_source", "edit_plan": {"edits": [
    {"word_index": 2, "old_word": "buys", "replacement": "purchases"}]}}]


def nli(label): return {"label": label, "reason": "synthetic"}


def edit(add=False, composite=False, **changes):
    keys = r.COMMON + (r.ADDITION if add else ()) + (("ordered_components_valid",) if composite else ())
    return {**{k: True for k in keys}, "reason": "synthetic", **changes}


def backend(outputs, calls):
    iterator = iter(outputs)
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(iterator))))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def typed(source=SOURCE, i=0):
    domain = r.preference(source, i)["domain"]
    return {"abstain": False, "evidence_fact": "outdoors", "detail": VALUES[domain]}


@pytest.mark.parametrize("domain", list(VALUES))
def test_typed_renderer_controls_domain_and_polarity_without_changing_source(domain):
    original = deepcopy(SOURCE)
    for plural in (False, True):
        entity = {"reference": "The men" if plural else "The man", "plural": plural}
        for polarity in ("affirmative", "negative"):
            text = r.realize_detail(entity, domain, VALUES[domain], polarity)
            assert text.startswith(entity["reference"] + " ")
            assert core.legacy.has_negation(text) == (polarity == "negative")
    assert SOURCE == original
    with pytest.raises(ValueError): r.realize_detail(entity, domain, {"predicate": "feels sad"}, "affirmative")


def test_typed_noun_phrase_cannot_inject_subject_connector_or_negation():
    for value in ("a hat and she is happy", "a hat. He works", "no hat", "a hat\n"):
        with pytest.raises(ValueError): r.realize_detail(r.subject(SOURCE["hypothesis"]), "clothing", {"garment_np": value}, "affirmative")


def test_subject_subset_is_explicit_and_preserves_exact_anchor():
    assert r.subject("A man is outside.") == {"anchor": "A man", "reference": "The man", "plural": False}
    assert r.subject("Two children play a game.")["reference"] == "The two children"
    for h in ("A man with a hat.", "There are two children.", "Who is outside?", "The man who is outside works.", "No man is outside."):
        assert r.subject(h) is None


def test_shared_generator_and_invariant_prompts_are_blind_to_roles_and_gold():
    changed = {**SOURCE, "label": 2, "pair_id": 99, "corpus": "TEST"}
    for i in range(2):
        assert r.generation_prompt(SOURCE, "adding_contradiction", i) == r.generation_prompt(changed, "uninformative", i)
        assert r.generation_schema(SOURCE, "adding_contradiction", i) == r.generation_schema(changed, "uninformative", i)
    changed["premise"] = "SECRET_NOT_AVAILABLE"
    assert r.generation_prompt(SOURCE, "synonym_replacement") == r.generation_prompt(changed, "synonym_replacement")
    proposal = {"hypothesis": "The man purchases the book."}
    assert r.edit_request(SOURCE, "synonym_replacement", proposal) == r.edit_request(changed, "synonym_replacement", proposal)


def test_single_pair_inv_requires_both_directions_and_replays_exact_raw_calls():
    calls = []; api = backend([SYN, nli("entailment"), nli("entailment"), edit()], calls)
    text, trace = ENGINE.generate_one(api, SOURCE, "synonym_replacement")
    assert text == "The man purchases the book."
    assert [c["max_tokens"] for c in calls] == [768, 384, 384, 512]
    assert ENGINE.replay_one(SOURCE, "synonym_replacement", trace) == text
    assert all("outdoors" not in c["messages"][1]["content"] for c in calls)
    changed = deepcopy(trace); changed["attempts"][0]["calls"][2]["raw"] = json.dumps(nli("neutral"))
    with pytest.raises(ValueError): ENGINE.replay_one(SOURCE, "synonym_replacement", changed)


def test_one_way_synonym_does_not_pass_despite_true_edit_booleans():
    calls = []; api = backend([SYN, nli("entailment"), nli("neutral"), edit()] * 2, calls)
    text, trace = ENGINE.generate_one(api, SOURCE, "synonym_replacement")
    assert text is None and len(calls) == 8


@pytest.mark.parametrize("bad", [{"label": "E", "reason": "bad"}, {"label": ["entailment"], "reason": "bad"}, {"label": "entailment"}])
def test_invalid_nli_is_a_response_error_not_a_new_label(bad):
    calls = []; api = backend([SYN, bad, SYN, bad], calls)
    text, trace = ENGINE.generate_one(api, SOURCE, "synonym_replacement")
    assert text is None and all(a["error_kind"] == "response_format" for a in trace["attempts"])


def test_addition_requires_independent_fact_not_just_full_pair_contradiction():
    for relation in ("neutral", "contradiction"):
        outputs = [typed(), nli("entailment"), nli("contradiction"), nli("contradiction"), nli(relation), edit(add=True)]
        outputs += [typed(i=1), nli("entailment"), nli("contradiction"), nli("contradiction"), nli(relation), edit(add=True)]
        calls = []; text, trace = ENGINE.generate_one(backend(outputs, calls), SOURCE, "adding_contradiction")
        assert bool(text) == (relation == "neutral")
        if text:
            assert len(calls) == 6 and ENGINE.replay_one(SOURCE, "adding_contradiction", trace) == text
        else: assert len(calls) == 12


def test_missing_false_or_string_edit_check_cannot_be_invented():
    for bad in (edit(grammatical=False), edit(grammatical="true"), {"reason": "missing"}):
        calls = []; outputs = [SYN, nli("entailment"), nli("entailment"), bad] * 2
        text, _ = ENGINE.generate_one(backend(outputs, calls), SOURCE, "synonym_replacement")
        assert text is None


def test_interrupted_single_pair_consumes_original_slot_and_resumes_second_round(tmp_path):
    outputs = iter([SYN, SYN, nli("entailment"), nli("entailment"), edit()]); sent = []
    class API:
        def __init__(self): self.chat = self.completions = self
        def create(self, **kw):
            sent.append(kw)
            if len(sent) == 2: raise KeyboardInterrupt("synthetic interruption")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(outputs))))])
    api = API()
    with pytest.raises(KeyboardInterrupt): ENGINE.generate_one(DurableRequestClient(api, tmp_path), SOURCE, "synonym_replacement")
    text, trace = ENGINE.generate_one(DurableRequestClient(api, tmp_path), SOURCE, "synonym_replacement")
    assert text and len(sent) == 6 and len(list(tmp_path.glob('*.json'))) == 6
    assert trace['attempts'][0]['error_kind'] == 'interrupted_call'
    assert ENGINE.replay_one(SOURCE, 'synonym_replacement', trace) == text


def test_ordered_composite_requires_final_equivalence_and_raw_trajectory():
    voice = [{"label_rule": "inherit_source", "hypothesis": "The book is bought by the man."}]
    syn = [{"label_rule": "inherit_source", "edit_plan": {"edits": [{"word_index": 3, "old_word": "bought", "replacement": "purchased"}]}}]
    outputs = [voice, nli("entailment"), nli("entailment"), edit(),
        syn, nli("entailment"), nli("entailment"), edit(),
        nli("entailment"), nli("entailment"), edit(composite=True)]
    calls = []; text, trace = ENGINE.generate_one(backend(outputs, calls), SOURCE, "composite_inv")
    assert text == "The book is purchased by the man." and len(calls) == 11
    assert trace["applied_steps"] == ["voice_switch", "synonym_replacement"]
    assert ENGINE.replay_one(SOURCE, "composite_inv", trace) == text
    changed = deepcopy(trace); changed['applied_steps'].reverse()
    with pytest.raises((ValueError, RuntimeError)): ENGINE.replay_one(SOURCE, 'composite_inv', changed)


def test_new_pilot_strata_caps_identity_and_completed_resume(tmp_path, monkeypatch):
    import run_nli_v3_4_refinement as cli
    import openai
    sources = {d: [{**SOURCE, "pair_id": i, "hypothesis": f"The man buys {i} books."} for i in range(12)] for d in cli.previous.ORIGINS}
    monkeypatch.setattr(cli.previous, 'frozen_sources', lambda path: ({}, sources))
    manifest = tmp_path/'manifest.json'; manifest.write_text('{}')
    keys = tmp_path/'keys.jsonl'; write_jsonl(keys, [{"dataset": d, "pair_id": 0} for d in sources])
    path = tmp_path/'plan.json'; plan = cli.prepare(manifest, [keys], path)
    assert len(plan['requests']) == 25 and sum(x['max_calls'] for x in plan['requests']) == 295
    assert plan['max_output_tokens'] == sum(r.token_cap(x['operation']) for x in plan['requests']) == 144640
    assert all(x['pair_id'] != 0 for x in plan['requests'])
    cli.validate(path)
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: object())
    monkeypatch.setattr(cli.ENGINE, 'generate_one', lambda *args: (None, {'status': 'rejected'}))
    cli.run(path)
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: pytest.fail('completed run cannot dispatch'))
    cli.run(path, resume=True)
    state = json.loads((tmp_path/'execution/status.json').read_text())
    assert state['requests_complete'] and not state['quality_passed'] and not state['production_allowed']


def test_refinement_human_review_is_blind_and_cannot_freeze_draft(tmp_path):
    from analysis.nli_v3_4_review import package, HISTORICAL_DEVELOPMENT
    from nli_v3_4.data import export_source, export_followup, write_json, digest
    old, new = tmp_path/'old', tmp_path/'new'
    for d in ('train','snli','mnlim','mnlimm','sick'):
        schema = 'train' if d == 'train' else 'test'
        rows = [export_source(SOURCE,schema), export_followup(SOURCE,'synonym_replacement',
            'The man purchases the book.',0,'inv',None,schema)]
        for directory in (old,new): write_jsonl(directory/f'{d}.jsonl',rows)
        write_json(new/f'{d}.run.json',{'stage':'development','engine':'refinement',
            'source_sha256':digest([SOURCE]),'contract_sha256':ENGINE.fingerprint()})
    package(old,new,tmp_path/'review',stage=HISTORICAL_DEVELOPMENT)
    meta=json.loads((tmp_path/'review/manifest.json').read_text())
    assert meta['contract_sha256']==ENGINE.fingerprint() and not meta['freeze_eligible']
    answers=json.loads((tmp_path/'review/answers.template.json').read_text())
    assert not answers['human_attestation'] and not answers['reviewer']


def test_refinement_fact_contexts_replay_bound_calls_and_reject_clause_tampering(tmp_path):
    from analysis.design_nli_quality import addition_contexts
    from nli_v3_4.data import export_followup, write_json
    calls=[]; api=backend([typed(),nli('entailment'),nli('neutral'),nli('neutral'),nli('neutral'),edit(add=True)],calls)
    text,trace=ENGINE.generate_one(api,SOURCE,'uninformative')
    row=export_followup(SOURCE,'uninformative',text,1,'neutral',None,'train')
    generated=tmp_path/'train.jsonl';journal=tmp_path/'entries.json'
    write_json(generated.with_suffix('.run.json'),{'stage':'development','engine':'refinement','contract_sha256':ENGINE.fingerprint()})
    entry={'request':{'dataset':'train','operation':'uninformative','seed':42},'trace':trace,'row':row}
    write_json(journal,[entry])
    contexts=addition_contexts('train',generated,journal,[(row,SOURCE)])
    assert len(contexts)==1 and len(calls)==6
    entry['trace']['attempts'][0]['proposals'][0]['added_clause']='Someone is lying.'
    write_json(journal,[entry])
    with pytest.raises(ValueError,match='raw replay'):
        addition_contexts('train',generated,journal,[(row,SOURCE)])
