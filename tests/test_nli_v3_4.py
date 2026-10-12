"""Research invariants: isolation, source preservation, quotas and fail-closed gates."""
import json
from pathlib import Path

import pytest

pytest.importorskip("openai")
from nli_v3_4 import core, workflow
from nli_v3_4.data import canonical_source, digest, load_dataset, quotas, write_json, write_jsonl
from analysis.design_nli_quality import development_rows

SOURCE = {"premise": "A man in a red shirt is carrying a box outdoors.",
          "hypothesis": "A man carries a box.", "label": 0, "pair_id": 7}


def plan(added="the man is not a doctor", connector=" and ", label="neutral"):
    return {"target_label": label, "edit_plan": {"added_clause": added, "connector": connector}}


def test_roles_and_corpus_metadata_cannot_change_prompt_or_style():
    train = {**SOURCE, "is_syn": False, "corpus": "training", "mr_id": "none"}
    test = {**SOURCE, "is_source": True, "corpus": "test", "idx": 100}
    for op in core.OPERATION_SPECS:
        assert core.build_generation_prompt(train, op) == core.build_generation_prompt(test, op)
        assert core.style_for(train, op, 1) == core.style_for(test, op, 1)


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
