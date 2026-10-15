"""New diagnostic budgets stay bounded, isolated and distinct from final gates."""
import json
from types import SimpleNamespace

import pytest

import run_nli_quality_repair_pilot as pilot
from analysis.nli_v3_4_review import HISTORICAL_DEVELOPMENT, package, assess
from nli_v3_4 import core
from nli_v3_4.data import digest, export_source, read_jsonl, write_json, write_jsonl


def completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{}')
    rows = {d: [{"premise": f"A manager carries {i} red boxes outdoors.",
                 "hypothesis": f"A manager carries {i} boxes.", "label": 0, "pair_id": i}
                for i in range(14)] for d in pilot.ORIGINS}
    monkeypatch.setattr(pilot, "frozen_sources", lambda p: ({}, rows))
    key = tmp_path / "key.jsonl"
    write_jsonl(key, [{"dataset": d, "pair_id": 0} for d in pilot.ORIGINS])
    path = tmp_path / "plan.json"
    pilot.prepare(manifest, key, path)
    return path, rows


def test_preparation_uses_unique_unreviewed_sources_and_exact_budget(prepared, tmp_path):
    path, _ = prepared
    plan, _ = pilot.validate(path)
    assert len(plan["requests"]) == 50
    assert sum(r["max_calls"] for r in plan["requests"]) == 250
    assert len({(r["dataset"], r["pair_id"]) for r in plan["requests"]}) == 50
    assert all(r["pair_id"] != 0 for r in plan["requests"])
    assert not list(tmp_path.rglob("*.calls"))


def test_dispatch_budget_is_hard_and_replay_does_not_spend_it_again(tmp_path):
    sent = []
    api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: sent.append(kw) or completion("saved response"))))
    for _ in range(2):
        client = pilot.BoundedRequestClient(api, tmp_path / "calls", 4)
        for index in range(4):
            assert client.create(prompt=f"call {index}").choices[0].message.content == "saved response"
        with pytest.raises(RuntimeError, match="budget exhausted"):
            client.create(prompt="unbudgeted")
    assert len(sent) == 4
    assert len(list((tmp_path / "calls").glob("*.json"))) == 4


def test_changed_budget_rejected_before_client_or_output_creation(prepared, tmp_path, monkeypatch):
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: pytest.fail("unexpected client construction"))
    path, _ = prepared
    plan = json.loads(path.read_text())
    plan.pop("plan_fingerprint"); plan["max_model_calls"] = 251
    plan["plan_fingerprint"] = digest(plan); write_json(path, plan)
    output = tmp_path / "execution"
    with pytest.raises(ValueError, match="budget"):
        pilot.run(path, output)
    assert not output.exists()


def test_completed_requests_are_not_run_again_and_sources_are_retained(prepared, tmp_path, monkeypatch):
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: object())
    path, sources = prepared
    executed = []
    def reject(client, source, operation, seed):
        executed.append((source, operation)); return None, {"status": "rejected"}
    monkeypatch.setattr(core, "generate_one", reject)
    output = tmp_path / "execution"
    pilot.run(path, output)
    assert len(executed) == 50
    for origin in pilot.ORIGINS:
        rows = read_jsonl(output / "v3_4" / f"{origin}.jsonl")
        assert [{k: r[k] for k in ("premise", "hypothesis", "label", "pair_id")} for r in rows] == sources[origin]
    status = json.loads((output / "status.json").read_text())
    assert status["requests_complete"] is True
    assert status["quality_passed"] is False and status["production_allowed"] is False
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: pytest.fail("finished pilot must not create client"))
    pilot.run(path, output, resume=True)
    assert len(executed) == 50


def test_review_exposure_change_prevents_new_run(prepared, tmp_path):
    path, _ = prepared
    plan = json.loads(path.read_text())
    from pathlib import Path
    Path(plan["previous_review_key"]).write_text('[]')
    with pytest.raises(ValueError, match="review exposure changed"):
        pilot.validate(path)


def test_budget_cannot_be_restarted_in_another_output_directory(prepared, tmp_path):
    path, _ = prepared
    with pytest.raises(ValueError, match="single planned output"):
        pilot.run(path, tmp_path / "another_execution")
    assert not (tmp_path / "another_execution").exists()


def test_micro_pilot_review_keeps_preliminary_historical_scope(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    source = {"premise": "The children buy books.", "hypothesis": "The children buy books.", "label": 0, "pair_id": 7}
    for origin in pilot.ORIGINS:
        original = export_source(source, "train" if origin == "train" else "test")
        followup = {**original, "mr_id": "synonym_replacement", "hypothesis": "The children purchase books."}
        followup["is_syn" if origin == "train" else "is_source"] = origin == "train"
        write_jsonl(old / f"{origin}.jsonl", [original, followup])
        write_jsonl(new / f"{origin}.jsonl", [original, followup])
        write_json(new / f"{origin}.run.json", {"stage": "development", "kind": pilot.KIND,
            "source_sha256": digest([source]), "contract_sha256": core.fingerprint()})
    package(old, new, tmp_path / "review", stage=HISTORICAL_DEVELOPMENT)
    manifest = json.loads((tmp_path / "review/manifest.json").read_text())
    assert manifest["freeze_eligible"] is False
    assert manifest["stage"] == HISTORICAL_DEVELOPMENT
    assert manifest["pairs"] == 5
    with pytest.raises(ValueError):
        package(old, new, tmp_path / "wrong_scope", stage="development")


def test_unpaired_invariant_quality_is_reported_without_fabricated_comparison(tmp_path):
    """All data/answers here are synthetic fixtures, never human experiment input."""
    source = {"premise":"The children buy books.","hypothesis":"The children buy books.","label":0,"pair_id":7}
    old, new = tmp_path/"old", tmp_path/"new"
    for origin in pilot.ORIGINS:
        original = export_source(source,"train" if origin=="train" else "test")
        followup = {**original,"mr_id":"synonym_replacement","hypothesis":"The children purchase books."}
        followup["is_syn" if origin=="train" else "is_source"] = origin=="train"
        write_jsonl(old/f"{origin}.jsonl",[original])
        write_jsonl(new/f"{origin}.jsonl",[original,followup])
        write_json(new/f"{origin}.run.json",{"stage":"development","kind":pilot.KIND,
            "source_sha256":digest([source]),"contract_sha256":core.fingerprint()})
    review=tmp_path/"review"
    package(old,new,review,stage=HISTORICAL_DEVELOPMENT)
    answers=json.loads((review/"answers.template.json").read_text())
    answers.update(reviewer="synthetic test fixture",human_attestation=True)
    for a in answers["answers"]:
        a.update(label=0,grammatical=True,operation_valid=True,source_equivalent=True)
    write_json(tmp_path/"answers.json",answers)
    assess(review,tmp_path/"answers.json",tmp_path/"assessment.json")
    result=json.loads((tmp_path/"assessment.json").read_text())
    assert set(result["criteria"])=={"label","grammar","operation","inv_equivalence"}
    criterion=result["criteria"]["inv_equivalence"]
    assert criterion["v3_4"]==1 and criterion["v3_4_n"]==5
    assert criterion["v3_3"] is None and criterion["v3_3_n"]==0
    assert criterion["difference_ci95"] is None
    assert criterion["comparison_status"]=="no_paired_reference"
    assert criterion["passes"] is False and result["quality_passed"] is False
    with (review/"operation_blind.jsonl").open("a") as stream:
        stream.write('\n')
    with pytest.raises(ValueError,match="blind review data changed"):
        assess(review,tmp_path/"answers.json",tmp_path/"changed.json")
