"""Engineering profile identity, bounded resume and preliminary-review scope."""
import json
from pathlib import Path
import pytest

import run_nli_v3_4_refinement as original
from run_nli_v3_4_article_repair import configured_pilot
from nli_v3_4.article_guard import ArticleCheckedEngine
from nli_v3_4.data import digest, export_source, export_followup, write_json, write_jsonl

SOURCE = {"premise": "The man buys the book outdoors.", "hypothesis": "The man buys the book.", "label": 0, "pair_id": 7}


def test_profile_changes_are_scoped_and_old_hashes_are_preserved():
    before = (original.ENGINE, original.KIND, original.REVISION, original.ENGINE_NAME, original.code_hashes())
    with configured_pilot() as runner:
        assert isinstance(runner.ENGINE, ArticleCheckedEngine)
        assert runner.ENGINE_NAME == "refinement_v2_article"
        assert all(runner.code_hashes()[key] == value for key, value in before[-1].items())
        assert "run_nli_v3_4_article_repair.py" in runner.code_hashes()
        assert "nli_v3_4/article_guard.py" in runner.code_hashes()
    assert (original.ENGINE, original.KIND, original.REVISION, original.ENGINE_NAME, original.code_hashes()) == before
    with pytest.raises(RuntimeError), configured_pilot():
        raise RuntimeError("synthetic command error")
    assert original.ENGINE is before[0]


def test_preparation_excludes_all_three_review_batches_and_completed_resume_dispatches_nothing(tmp_path, monkeypatch):
    import openai
    sources = {d: [{**SOURCE, "pair_id": i, "hypothesis": f"The man buys {i} books."} for i in range(14)] for d in original.previous.ORIGINS}
    monkeypatch.setattr(original.previous, "frozen_sources", lambda path: ({}, sources))
    manifest = tmp_path/"manifest.json"; manifest.write_text("{}")
    keys = []
    for i in (0, 1, 2):
        key = tmp_path/f"review{i}.jsonl"
        write_jsonl(key, [{"dataset": d, "pair_id": i} for d in sources]); keys.append(key)
    with configured_pilot() as runner:
        path = tmp_path/"plan.json"; plan = runner.prepare(manifest, keys, path)
        assert len(plan["requests"]) == 25 and all(q["pair_id"] not in {0, 1, 2} for q in plan["requests"])
        assert plan["max_model_calls"] == 295 and plan["max_output_tokens"] == 144640
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: object())
        monkeypatch.setattr(runner.ENGINE, "generate_one", lambda *args: (None, {"status": "rejected"}))
        runner.run(path)
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: pytest.fail("completed requests must not dispatch"))
        runner.run(path, resume=True)
        status = json.loads((tmp_path/"execution/status.json").read_text())
        assert status["requests_complete"] and status["engine"] == "refinement_v2_article"
        assert not status["quality_passed"] and not status["production_allowed"]


def test_article_profile_cannot_be_used_as_final_freeze_review(tmp_path):
    from analysis.nli_v3_4_review import package, HISTORICAL_DEVELOPMENT
    old, new = tmp_path/"old", tmp_path/"new"
    for d in original.previous.ORIGINS:
        schema = "train" if d == "train" else "test"
        rows = [export_source(SOURCE, schema), export_followup(SOURCE, "synonym_replacement",
            "The man purchases the book.", 0, "inv", None, schema)]
        for root in (old, new): write_jsonl(root/f"{d}.jsonl", rows)
        write_json(new/f"{d}.run.json", {"stage": "development", "engine": "refinement_v2_article",
            "contract_sha256": ArticleCheckedEngine.fingerprint(), "source_sha256": digest([SOURCE])})
        write_json(old/f"{d}.run.json", {"stage": "development", "source_sha256": digest([SOURCE])})
    package(old, new, tmp_path/"review", stage=HISTORICAL_DEVELOPMENT)
    result = json.loads((tmp_path/"review/manifest.json").read_text())
    assert result["contract_sha256"] == ArticleCheckedEngine.fingerprint() and result["freeze_eligible"] is False
    with pytest.raises(ValueError, match="preliminary"):
        package(old, new, tmp_path/"wrong_scope", stage="development")


def test_design_diagnostics_bind_article_engine_to_raw_response_journal(tmp_path):
    from analysis.design_nli_quality import addition_contexts
    from test_nli_refinement import backend, nli
    from test_nli_refinement_v2 import typed, edit
    calls = []
    api = backend([typed(), nli("entailment"), nli("neutral"), nli("neutral"), nli("neutral"), edit(add=True)], calls)
    engine = ArticleCheckedEngine()
    text, trace = engine.generate_one(api, SOURCE, "uninformative")
    assert text and len(calls) == 6
    row = export_followup(SOURCE, "uninformative", text, 1, "neutral", None, "train")
    file, journal = tmp_path/"train.jsonl", tmp_path/"entries.json"
    write_json(file.with_suffix(".run.json"), {"stage": "development", "engine": "refinement_v2_article",
        "contract_sha256": engine.fingerprint()})
    entry = {"request": {"dataset": "train", "operation": "uninformative", "seed": 42}, "row": row, "trace": trace}
    write_json(journal, [entry])
    contexts = addition_contexts("train", file, journal, [(row, SOURCE)])
    assert len(contexts) == 1 and len(calls) == 6
    entry["trace"]["attempts"][0]["proposals"][0]["added_clause"] = "A made-up claim."
    write_json(journal, [entry])
    with pytest.raises(ValueError, match="raw replay"):
        addition_contexts("train", file, journal, [(row, SOURCE)])
