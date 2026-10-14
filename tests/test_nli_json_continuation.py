"""Saved malformed JSON can recover only unchanged proposals and unused judges."""
from collections import Counter
from copy import deepcopy
from itertools import permutations
import json

import pytest

from nli_v3_4 import core
from nli_v3_4.protocol import normalize_generator_json_suffix
from nli_v3_4.recovery import NeedsInference, OfflineClient

SOURCE = {"premise": "The children buy books.", "hypothesis": "The children buy books.",
          "label": 1, "pair_id": 7}
VALUE = [{"label_rule": "inherit_source", "edit_plan": {"edits": [
    {"word_index": 2, "old_word": "buy", "replacement": "purchase"}]}}]
RAW = json.dumps(VALUE, separators=(",", ":"))
BAD = RAW[:-5] + "}]}]}"
VERDICT = {"index": 0, "source_to_candidate": "entailment", "candidate_to_source": "entailment",
           "source_equivalent": True, "grammatical": True, "operation_valid": True}


def prior():
    return {"operation": "synonym_replacement", "attempts": [
        {"round": 0, "raw_generation": BAD, "status": "invalid_response", "error_kind": "response_format",
         "model_call_events": [{"role": "generation", "max_tokens": 700}]},
        {"round": 1, "raw_generation": "[]", "candidates": [], "status": "rejected",
         "model_call_events": [{"role": "generation", "max_tokens": 700}]}]}


def test_every_closer_order_preserves_all_json_values_and_strict_indices():
    suffix = RAW[-5:]
    for order in set(permutations(suffix)):
        malformed = RAW[:-5] + "".join(order)
        normalized, changed = normalize_generator_json_suffix(malformed)
        assert json.loads(normalized) == VALUE
        assert normalized[:len(RAW)-5] == RAW[:-5]
        assert Counter(normalized[-5:]) == Counter(malformed[-5:])
        assert core.parse_candidates(malformed, SOURCE, "synonym_replacement") == [
            "The children purchase books."]
    wrong = deepcopy(VALUE)
    wrong[0]["edit_plan"]["edits"][0]["word_index"] = 1
    raw = json.dumps(wrong, separators=(",", ":"))
    assert core.parse_candidates(raw[:-5] + "}]}]}", SOURCE, "synonym_replacement") == []


@pytest.mark.parametrize("raw", [RAW[:-1], RAW[:-3], BAD+'}', BAD+' trailing',
                               '[{"edit_plan" [{"broken": "value"}]}]',
                               '[{"value":"unclosed }]}'])
def test_incomplete_content_and_earlier_structure_are_not_invented(raw):
    normalized, changed = normalize_generator_json_suffix(raw)
    assert not changed
    assert normalized == raw


def test_brackets_inside_strings_remain_literal():
    value = [{"content": 'A quote " and brackets [ } must remain.'}]
    raw = json.dumps(value, separators=(",", ":"))
    normalized, changed = normalize_generator_json_suffix(raw[:-2]+']}')
    assert changed and json.loads(normalized) == value


def test_saved_proposal_calls_only_its_unused_judge_and_never_regenerates(monkeypatch):
    calls = []
    def call(client, system, prompt, temperature, tokens):
        calls.append((system, prompt, temperature, tokens))
        assert system == core.EQUIVALENCE_SYSTEM
        return json.dumps([VERDICT])
    monkeypatch.setattr(core, "_call", call)
    saved = prior()
    before = deepcopy(saved)
    text, trace = core.generate_atomic(object(), SOURCE, "synonym_replacement", prior_trace=saved)
    assert text == "The children purchase books." and len(calls) == 1
    assert len(trace["attempts"]) == 2 and saved == before
    assert trace["attempts"][0]["raw_generation"] == BAD
    assert trace["attempts"][0]["generation_json_normalization"]["literals_and_indices_changed"] is False


def test_dispatched_or_unproven_judge_budget_cannot_be_retried(monkeypatch):
    monkeypatch.setattr(core, "_call", lambda *a, **kw: pytest.fail("no unused judge slot"))
    for kind in ("dispatched", "unknown"):
        saved = prior()
        if kind == "dispatched":
            saved["attempts"][0]["model_call_events"].append({"role": "invariant_equivalence_judge"})
        else:
            saved["attempts"][0].pop("model_call_events")
        text, trace = core.generate_atomic(object(), SOURCE, "synonym_replacement", prior_trace=saved)
        assert text is None and len(trace["attempts"]) == 2


def test_recovered_shape_still_requires_semantic_equivalence(monkeypatch):
    monkeypatch.setattr(core, "_call", lambda *a, **kw: json.dumps([
        {**VERDICT, "candidate_to_source": "neutral", "source_equivalent": False}]))
    text, _ = core.generate_atomic(object(), SOURCE, "synonym_replacement", prior_trace=prior())
    assert text is None


def test_offline_migration_marks_real_unused_work_instead_of_accepting_it():
    with pytest.raises(NeedsInference):
        core.generate_atomic(OfflineClient(), SOURCE, "synonym_replacement", prior_trace=prior())
