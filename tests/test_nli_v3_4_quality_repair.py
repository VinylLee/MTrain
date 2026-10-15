"""Regression cases for implementation defects found in human development review."""
import json

import pytest

from nli_v3_4 import core
from nli_v3_4.invariants import realize_synonym_plan, voice_preserves_form
from nli_v3_4.protocol import replay_atomic
from nli_v3_4.realization import realize_added_clause


SOURCE = {"premise": "A man in a red shirt carries a box.",
          "hypothesis": "A man carries a box.", "label": 0, "pair_id": 7}


@pytest.mark.parametrize("clause,expected", [
    ("The dog is not white.", "the dog is not white"),
    ("She wears a coat.", "she wears a coat"),
    ("There is a sign.", "there is a sign"),
    ("A child owns it.", "a child owns it"),
    ("I own it.", "I own it"),
    ("John owns it.", "John owns it"),
    ("NASA owns it.", "NASA owns it"),
    ("The Beatles own it.", "The Beatles own it"),
    ('"She" is printed on it.', '"She" is printed on it'),
])
def test_clause_case_is_mechanical_and_preserves_names(clause, expected):
    for connector in (" and ", "; ", " only if "):
        assert realize_added_clause("A dog runs.", connector, clause) == "A dog runs" + connector + expected + "."


@pytest.mark.parametrize("active,passive", [
    ("The men carry a box.", "A box is carried by the men."),
    ("The men are carrying a box.", "A box is being carried by the men."),
    ("The men have carried a box.", "A box has been carried by the men."),
    ("The men could carry a box.", "A box could be carried by the men."),
    ("The men do not carry a box.", "A box is not carried by the men."),
    ("John saw Mary.", "Mary was seen by John."),
    ("She helps him.", "He is helped by her."),
    ("A young girl is using sign language.", "Sign language is being used by a young girl."),
    ("The crowd donates money to the performers.", "Money is donated by the crowd to the performers."),
    ("The manager rejected the proposal by the river.", "The proposal was rejected by the manager by the river."),
])
def test_voice_retains_roles_tense_aspect_and_modality_in_both_directions(active, passive):
    assert voice_preserves_form(active, passive)
    assert voice_preserves_form(passive, active)


@pytest.mark.parametrize("active,passive", [
    ("John saw Mary.", "John was seen by Mary."),
    ("John sees Mary.", "Mary was seen by John."),
    ("John could help Mary.", "Mary is helped by John."),
    ("John is helping Mary.", "Mary has been helped by John."),
    ("John always helps Mary.", "Mary is helped by John."),
    ("A man skateboards.", "Skateboarding is done by a man."),
    ("A man carries a box.", "A box is transported by a man."),
    ("At the very minimum, you could help a child repent from their past sins.",
     "A child could be helped by you to repent from their past sins."),
    ("The men carry two boxes.", "Two boxes are carried by the man."),
])
def test_voice_rejects_role_flip_tense_loss_nominalization_and_lexical_loss(active, passive):
    assert not voice_preserves_form(active, passive)
    assert not voice_preserves_form(passive, active)


def test_synonym_cannot_break_a_fixed_aspect_or_qualification_span():
    text = "A child is about to hit the panda with a stick."
    assert realize_synonym_plan(text, {"edits": [
        {"word_index": 3, "old_word": "about", "replacement": "nearly"}]}) is None
    assert realize_synonym_plan(text, {"edits": [
        {"word_index": 10, "old_word": "stick", "replacement": "rod"}]}) == text.replace("stick", "rod")


def test_judge_sees_final_realized_candidate_and_raw_output_is_retained(monkeypatch):
    raw = json.dumps([{"target_label": "neutral", "edit_plan": {
        "added_clause": "She owns a blue coat.", "connector": " and "}}])
    monkeypatch.setattr(core, "_call", lambda *args: raw)
    seen = []
    def judge(client, source, operation, candidates, trace):
        seen.extend(candidates)
        return {}
    monkeypatch.setattr(core, "judge_current", judge)
    _, trace = core.generate_atomic(object(), SOURCE, "uninformative")
    expected = "A man carries a box and she owns a blue coat."
    assert seen == [expected, expected]
    for attempt in trace["attempts"]:
        assert attempt["raw_generation"] == raw
        assert attempt["candidates"] == [expected]
        change = attempt["candidate_realization"]["changes"][0]
        assert change["before"].endswith("and She owns a blue coat.")
        assert change["after"] == expected


def test_changed_realization_cannot_reuse_old_cached_judgment():
    raw = json.dumps([{"target_label": "neutral", "edit_plan": {
        "added_clause": "She owns a blue coat.", "connector": " and "}}])
    old = "A man carries a box and She owns a blue coat."
    verdict = {"index": 0, "label": "neutral", "source_label": "entailment",
               "grammatical": True, "operation_valid": True, "source_equivalent": False}
    trace = {"status": "accepted", "selected_hypothesis": old, "attempts": [{
        "raw_generation": raw, "candidates": [old], "raw_judgment": json.dumps([verdict])}]}
    with pytest.raises(ValueError, match="saved candidates differ"):
        replay_atomic(SOURCE, "uninformative", trace, core.parse_candidates, old)
    assert trace["attempts"][0]["candidates"] == [old]
