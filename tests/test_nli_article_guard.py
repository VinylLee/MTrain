"""Real generation regression plus pronunciation contrasts and source preservation."""
import json
import pytest
from nli_v3_4.article_guard import introduced_article_errors, ArticleCheckedEngine


@pytest.mark.parametrize("old,new", [
    ("A person is holding two objects.", "A individual is holding two objects."),
    ("An adult holds a bag.", "An person holds a bag."),
    ("She chose a university.", "She chose an university."),
    ("He is an honest man.", "He is a honest man."),
])
def test_new_unambiguous_article_errors_are_detected(old, new):
    assert introduced_article_errors(old,new)


@pytest.mark.parametrize("text", [
    "An individual carries an umbrella.", "A useful object is in a house.",
    "An honest person waits for an hour.", "A university offers a one-off course.",
    "An MRI scan is available.", "An herb is on a plate.",
])
def test_correct_or_pronunciation_uncertain_cases_are_not_guessed(text):
    assert not introduced_article_errors("",text)


def test_existing_quoted_error_is_not_silently_repaired_or_newly_flagged():
    old='The notice quotes "a individual", and a man reads it.'
    new='The notice quotes "a individual", and a man reviews it.'
    assert not introduced_article_errors(old,new)
    assert introduced_article_errors(old,new+' It also says "a individual".')


def test_actual_pilot_article_regression_is_rejected_before_any_judgment():
    source={"premise":"A man holds a helicopter and a controller.",
        "hypothesis":"A person is holding two objects.","label":0,"pair_id":9617}
    raw=json.dumps([{"label_rule":"inherit_source","edit_plan":{"edits":[
        {"word_index":1,"old_word":"person","replacement":"individual"}]}}])
    original=dict(source)
    assert ArticleCheckedEngine.parse_proposals(raw,source,'synonym_replacement')==[]
    assert source==original
