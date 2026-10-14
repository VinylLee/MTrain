"""Constrained decoding must bind indices to actual text without approving edits."""
import pytest

jsonschema=pytest.importorskip("jsonschema")
from nli_v3_4.schemas import invariant_generation_schema


def test_schema_binds_each_index_to_its_word_and_permits_abstention():
    source={"hypothesis":"The children buy books.","premise":"secret","label":0,"pair_id":2}
    schema=invariant_generation_schema(source,"synonym_replacement")["json_schema"]["schema"]
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate([],schema)
    correct=[{"label_rule":"inherit_source","edit_plan":{"edits":[{"word_index":2,"old_word":"buy","replacement":"purchase"}]}}]
    jsonschema.validate(correct,schema)
    correct[0]["edit_plan"]["edits"][0]["word_index"]=1
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(correct,schema)
    changed={**source,"premise":"different","label":2,"pair_id":900}
    assert invariant_generation_schema(source,"synonym_replacement")==invariant_generation_schema(changed,"synonym_replacement")


def test_schema_does_not_filter_semantics_or_relabel_noninvariant_data():
    source={"hypothesis":"He walks."}
    schema=invariant_generation_schema(source,"synonym_replacement")["json_schema"]["schema"]
    # A broader replacement can satisfy syntax; existing semantic checks must
    # still reject it. The schema provides no quality score or acceptance oracle.
    jsonschema.validate([{"label_rule":"inherit_source","edit_plan":{"edits":[{"word_index":1,"old_word":"walks","replacement":"moves"}]}}],schema)
    with pytest.raises(ValueError):invariant_generation_schema(source,"negation_flip")
