"""Tests for the shared provenance helpers."""

import provenance_schema as ps
import pytest

PROMPT_VERSION = "snli-pair-aware-all-label-mrs-v3.3"
OPERATION_SPECS = {
    "negation_flip": {"mr_type": "flip", "instruction": "flip it"},
    "synonym_replacement": {"mr_type": "inv", "instruction": "swap words"},
    "composite_flip": {"mr_type": "flip", "instruction": "two edits"},
}
COMPOSITE_STAGE_PLANS = {
    "composite_flip": (("synonym_replacement", "negation_flip"),),
}


def fingerprint():
    return ps.config_fingerprint(PROMPT_VERSION, OPERATION_SPECS, COMPOSITE_STAGE_PLANS)


def log_for(operation, steps=None, status="accepted"):
    entry = {"operation": operation, "status": status}
    if steps is not None:
        entry["applied_steps"] = steps
    return entry


def test_fingerprint_is_stable():
    assert fingerprint() == fingerprint()


def test_fingerprint_changes_when_an_operation_spec_changes():
    mutated = {**OPERATION_SPECS,
               "negation_flip": {"mr_type": "flip", "instruction": "flip it differently"}}
    assert ps.config_fingerprint(PROMPT_VERSION, mutated, COMPOSITE_STAGE_PLANS) != fingerprint()


def test_fingerprint_changes_when_the_prompt_version_changes():
    assert ps.config_fingerprint(PROMPT_VERSION + "x", OPERATION_SPECS,
                                 COMPOSITE_STAGE_PLANS) != fingerprint()


def test_fingerprint_is_independent_of_the_schema_version_constant(monkeypatch):
    """The legacy artifact predates the constant, so it must not participate in the digest.

    If this ever fails, fingerprints can no longer be reconstructed for data generated
    before the constant existed.
    """
    monkeypatch.setattr(ps, "PROVENANCE_SCHEMA_VERSION", 999)
    assert fingerprint() == fingerprint()


def test_fingerprint_treats_tuples_and_lists_alike():
    """In-memory plans are tuples; the backfill recovers lists via ast.literal_eval."""
    as_lists = {"composite_flip": [["synonym_replacement", "negation_flip"]]}
    assert ps.config_fingerprint(PROMPT_VERSION, OPERATION_SPECS, as_lists) == fingerprint()


def test_component_mrs_returns_none_for_atomic_rows():
    assert ps.component_mrs_from_log(
        "negation_flip", log_for("negation_flip"),
        composite_operations=COMPOSITE_STAGE_PLANS, operation_specs=OPERATION_SPECS,
    ) is None


def test_component_mrs_returns_the_plan_for_composite_rows():
    steps = ["synonym_replacement", "negation_flip"]
    assert ps.component_mrs_from_log(
        "composite_flip", log_for("composite_flip", steps),
        composite_operations=COMPOSITE_STAGE_PLANS, operation_specs=OPERATION_SPECS,
    ) == steps


@pytest.mark.parametrize(
    "operation, entry, reason",
    [
        ("composite_flip", log_for("composite_flip"), "missing applied_steps"),
        ("composite_flip", log_for("composite_flip", []), "empty applied_steps"),
        ("composite_flip", log_for("composite_flip", ["synonym_replacement"]), "wrong arity"),
        ("composite_flip", log_for("composite_flip", ["synonym_replacement", "nope"]), "unknown id"),
        ("composite_flip",
         log_for("composite_flip", ["synonym_replacement", "composite_flip"]), "nested composite"),
        ("negation_flip",
         log_for("negation_flip", ["synonym_replacement", "negation_flip"]), "atomic with steps"),
    ],
)
def test_component_mrs_fails_loud(operation, entry, reason):
    with pytest.raises(ValueError):
        ps.component_mrs_from_log(
            operation, entry,
            composite_operations=COMPOSITE_STAGE_PLANS, operation_specs=OPERATION_SPECS,
        )
