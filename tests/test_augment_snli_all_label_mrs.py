"""Tests for the generator's row construction.

The generator imports ``openai`` at module scope, which is not installed in every
interpreter. Skip rather than refactoring the import for the tests' convenience — the
import is part of how the script actually runs.
"""

import pytest

pytest.importorskip("openai")

import augment_snli_all_label_mrs as g  # noqa: E402
from provenance_schema import COMPONENT_MRS_FIELD  # noqa: E402

SOURCE = {"premise": "A dog runs.", "hypothesis": "A dog runs fast.", "label": 0, "pair_id": 7}

ATOMIC_KEYS = ("premise", "hypothesis", "label", "pair_id", "is_syn", "weight",
               "mr_id", "mr_type")
COMPOSITE_KEYS = ("premise", "hypothesis", "label", "pair_id", "is_syn", "weight",
                  "mr_id", COMPONENT_MRS_FIELD, "mr_type")


def test_atomic_rows_keep_the_original_key_set_and_order():
    """Artifacts generated before component_mrs existed must stay byte-compatible."""
    row = g.synthetic_record(SOURCE, 7, "negation_flip", "A dog does not run.")
    assert tuple(row) == ATOMIC_KEYS
    assert COMPONENT_MRS_FIELD not in row


def test_composite_rows_place_the_field_after_mr_id():
    row = g.synthetic_record(SOURCE, 7, "composite_flip", "No dog runs.",
                             component_mrs=["pronoun_substitution", "negation_flip"])
    assert tuple(row) == COMPOSITE_KEYS
    assert row[COMPONENT_MRS_FIELD] == ["pronoun_substitution", "negation_flip"]


def test_component_list_is_copied_not_aliased():
    steps = ["pronoun_substitution", "negation_flip"]
    row = g.synthetic_record(SOURCE, 7, "composite_flip", "No dog runs.", component_mrs=steps)
    steps.append("mutated")
    assert row[COMPONENT_MRS_FIELD] == ["pronoun_substitution", "negation_flip"]


def test_source_record_is_unchanged():
    row = g.source_record(SOURCE, 7)
    assert tuple(row) == ATOMIC_KEYS
    assert row["mr_id"] == "none"
    assert row["is_syn"] is False


def test_composite_operations_matches_the_stage_plans():
    assert g.COMPOSITE_OPERATIONS == frozenset(g.COMPOSITE_STAGE_PLANS)


def test_wrapper_raises_for_a_composite_row_without_a_plan():
    with pytest.raises(ValueError):
        g.component_mrs_from_log("composite_flip", {"operation": "composite_flip"})
