"""Unit tests for the v3.3 merged-test schema adapter."""

import pytest

pytest.importorskip("openai")

import generate_mr_test_v3_3 as g  # noqa: E402
from provenance_schema import COMPONENT_MRS_FIELD  # noqa: E402


SOURCE = {
    "premise": "A dog is running.",
    "hypothesis": "An animal is moving.",
    "label": 0,
    "pair_id": 7,
}


def test_source_record_uses_merged_test_schema():
    row = g.source_record(SOURCE)
    assert tuple(row) == (
        "idx", "premise", "hypothesis", "mr_id", "pair_id", "label",
        "mr_type", "is_source",
    )
    assert row["mr_id"] == "none"
    assert row["is_source"] is True


def test_atomic_followup_omits_component_mrs():
    row = g.followup_record(SOURCE, "negation_flip", "No animal is moving.")
    assert row["label"] == 2
    assert row["mr_type"] == "flip"
    assert row["is_source"] is False
    assert COMPONENT_MRS_FIELD not in row


def test_composite_followup_preserves_ordered_components():
    components = ["pronoun_substitution", "negation_flip"]
    row = g.followup_record(
        SOURCE, "composite_flip", "It is not moving.", components
    )
    assert row[COMPONENT_MRS_FIELD] == components
    assert tuple(row).index(COMPONENT_MRS_FIELD) > tuple(row).index("mr_id")


def test_materialize_groups_source_then_operations_and_reindexes():
    accepted = {
        (7, "negation_flip"): g.followup_record(
            SOURCE, "negation_flip", "No animal is moving."
        )
    }
    rows = g.materialize_rows(
        [SOURCE], accepted, ["negation_flip", "synonym_replacement"]
    )
    assert [row["idx"] for row in rows] == [0, 1]
    assert [row["mr_id"] for row in rows] == ["none", "negation_flip"]


def test_recover_state_rebuilds_accepted_row_from_log():
    log = {
        "pair_id": 7,
        "operation": "negation_flip",
        "status": "accepted",
        "selected_hypothesis": "No animal is moving.",
    }
    accepted, completed = g.recover_state(
        [], [log], {7: SOURCE}, ["negation_flip"]
    )
    assert (7, "negation_flip") in completed
    assert accepted[(7, "negation_flip")]["hypothesis"] == "No animal is moving."


def test_recover_state_rejects_duplicate_log_keys():
    log = {"pair_id": 7, "operation": "negation_flip", "status": "rejected"}
    with pytest.raises(ValueError, match="duplicate key"):
        g.recover_state([], [log, dict(log)], {7: SOURCE}, ["negation_flip"])


def test_collect_generation_errors_finds_nested_composite_failures():
    log = {
        "attempts": [
            {"stage_logs": [{"attempts": [{"status": "error", "error": "unloaded"}]}]}
        ]
    }
    assert g.collect_generation_errors(log) == ["unloaded"]


# --- one pathological pair must not discard the rest of the run ---------------


def test_request_error_streak_aborts_only_at_the_threshold():
    assert g.MAX_CONSECUTIVE_REQUEST_ERROR_SOURCES == 5
    assert [g.should_abort_on_request_errors(n) for n in range(0, 5)] == [False] * 5
    assert all(g.should_abort_on_request_errors(n) for n in (5, 6, 100))


def test_request_error_marker_is_not_recollected_as_an_error():
    """The marker must stay inert to collect_generation_errors: if it were
    picked up as an "error", every skipped row would look like a fresh failure."""
    log = {
        "pair_id": 4250,
        "operation": "voice_switch",
        "status": "rejected",
        "request_error": "ValueError: generator response is not a JSON array",
    }
    assert g.collect_generation_errors(log) == []


def test_a_skipped_pair_still_lands_in_the_generation_log():
    """The row is appended even when every operation failed, which is what keeps
    the log's pair_id sequence contiguous and the checkpoint verifier happy."""
    log = {
        "pair_id": 7,
        "operation": "negation_flip",
        "status": "rejected",
        "request_error": "ValueError: generator response is not a JSON array",
    }
    accepted, completed = g.recover_state([], [log], {7: SOURCE}, ["negation_flip"])
    assert (7, "negation_flip") in completed
    assert accepted == {}, "a request-error row must not be accepted as a followup"


def test_a_skipped_pair_is_not_regenerated_on_resume():
    log = {
        "pair_id": 7,
        "operation": "negation_flip",
        "status": "rejected",
        "request_error": "ValueError: generator response is not a JSON array",
    }
    _, completed = g.recover_state([], [log], {7: SOURCE}, ["negation_flip"])
    pending = [op for op in ["negation_flip"] if (7, op) not in completed]
    assert pending == [], "the skipped pair must not be retried after a resume"
