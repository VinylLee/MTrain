"""Tests for the component_mrs backfill tool, driven by synthetic fixtures."""

import ast
import json
from pathlib import Path

import backfill_component_mrs as bf
import pytest

FIELD = bf.COMPONENT_MRS_FIELD


def generator_constants():
    """Read the real generator's contract so fixtures stay in sync with it."""
    tree = ast.parse(Path(bf.DEFAULT_GENERATOR_SOURCE).read_text(encoding="utf-8"))
    wanted = {"PROMPT_VERSION", "OPERATION_SPECS", "COMPOSITE_STAGE_PLANS"}
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    found[target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in wanted:
                found[node.target.id] = ast.literal_eval(node.value)
    return found


def source_row(pair_id=1):
    return {"premise": "P1", "hypothesis": "H1", "label": 0, "pair_id": pair_id,
            "is_syn": False, "weight": 1.0, "mr_id": "none", "mr_type": "inv"}


def atomic_row(pair_id=1):
    return {"premise": "P1", "hypothesis": "H2", "label": 2, "pair_id": pair_id,
            "is_syn": True, "weight": 1.0, "mr_id": "negation_flip", "mr_type": "flip"}


def composite_row(pair_id=1, mr_id="composite_flip"):
    return {"premise": "P1", "hypothesis": "H3", "label": 2, "pair_id": pair_id,
            "is_syn": True, "weight": 1.0, "mr_id": mr_id,
            "mr_type": "flip" if mr_id != "composite_neutral" else "neutral"}


COMPOSITE_STEPS = ["synonym_replacement", "negation_flip"]


def build(tmp_path, rows, log_entries, *, report_overrides=None):
    constants = generator_constants()
    data = tmp_path / "artifact.json"
    data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")

    log = tmp_path / "artifact.generation_log.jsonl"
    log.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in log_entries),
                   encoding="utf-8")

    operation_results = {}
    for operation in constants["OPERATION_SPECS"]:
        attempted = sum(1 for e in log_entries if e.get("operation") == operation)
        accepted = sum(1 for e in log_entries
                       if e.get("operation") == operation and e.get("status") == "accepted")
        operation_results[operation] = {"attempted": attempted, "accepted": accepted,
                                        "acceptance_rate": None}

    report = {
        "prompt_version": constants["PROMPT_VERSION"],
        "label_mrs": list(constants["OPERATION_SPECS"]),
        "operation_results": operation_results,
        "source_count": sum(1 for r in rows if not r.get("is_syn")),
        "synthetic_count": sum(1 for r in rows if r.get("is_syn")),
        "output": str(data),
    }
    report.update(report_overrides or {})
    report_path = tmp_path / "artifact.report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return data, log, report_path


def run_backfill(data, log, report_path, tmp_path, *extra):
    args = bf.parse_args([
        "--data", str(data), "--generation-log", str(log), "--report", str(report_path),
        "--backup", str(tmp_path / "backup.json"),
        "--backfill-report", str(tmp_path / "backfill.json"),
        *extra,
    ])
    return bf.run(args)


def good_log(pair_id=1):
    return [
        {"operation": "negation_flip", "status": "accepted", "pair_id": pair_id,
         "source_label": 0, "target_label": "contradiction"},
        {"operation": "composite_flip", "status": "accepted", "pair_id": pair_id,
         "source_label": 0, "target_label": "contradiction", "applied_steps": COMPOSITE_STEPS},
    ]


def test_happy_path_enriches_only_the_composite_row(tmp_path):
    data, log, report = build(
        tmp_path, [source_row(), atomic_row(), composite_row()], good_log())
    before = data.read_bytes()
    result = run_backfill(data, log, report, tmp_path)

    assert result["status"] == "applied"
    assert result["data"]["changed_lines"] == 1
    assert result["data"]["unchanged_lines"] == 2
    assert result["join"]["bijection"] is True

    lines = data.read_text(encoding="utf-8").splitlines()
    assert FIELD not in json.loads(lines[0])   # source row untouched
    assert FIELD not in json.loads(lines[1])   # atomic row untouched
    assert json.loads(lines[2])[FIELD] == COMPOSITE_STEPS
    assert tuple(json.loads(lines[2])) == bf.COMPOSITE_ROW_KEY_ORDER
    # Untouched lines keep their original bytes exactly.
    assert lines[0].encode() == before.split(b"\n")[0]


def test_backup_is_written_and_matches_the_original(tmp_path):
    data, log, report = build(tmp_path, [source_row(), composite_row()], good_log())
    original = bf.sha256_file(data)
    result = run_backfill(data, log, report, tmp_path)
    backup = Path(result["backup"]["path"])
    assert result["backup"]["created"] is True
    assert bf.sha256_file(backup) == original


def test_second_run_is_a_byte_identical_no_op(tmp_path):
    data, log, report = build(tmp_path, [source_row(), composite_row()], good_log())
    run_backfill(data, log, report, tmp_path)
    after_first = data.read_bytes()

    result = run_backfill(data, log, report, tmp_path)
    assert result["status"] == "already_applied"
    assert result["data"]["changed_lines"] == 0
    assert data.read_bytes() == after_first


def test_check_mode_writes_nothing(tmp_path):
    data, log, report = build(tmp_path, [source_row(), composite_row()], good_log())
    original = data.read_bytes()
    result = run_backfill(data, log, report, tmp_path, "--check")
    assert result["status"] == "check_only"
    assert data.read_bytes() == original
    assert not (tmp_path / "backup.json").exists()
    assert not (tmp_path / "backfill.json").exists()


def test_aborts_when_a_composite_log_entry_lacks_applied_steps(tmp_path):
    log_entries = good_log()
    del log_entries[1]["applied_steps"]
    data, log, report = build(tmp_path, [source_row(), composite_row()], log_entries)
    original = data.read_bytes()
    with pytest.raises(bf.BackfillError):
        run_backfill(data, log, report, tmp_path)
    assert data.read_bytes() == original


def test_aborts_when_a_log_entry_is_never_consumed(tmp_path):
    log_entries = good_log() + [
        {"operation": "composite_flip", "status": "accepted", "pair_id": 99,
         "source_label": 0, "target_label": "contradiction", "applied_steps": COMPOSITE_STEPS}
    ]
    data, log, report = build(tmp_path, [source_row(), composite_row()], log_entries)
    with pytest.raises(bf.BackfillError):
        run_backfill(data, log, report, tmp_path)


def test_aborts_when_the_composite_row_has_no_log_entry(tmp_path):
    data, log, report = build(
        tmp_path, [source_row(), composite_row(pair_id=5)], good_log(pair_id=1))
    with pytest.raises(bf.BackfillError):
        run_backfill(data, log, report, tmp_path)


def test_aborts_on_a_duplicate_log_key(tmp_path):
    log_entries = good_log() + [dict(good_log()[1])]
    data, log, report = build(tmp_path, [source_row(), composite_row()], log_entries)
    with pytest.raises(bf.BackfillError):
        run_backfill(data, log, report, tmp_path)


def test_aborts_when_a_non_composite_row_carries_the_field(tmp_path):
    row = atomic_row()
    row[FIELD] = COMPOSITE_STEPS
    data, log, report = build(tmp_path, [source_row(), row], good_log())
    with pytest.raises(bf.BackfillError):
        run_backfill(data, log, report, tmp_path)


def test_aborts_when_an_existing_value_disagrees_and_force_rewrites_it(tmp_path):
    row = composite_row()
    row[FIELD] = ["negation_flip", "synonym_replacement"]  # reversed: disagrees with the log
    data, log, report = build(tmp_path, [source_row(), row], good_log())

    with pytest.raises(bf.BackfillError):
        run_backfill(data, log, report, tmp_path)

    result = run_backfill(data, log, report, tmp_path, "--force")
    assert result["status"] == "applied"
    assert json.loads(data.read_text(encoding="utf-8").splitlines()[1])[FIELD] == COMPOSITE_STEPS


def test_aborts_when_the_report_disagrees_with_the_log(tmp_path):
    data, log, report = build(
        tmp_path, [source_row(), composite_row()], good_log(),
        report_overrides={"operation_results": {"negation_flip":
                                                {"attempted": 99, "accepted": 99}}},
    )
    with pytest.raises(bf.BackfillError):
        run_backfill(data, log, report, tmp_path)


def test_aborts_when_prompt_version_does_not_match_the_generator(tmp_path):
    data, log, report = build(
        tmp_path, [source_row(), composite_row()], good_log(),
        report_overrides={"prompt_version": "some-other-version"})
    with pytest.raises(bf.BackfillError):
        run_backfill(data, log, report, tmp_path)


def test_report_records_the_join_and_the_component_distribution(tmp_path):
    data, log, report = build(tmp_path, [source_row(), composite_row()], good_log())
    result = run_backfill(data, log, report, tmp_path)
    assert result["component_distribution"] == {"synonym_replacement+negation_flip": 1}
    assert result["generator_report"]["config_fingerprint_source"] == \
        "ast_snapshot_of_generator_source"
    assert result["legacy_run_command_recovered"]["confidence"] == "unofficial_recovery"
    assert all(result["validation"].values())
