"""Tests for the read-only MR-test v3.3 health checker.

Everything under test is a pure function. No test reads the real /proc, the real
generation logs, or the production state file.
"""

from __future__ import annotations

import inspect
import json

import pytest

import check_mr_test_v3_3_claude as checker


# --------------------------------------------------------------------------- #
# Progress-line parsing
# --------------------------------------------------------------------------- #


def test_parse_progress_pair_reads_the_pair_id() -> None:
    line = "[mnlim 1145/9815] pair_id=1144 eligible=3 pending=3"
    assert checker.parse_progress_pair(line, "mnlim") == 1144


def test_parse_progress_pair_isolates_datasets_sharing_a_prefix() -> None:
    """`mnlim` is a prefix of `mnlimm`; the parser must not confuse them."""
    line = "[mnlimm 229/9832] pair_id=228 eligible=11 pending=11"
    assert checker.parse_progress_pair(line, "mnlim") is None
    assert checker.parse_progress_pair(line, "mnlimm") == 228


@pytest.mark.parametrize(
    "line",
    [
        "",
        "[mnlim 1/9815] eligible=3 pending=3",
        "[mnlim 1/9815] pair_id=notanumber eligible=3",
        "RuntimeError: model request failed for dataset=mnlim pair_id=1",
        "[mnlim 1/9815] pair_id= pending=0",
    ],
)
def test_parse_progress_pair_rejects_malformed_lines(line: str) -> None:
    assert checker.parse_progress_pair(line, "mnlim") is None


# --------------------------------------------------------------------------- #
# Incremental log scanning
# --------------------------------------------------------------------------- #


def test_scan_growth_counts_only_new_lines(tmp_path) -> None:
    log = tmp_path / "mnlim.stdout.log"
    log.write_text(
        "[mnlim 1/9815] pair_id=0 eligible=3 pending=3\n"
        "[mnlim 2/9815] pair_id=1 eligible=3 pending=3\n",
        encoding="utf-8",
    )
    carry, offset = checker.scan_growth(log, 0, {}, "mnlim")
    assert carry["latest_pair"] == 1
    assert carry["progress_lines"] == 2

    with log.open("a", encoding="utf-8") as handle:
        handle.write("[mnlim 3/9815] pair_id=2 eligible=3 pending=3\n")
    carry, offset = checker.scan_growth(log, offset, carry, "mnlim")
    assert carry["latest_pair"] == 2
    assert carry["progress_lines"] == 3


def test_scan_growth_rereads_a_partially_written_line_exactly_once(tmp_path) -> None:
    """A line caught mid-write must not be dropped or double-counted."""
    log = tmp_path / "mnlim.stdout.log"
    full = "[mnlim 7/9815] pair_id=6 eligible=3 pending=3\n"
    partial = full[:20]
    log.write_text(partial, encoding="utf-8")

    carry, offset = checker.scan_growth(log, 0, {}, "mnlim")
    assert carry.get("latest_pair") is None
    assert offset == 0, "offset must stay at the start of the incomplete line"

    log.write_text(full, encoding="utf-8")
    carry, offset = checker.scan_growth(log, offset, carry, "mnlim")
    assert carry["latest_pair"] == 6
    assert carry["progress_lines"] == 1

    carry, offset = checker.scan_growth(log, offset, carry, "mnlim")
    assert carry["progress_lines"] == 1, "a re-scan of the same bytes must add nothing"


def test_a_line_split_across_two_runs_is_counted_as_progress(tmp_path) -> None:
    """The exact boundary the live lanes hit: a run ends mid-line, the next run
    sees it completed. It must register as one new pair, not as no progress."""
    log = tmp_path / "mnlim.stdout.log"
    prefix = "".join(
        f"[mnlim {i + 1}/9815] pair_id={i} eligible=3 pending=3\n" for i in range(1146)
    )
    half_written = "[mnlim 1147/9815] pair_id=1146 eligible=3 pend"
    log.write_text(prefix + half_written, encoding="utf-8")

    carry, offset = checker.scan_growth(log, 0, {}, "mnlim")
    assert carry["latest_pair"] == 1145
    stored = {"latest_pair": carry["latest_pair"], "offset": offset,
              "streak": 0, "change_epoch": 1_000_000, "pid": "p"}

    with log.open("a", encoding="utf-8") as handle:
        handle.write("ing=3\n")
    carry, _ = checker.scan_growth(log, offset, stored, "mnlim")
    assert carry["latest_pair"] == 1146

    stall = checker.update_stall(
        stored, carry["latest_pair"], "p", True, 1_000_900, 64, 96
    )
    assert stall["streak"] == 0
    assert stall["stall_state"] == "progress"


def test_scan_growth_detects_truncation(tmp_path) -> None:
    log = tmp_path / "mnlim.stdout.log"
    log.write_text(
        "[mnlim 1/9815] pair_id=0 eligible=3 pending=3\n"
        "RuntimeError: model request failed for dataset=mnlim pair_id=1: boom\n",
        encoding="utf-8",
    )
    carry, offset = checker.scan_growth(log, 0, {}, "mnlim")
    assert carry["error_count"] == 1
    assert "boom" in carry["last_error"]

    # Lane restarted with a fresh log -> counter restarts from zero.
    log.write_text("[mnlim 1/9815] pair_id=0 eligible=3 pending=0\n", encoding="utf-8")
    carry, _ = checker.scan_growth(log, offset, carry, "mnlim")
    assert carry["error_count"] == 0


# --------------------------------------------------------------------------- #
# Stall detection -- the core of the change
# --------------------------------------------------------------------------- #


def test_update_stall_does_not_accept_delta_pairs() -> None:
    """delta_pairs is flushed only every --checkpoint-every sources and reads 0
    for whole cycles while the lane is progressing. It must never be an input."""
    parameters = inspect.signature(checker.update_stall).parameters
    assert not [name for name in parameters if "delta" in name or "checkpoint" in name]


def test_checkpoint_stall_reads_as_progress_not_as_a_stall() -> None:
    """The exact observed sequence: three cycles at delta_pairs=0 while
    latest_pair advanced 1135 -> 1137 -> 1144. None of these is a stall."""
    warn, alert = checker.stall_thresholds(checker.stall_budget_minutes({}, "mnlim"))
    assert (warn, alert) == (64, 96)

    state: dict[str, object] = {}
    observed = []
    for cycle, pair in enumerate([1135, 1137, 1144], start=1):
        state = checker.update_stall(
            state, pair, "1903510", True, 1_000_000 + cycle * 900, warn, alert
        )
        observed.append(state["stall_state"])

    assert observed == ["seeded", "progress", "progress"]
    assert all(checker.stall_level(s, "mnlim") == checker.LEVEL_OK for s in observed)


def test_update_stall_flags_a_genuine_stall() -> None:
    warn, alert = 64, 96
    state = checker.update_stall({}, 1144, "1903510", True, 1_000_000, warn, alert)
    state = checker.update_stall(state, 1144, "1903510", True, 1_000_300, warn, alert)
    assert state["stall_state"] == checker.LEVEL_OK

    state = checker.update_stall(
        state, 1144, "1903510", True, 1_000_000 + 64 * 60, warn, alert
    )
    assert state["stall_state"] == checker.LEVEL_WARN

    state = checker.update_stall(
        state, 1144, "1903510", True, 1_000_000 + 96 * 60, warn, alert
    )
    assert state["stall_state"] == checker.LEVEL_ALERT


def test_sick_uses_a_larger_budget_than_mnlim_for_the_same_elapsed_time() -> None:
    """sick runs --workers 4 --request-timeout 600, mnlim 3 / 120."""
    mnlim_warn, _ = checker.stall_thresholds(
        checker.stall_budget_minutes({}, "mnlim")
    )
    sick_warn, _ = checker.stall_thresholds(checker.stall_budget_minutes({}, "sick"))
    assert mnlim_warn == 64
    assert sick_warn == 240

    now = 1_000_000
    elapsed = now + 100 * 60  # 100 minutes with no movement
    mnlim = checker.update_stall({}, 10, "p", True, now, mnlim_warn, 96)
    mnlim = checker.update_stall(mnlim, 10, "p", True, elapsed, mnlim_warn, 96)
    sick = checker.update_stall({}, 10, "q", True, now, sick_warn, 360)
    sick = checker.update_stall(sick, 10, "q", True, elapsed, sick_warn, 360)

    assert mnlim["stall_state"] == checker.LEVEL_ALERT
    assert sick["stall_state"] == checker.LEVEL_OK


def test_update_stall_treats_a_rewind_as_a_rescan_not_a_stall() -> None:
    """On resume the generator re-prints every already-completed source, so
    latest_pair falls back to 0. A `>` comparison would keep a stale streak."""
    state = checker.update_stall({}, 1144, "p", True, 1_000_000, 64, 96)
    state = checker.update_stall(state, 0, "p", True, 1_000_900, 64, 96)
    assert state["stall_state"] == "rescan"
    assert state["streak"] == 0

    state = checker.update_stall(state, 1145, "p", True, 1_001_800, 64, 96)
    assert state["stall_state"] == "progress"


def test_update_stall_reseeds_when_the_lane_is_relaunched() -> None:
    state = checker.update_stall({}, 1144, "111", True, 1_000_000, 64, 96)
    state = checker.update_stall(
        state, 1144, "111", True, 1_000_000 + 90 * 60, 64, 96
    )
    assert state["stall_state"] == checker.LEVEL_WARN

    state = checker.update_stall(state, 1144, "222", True, 1_000_000 + 91 * 60, 64, 96)
    assert state["streak"] == 0
    assert state["stall_state"] == "seeded"


def test_a_dead_lane_is_down_and_never_a_stall() -> None:
    state = checker.update_stall({}, 1144, "p", True, 1_000_000, 64, 96)
    state = checker.update_stall(state, 1144, None, False, 1_000_000 + 600 * 60, 64, 96)
    assert state["stall_state"] == "down"


def test_stall_level_only_alerts_on_down_for_in_scope_lanes() -> None:
    assert checker.stall_level("down", "mnlim") == checker.LEVEL_ALERT
    assert checker.stall_level("down", "sick") == checker.LEVEL_ALERT
    # Report-only lanes must not raise an alert just for being stopped.
    assert checker.stall_level("down", "mnlimm") == checker.LEVEL_OK
    assert checker.stall_level("down", "snli") == checker.LEVEL_OK


def test_stall_level_does_not_alert_on_a_lane_that_finished() -> None:
    """A finished lane is ``down`` by design: the generator exits and writes
    ``<dataset>.report.json`` with status=complete. That is completion, not a
    stall, and must not be reported as one."""
    assert checker.stall_level("down", "sick", "complete") == checker.LEVEL_OK
    assert checker.stall_level("down", "mnlim", "complete") == checker.LEVEL_OK


def test_stall_level_still_alerts_on_a_down_lane_that_did_not_finish() -> None:
    """Only a *completion* report suppresses the down-alert. A lane that died
    without writing one is still the failure this alert exists to catch."""
    assert checker.stall_level("down", "sick", "none") == checker.LEVEL_ALERT
    assert checker.stall_level("down", "mnlim", "unreadable") == checker.LEVEL_ALERT


def test_stall_level_keeps_explicit_stall_states_regardless_of_report() -> None:
    """An explicit stall verdict outranks the completion report."""
    assert (
        checker.stall_level(checker.LEVEL_ALERT, "sick", "complete")
        == checker.LEVEL_ALERT
    )
    assert (
        checker.stall_level(checker.LEVEL_WARN, "sick", "complete") == checker.LEVEL_WARN
    )


# --------------------------------------------------------------------------- #
# Process identification
# --------------------------------------------------------------------------- #


def test_is_dataset_generator_matches_the_dataset_value_exactly() -> None:
    argv = ["/home/ubuntu/anaconda3/bin/python", "generate_mr_test_v3_3.py",
            "--dataset", "mnlimm", "--resume"]
    assert checker.is_dataset_generator(argv, "mnlimm") is True
    assert checker.is_dataset_generator(argv, "mnlim") is False


def test_is_dataset_generator_rejects_other_processes() -> None:
    assert checker.is_dataset_generator([], "mnlim") is False
    assert checker.is_dataset_generator(None, "mnlim") is False
    assert checker.is_dataset_generator(
        ["python", "verify_mr_test_checkpoint.py", "--dataset", "mnlim"], "mnlim"
    ) is False


def test_parse_argv_dataset_handles_both_spellings() -> None:
    assert checker.parse_argv_dataset(["--dataset", "sick"]) == "sick"
    assert checker.parse_argv_dataset(["--dataset=sick"]) == "sick"
    assert checker.parse_argv_dataset(["--dataset"]) is None
    assert checker.parse_argv_dataset([]) is None


def test_parse_lane_args_extracts_the_stall_budget_inputs() -> None:
    argv = ["python", "g.py", "--dataset", "sick", "--workers", "4",
            "--request-timeout", "600", "--checkpoint-every=10"]
    parsed = checker.parse_lane_args(argv)
    assert parsed == {"workers": 4, "request_timeout": 600.0, "checkpoint_every": 10}


def test_process_start_epoch_degrades_to_none() -> None:
    # /proc/<pid>/stat field 2 is parenthesised and may contain spaces.
    # An unreadable pid must degrade to None rather than raise.
    assert checker.process_start_epoch("999999999", 1000.0, 100) is None
    # Without a boot epoch there is no absolute reference; do not guess.
    assert checker.process_start_epoch("1", None, 100) is None


def test_read_boot_epoch_returns_a_plausible_unix_time() -> None:
    btime = checker.read_boot_epoch()
    assert btime is not None
    # Later than 2020 and not in the future.
    assert 1_577_836_800 < btime < 4_102_444_800


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "free,temp,expected",
    [
        (178, 64, checker.LEVEL_WARN),   # the live GPU0 baseline
        (4098, 63, checker.LEVEL_OK),
        (4124, 67, checker.LEVEL_OK),
        (1916, 65, checker.LEVEL_OK),
        (256, 60, checker.LEVEL_WARN),
        (255, 60, checker.LEVEL_WARN),
        (96, 60, checker.LEVEL_ALERT),
        (95, 60, checker.LEVEL_ALERT),
        (4098, 78, checker.LEVEL_WARN),
        (4098, 77, checker.LEVEL_OK),
        (4098, 85, checker.LEVEL_ALERT),
        (4098, 84, checker.LEVEL_WARN),
        (None, None, checker.LEVEL_OK),
    ],
)
def test_classify_gpu_boundaries(free, temp, expected) -> None:
    level, _ = checker.classify_gpu(free, temp)
    assert level == expected


def test_classify_ckpt_lag() -> None:
    assert checker.classify_ckpt_lag(8, 10) == checker.LEVEL_OK
    assert checker.classify_ckpt_lag(10, 10) == checker.LEVEL_OK
    assert checker.classify_ckpt_lag(11, 10) == checker.LEVEL_WARN
    assert checker.classify_ckpt_lag(30, 10) == checker.LEVEL_WARN
    assert checker.classify_ckpt_lag(31, 10) == checker.LEVEL_ALERT
    # Behind the checkpoint (mid-rescan) is not a lag problem.
    assert checker.classify_ckpt_lag(-5, 10) == checker.LEVEL_OK
    assert checker.classify_ckpt_lag(None, 10) == checker.LEVEL_OK


def test_stall_budget_matches_the_generator_worst_case() -> None:
    # ceil(11 ops / 3 workers) * 2 retries * 2 calls * 120s = 32 min
    assert checker.stall_budget_minutes({}, "mnlim") == 32
    # ceil(11 / 4) * 2 * 2 * 600s = 120 min
    assert checker.stall_budget_minutes({}, "sick") == 120
    # Live argv always wins over the dataset default.
    assert checker.stall_budget_minutes(
        {"workers": 4, "request_timeout": 600}, "mnlim"
    ) == 120


def test_stall_thresholds_honour_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("MR33_STALL_WARN_MIN", "5")
    monkeypatch.setenv("MR33_STALL_ALERT_MIN", "9")
    assert checker.stall_thresholds(32) == (5, 9)


def test_verdict_mapping() -> None:
    assert checker.verdict_for([checker.LEVEL_OK]) == ("OK", 0)
    assert checker.verdict_for([checker.LEVEL_OK, checker.LEVEL_WARN]) == ("WARN", 1)
    assert checker.verdict_for([checker.LEVEL_WARN, checker.LEVEL_ALERT]) == ("ALERT", 2)
    assert checker.verdict_for([]) == ("OK", 0)


# --------------------------------------------------------------------------- #
# Report contract (the scheduled task parses this)
# --------------------------------------------------------------------------- #


def _sections() -> dict:
    lane = {
        "dataset": "mnlim", "alive": True, "pid": "1903510", "age": "3h12m",
        "workers": 3, "timeout": 120, "checkpoint_every": 10, "report": "none",
        "budget_minutes": 32, "latest_pair": 1144, "latest_delta": 15,
        "latest_ago_minutes": 0, "streak": 0, "stalled_minutes": 0,
        "stall_state": "progress", "checkpoint_pairs": 1130,
        "checkpoint_pairs_prev": 1130, "ckpt_lag": 15,
        "output_bytes": 3298452, "output_bytes_prev": 3298452,
        "log_bytes": 11902302, "log_bytes_prev": 11902302,
        "error_count": 2, "error_count_prev": 2, "last_error": "none",
        "progress_lines": 3546, "offset": 1,
    }
    lane_report_only = dict(lane, dataset="snli", alive=False, pid=None,
                            stall_state="down")
    return {
        "header": {"ts": "2026-09-21T13:47:12+08:00", "run": 12, "verdict": "OK",
                   "exit_code": 0, "alerts": 0, "warns": 0},
        "lanes": [lane, lane_report_only],
        "models": [("gemma-3-4b-it-qat", "LOADED"),
                   ("gemma-3-4b-it-qat-worker2", "LOADED")],
        "gpus": [{"index": "0", "used": 45225, "total": 46068, "free": 178,
                  "free_delta": 0, "util": 100, "temp": 64,
                  "level": checker.LEVEL_WARN, "reasons": ["free<=256"]}],
        "monitor": {"loop_alive": True, "loop_pid": "1909969", "tmux_alive": True,
                    "lock": "held", "last_cycle": "2026-09-21 13:30:18",
                    "last_cycle_age_minutes": 17, "scheduler": "loop+systemd"},
        "findings": [{"level": checker.LEVEL_WARN, "kind": "gpu_resource",
                      "detail": "index=0 free=178MiB"}],
        "summary": {"verdict": "OK", "lanes_up": 1, "lanes_total": 2,
                    "in_scope": "mnlim,sick", "stalled": 0, "gpu_max_temp": 64,
                    "gpu_min_free": 178, "models": "2/2", "monitor": "UP",
                    "errors_new": 0},
    }


def test_format_report_puts_summary_last() -> None:
    """The scheduled task reads the SUMMARY line with `tail -1`."""
    lines = checker.format_report(_sections()).splitlines()
    assert lines[0].startswith("# MR33CHECK ")
    assert lines[-1].startswith("SUMMARY ")


def test_format_report_marks_report_only_lanes() -> None:
    text = checker.format_report(_sections())
    assert "LANE mnlim scope=recover" in text
    assert "LANE snli scope=report" in text


def test_format_report_renders_the_documented_prefixes() -> None:
    text = checker.format_report(_sections())
    for prefix in ("LANE ", "MODEL ", "GPU ", "MONITOR ", "SUMMARY ", "WARN "):
        assert any(line.startswith(prefix) for line in text.splitlines()), prefix


def test_load_state_tolerates_a_corrupt_file(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    assert checker.load_state(path) == {}
    assert checker.load_state(tmp_path / "absent.json") == {}


def test_save_state_replaces_atomically(tmp_path) -> None:
    path = tmp_path / "state.json"
    checker.save_state(path, {"runs": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"runs": 1}
    checker.save_state(path, {"runs": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"runs": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_recover_and_report_only_scopes_are_disjoint() -> None:
    assert set(checker.RECOVER_DATASETS) == {"mnlim", "sick"}
    assert set(checker.REPORT_ONLY_DATASETS) == {"mnlimm", "snli"}
    assert not set(checker.RECOVER_DATASETS) & set(checker.REPORT_ONLY_DATASETS)
