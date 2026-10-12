#!/usr/bin/env python3
"""Read-only health check for the MR-test v3.3 generation lanes.

This is the reporting layer for the Claude Code scheduled task. It observes and
classifies; it never recovers. Recovery belongs solely to the deterministic shell
layer (``monitor_mr_test_v3_3_full.sh`` driven by the tmux loop and the systemd
timer). Two recovery agents running at once is what produces
``RuntimeError: checkpoint changed while snapshot was being copied`` and double
relaunches.

Therefore this script must never:

* call ``verify_mr_test_checkpoint.py --snapshot`` (no writes to the backup root);
* kill, signal, restart, ``tmux``-mutate, or ``lms load`` anything;
* block waiting on ``supervisor.lock`` -- the tmux loop holds it for its whole
  lifetime, so only a non-blocking probe is used.

Its only write is its own state file, replaced atomically.

Exit codes: 0 OK, 1 WARN, 2 ALERT, 3 ERR (the checker itself failed).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/ubuntu/MTrain")
RUN_ROOT = ROOT / "mr_test_runs_v3_3_full"
OUTPUT_ROOT = ROOT / "mr_test_data_v3_3_full"
BACKUP_ROOT = ROOT / "mr_test_checkpoint_backups"
SOURCE_ROOT = ROOT / "mr_test_sources_v3_3"
VERIFY_SCRIPT = ROOT / "verify_mr_test_checkpoint.py"
STATE_PATH = RUN_ROOT / "claude_check_state.json"
PYTHON_BIN = "/home/ubuntu/anaconda3/bin/python"
LMS_FALLBACK = "/home/ubuntu/.lmstudio/bin/lms"
TMUX_SESSION = "MTrain-MR-v33-monitor"

# Auto-recovery scope. mnlimm and snli are report-only: they were stopped on
# 2026-09-19 and must never be auto-restarted by this layer or recommended for
# restart in the report.
RECOVER_DATASETS = ("mnlim", "sick")
REPORT_ONLY_DATASETS = ("mnlimm", "snli")
DATASETS = RECOVER_DATASETS + REPORT_ONLY_DATASETS

MODEL_IDENTIFIERS = ("gemma-3-4b-it-qat", "gemma-3-4b-it-qat-worker2")

# Per-dataset launch parameters, as used by the queue scripts. These seed the
# stall budget when the lane is down and its argv is unavailable.
LANE_DEFAULTS = {
    "mnlim": {"workers": 3, "request_timeout": 120, "checkpoint_every": 10},
    "mnlimm": {"workers": 3, "request_timeout": 120, "checkpoint_every": 10},
    "sick": {"workers": 4, "request_timeout": 600, "checkpoint_every": 10},
    "snli": {"workers": 2, "request_timeout": 600, "checkpoint_every": 10},
}

# Operations per composite/atomic MR in augment_snli_all_label_mrs.py, and the
# generator's --retries default. Worst case for one source pair is
# ceil(ops / workers) rounds * retries * 2 calls (generator + judge) * timeout.
OPERATION_COUNT = 11
RETRY_ATTEMPTS = 2

GPU_TEMP_WARN_C = 78
GPU_TEMP_ALERT_C = 85
GPU_FREE_WARN_MIB = 256
GPU_FREE_ALERT_MIB = 96

MONITOR_CYCLE_STALE_MIN = 45
CKPT_LAG_ALERT_MULTIPLE = 3

LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_ALERT = "alert"

LEVEL_RANK = {LEVEL_OK: 0, LEVEL_WARN: 1, LEVEL_ALERT: 2}


# --------------------------------------------------------------------------- #
# Pure logic (no I/O -- imported directly by the tests)
# --------------------------------------------------------------------------- #


def parse_progress_pair(line: str, dataset: str) -> int | None:
    """Extract the pair id from a generator progress line.

    Line shape: ``[mnlim 1145/9815] pair_id=1144 eligible=3 pending=3``.
    The dataset prefix is matched as a whole token plus a space, so ``mnlim``
    never matches an ``mnlimm`` line (the two differ only by a suffix, and a
    naive substring test on ``--dataset mnlim`` matches ``--dataset mnlimm``).
    """
    prefix = f"[{dataset} "
    if not line.startswith(prefix) or "] pair_id=" not in line:
        return None
    remainder = line.split("] pair_id=", 1)[1]
    token = remainder.split(None, 1)[0] if remainder.strip() else ""
    try:
        return int(token)
    except ValueError:
        return None


def parse_argv_dataset(argv: list[str]) -> str | None:
    """Return the value of ``--dataset`` from an argv list, matched exactly."""
    for index, token in enumerate(argv):
        if token == "--dataset" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--dataset="):
            return token.split("=", 1)[1]
    return None


def is_dataset_generator(argv: list[str] | None, dataset: str) -> bool:
    """True when argv is the generator process for exactly this dataset.

    ``--dataset mnlim`` is a prefix of ``--dataset mnlimm``, so the comparison
    must be on the parsed value, never on a substring of the joined command line.
    """
    if not argv:
        return False
    if not any("generate_mr_test_v3_3.py" in token for token in argv):
        return False
    return parse_argv_dataset(argv) == dataset


def parse_lane_args(argv: list[str] | None) -> dict[str, object]:
    """Pull --workers / --request-timeout / --checkpoint-every out of argv."""
    result: dict[str, object] = {}
    if not argv:
        return result
    mapping = {
        "--workers": ("workers", int),
        "--request-timeout": ("request_timeout", float),
        "--checkpoint-every": ("checkpoint_every", int),
    }
    for index, token in enumerate(argv):
        name = token.split("=", 1)[0]
        if name not in mapping:
            continue
        key, cast = mapping[name]
        if "=" in token:
            raw = token.split("=", 1)[1]
        elif index + 1 < len(argv):
            raw = argv[index + 1]
        else:
            continue
        try:
            result[key] = cast(raw)
        except (TypeError, ValueError):
            continue
    return result


def stall_budget_minutes(lane_args: dict[str, object], dataset: str | None = None) -> int:
    """Worst-case wall time for one source pair, in minutes.

    The generator submits every pending operation for a source to a
    ThreadPoolExecutor(workers); the slowest operation dominates the source.
    """
    defaults = LANE_DEFAULTS.get(dataset or "", {})
    workers = int(lane_args.get("workers") or defaults.get("workers") or 3)
    timeout = float(
        lane_args.get("request_timeout") or defaults.get("request_timeout") or 120
    )
    rounds = math.ceil(OPERATION_COUNT / max(workers, 1))
    return math.ceil(rounds * RETRY_ATTEMPTS * 2 * timeout / 60.0)


def stall_thresholds(budget_minutes: int) -> tuple[int, int]:
    """(warn, alert) minutes. Env overrides win."""
    warn = max(2 * budget_minutes, 60)
    alert = max(3 * budget_minutes, 90)
    override_warn = os.environ.get("MR33_STALL_WARN_MIN")
    override_alert = os.environ.get("MR33_STALL_ALERT_MIN")
    if override_warn:
        try:
            warn = int(override_warn)
        except ValueError:
            pass
    if override_alert:
        try:
            alert = int(override_alert)
        except ValueError:
            pass
    return warn, alert


def update_stall(
    previous: dict[str, object],
    latest_pair: int | None,
    pid: str | None,
    alive: bool,
    now: int,
    warn_minutes: int,
    alert_minutes: int,
) -> dict[str, object]:
    """Advance the per-lane no-progress tracker.

    The progress predicate is ``!=``, never ``>``: on resume the generator
    re-prints a progress line for every already-completed source (``pending=0``),
    so ``latest_pair`` rewinds to 0 and climbs back. A ``>`` comparison would
    leave a stale streak in place across a restart.

    ``delta_pairs`` (checkpoint pairs, flushed only every --checkpoint-every
    sources) is deliberately NOT an input: it reads 0 for several consecutive
    cycles while the lane is making progress.
    """
    previous_pair = previous.get("latest_pair")
    previous_pid = previous.get("pid")
    previous_change = previous.get("change_epoch")
    previous_streak = int(previous.get("streak") or 0)

    if latest_pair is None:
        return {
            "latest_pair": None,
            "pid": pid,
            "change_epoch": previous_change or now,
            "streak": previous_streak,
            "stall_state": "unknown",
            "stalled_minutes": None,
        }

    if previous_pair is None or pid != previous_pid:
        # First observation for this lane, or the lane was relaunched. A restart
        # is activity, so reseed rather than carry a stale stall streak forward.
        return {
            "latest_pair": latest_pair,
            "pid": pid,
            "change_epoch": now,
            "streak": 0,
            "stall_state": "down" if not alive else "seeded",
            "stalled_minutes": 0,
        }

    if latest_pair != previous_pair:
        return {
            "latest_pair": latest_pair,
            "pid": pid,
            "change_epoch": now,
            "streak": 0,
            "stall_state": "down" if not alive else (
                "rescan" if latest_pair < previous_pair else "progress"
            ),
            "stalled_minutes": 0,
        }

    change_epoch = int(previous_change or now)
    stalled = max(0, (now - change_epoch) // 60)
    streak = previous_streak + 1 if alive else previous_streak
    if not alive:
        state = "down"
    elif stalled >= alert_minutes:
        state = LEVEL_ALERT
    elif stalled >= warn_minutes:
        state = LEVEL_WARN
    else:
        state = LEVEL_OK
    return {
        "latest_pair": latest_pair,
        "pid": pid,
        "change_epoch": change_epoch,
        "streak": streak,
        "stall_state": state,
        "stalled_minutes": stalled,
    }


def stall_level(stall_state: str, dataset: str, report_status: str = "none") -> str:
    """Map a stall state to a report level, honouring the recovery scope.

    A lane that finished normally is left ``down`` by design: the generator
    exits once it has written ``<dataset>.report.json``. Completion is not a
    stall, so a ``complete`` report keeps an in-scope lane from alerting --
    otherwise every check after a lane finishes reports a false ALERT.
    """
    if stall_state == LEVEL_ALERT:
        return LEVEL_ALERT
    if stall_state == LEVEL_WARN:
        return LEVEL_WARN
    if stall_state == "down" and dataset in RECOVER_DATASETS:
        if report_status == "complete":
            return LEVEL_OK
        return LEVEL_ALERT
    return LEVEL_OK


def classify_gpu(free_mib: int | None, temp_c: int | None) -> tuple[str, list[str]]:
    """Record-only GPU classification. Returns (level, reasons)."""
    level = LEVEL_OK
    reasons: list[str] = []
    if temp_c is not None:
        if temp_c >= GPU_TEMP_ALERT_C:
            level = LEVEL_ALERT
            reasons.append(f"temp>={GPU_TEMP_ALERT_C}")
        elif temp_c >= GPU_TEMP_WARN_C:
            level = max_level(level, LEVEL_WARN)
            reasons.append(f"temp>={GPU_TEMP_WARN_C}")
    if free_mib is not None:
        if free_mib <= GPU_FREE_ALERT_MIB:
            level = LEVEL_ALERT
            reasons.append(f"free<={GPU_FREE_ALERT_MIB}")
        elif free_mib <= GPU_FREE_WARN_MIB:
            level = max_level(level, LEVEL_WARN)
            reasons.append(f"free<={GPU_FREE_WARN_MIB}")
    return level, reasons


def classify_ckpt_lag(lag: int | None, checkpoint_every: int) -> str:
    """Distance between in-memory progress and the flushed checkpoint."""
    if lag is None or lag < 0:
        return LEVEL_OK
    if lag > CKPT_LAG_ALERT_MULTIPLE * checkpoint_every:
        return LEVEL_ALERT
    if lag > checkpoint_every:
        return LEVEL_WARN
    return LEVEL_OK


def max_level(*levels: str) -> str:
    worst = LEVEL_OK
    for level in levels:
        if LEVEL_RANK.get(level, 0) > LEVEL_RANK[worst]:
            worst = level
    return worst


def verdict_for(levels: list[str]) -> tuple[str, int]:
    worst = max_level(*levels) if levels else LEVEL_OK
    if worst == LEVEL_ALERT:
        return "ALERT", 2
    if worst == LEVEL_WARN:
        return "WARN", 1
    return "OK", 0


def scan_growth(
    path: Path, offset: int, carry: dict[str, object], dataset: str
) -> tuple[dict[str, object], int]:
    """Consume only the new complete lines of a log, from a byte offset.

    A partially written trailing line is left unconsumed: the returned offset
    points at its first byte so the next invocation re-reads it. Without this a
    line caught mid-write would be dropped or double-counted.
    """
    if not path.is_file():
        return carry, 0
    size = path.stat().st_size
    if size < offset:
        # Truncated or rotated -- the lane restarted with a fresh log.
        offset = 0
        carry = {**carry, "_reset": True}
    with path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
    if not chunk:
        return carry, offset
    text = chunk.decode("utf-8", errors="replace")
    complete, _, partial = text.rpartition("\n")
    new_offset = size - len(partial.encode("utf-8"))

    updated = dict(carry)
    if carry.get("_reset"):
        updated["error_count"] = 0
        updated["progress_lines"] = 0
        updated.pop("_reset", None)
    for line in complete.splitlines():
        pair = parse_progress_pair(line, dataset)
        if pair is not None:
            updated["latest_pair"] = pair
            updated["latest_line"] = line
            updated["progress_lines"] = int(updated.get("progress_lines") or 0) + 1
        elif line.startswith("RuntimeError:"):
            updated["error_count"] = int(updated.get("error_count") or 0) + 1
            updated["last_error"] = line[:400]
    return updated, new_offset


def read_boot_epoch() -> float | None:
    """Host boot time from /proc/stat.

    Deliberately NOT /proc/uptime: inside this container the two disagree by
    several days (uptime reads ~59.7d while a process's starttime tick count
    implies ~64d against btime), which made ages come out negative. `ps -o
    etimes` is broken the same way, so btime + starttime is the reliable pair.
    """
    try:
        with open("/proc/stat", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return None


def process_start_epoch(
    pid: str, boot_epoch: float | None, clock_ticks: int
) -> float | None:
    """Absolute start time of a process from /proc/<pid>/stat field 22.

    The comm field is parenthesised and may contain spaces, so everything before
    the final ')' is discarded before splitting into fields.
    """
    if boot_epoch is None:
        return None
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read()
        after_comm = raw.rsplit(b")", 1)[1].split()
        starttime_ticks = int(after_comm[19])
    except (OSError, IndexError, ValueError):
        return None
    return boot_epoch + starttime_ticks / clock_ticks


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = int(seconds)
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def format_delta(current: int | None, previous: int | None) -> str:
    if current is None:
        return "-"
    if previous is None:
        return f"{current}"
    delta = current - previous
    return f"{current}(+{delta})" if delta >= 0 else f"{current}({delta})"


def format_report(sections: dict[str, object]) -> str:
    """Render the fixed-prefix text report. SUMMARY is always the last line."""
    lines: list[str] = []
    header = sections["header"]
    lines.append(
        "# MR33CHECK {ts} run={run} verdict={verdict} exit={exit_code} "
        "alerts={alerts} warns={warns}".format(**header)
    )

    for lane in sections["lanes"]:
        scope = "recover" if lane["dataset"] in RECOVER_DATASETS else "report"
        state = "UP" if lane["alive"] else "DOWN"
        lines.append(
            f"LANE {lane['dataset']} scope={scope} state={state} "
            f"pid={lane['pid'] or 'none'} age={lane['age']} "
            f"args=w{lane['workers']}/t{lane['timeout']}/ck{lane['checkpoint_every']} "
            f"report={lane['report']} budget_min={lane['budget_minutes']}"
        )
        lines.append(
            f"LANE {lane['dataset']} "
            f"latest_pair={lane['latest_pair']}"
            f"(+{lane['latest_delta']},{lane['latest_ago_minutes']}m_ago) "
            f"streak={lane['streak']} stalled_min={lane['stalled_minutes']} "
            f"ckpt={format_delta(lane['checkpoint_pairs'], lane['checkpoint_pairs_prev'])} "
            f"lag={lane['ckpt_lag']} "
            f"out={format_delta(lane['output_bytes'], lane['output_bytes_prev'])} "
            f"log={format_delta(lane['log_bytes'], lane['log_bytes_prev'])} "
            f"err={format_delta(lane['error_count'], lane['error_count_prev'])} "
            f"lines={lane['progress_lines']} stall={lane['stall_state']}"
        )
        lines.append(f"LANE {lane['dataset']} last_error={lane['last_error']}")

    model_parts = [f"{name}={state}" for name, state in sections["models"]]
    lines.append("MODEL " + " | ".join(model_parts))

    for gpu in sections["gpus"]:
        reason = (" " + ",".join(gpu["reasons"])) if gpu["reasons"] else ""
        lines.append(
            f"GPU gpu{gpu['index']} used={gpu['used']}/{gpu['total']} "
            f"free={gpu['free']}(d={gpu['free_delta']}) util={gpu['util']} "
            f"temp={gpu['temp']}C {gpu['level']}{reason}"
        )

    monitor = sections["monitor"]
    lines.append(
        f"MONITOR loop={'UP' if monitor['loop_alive'] else 'DOWN'} "
        f"pid={monitor['loop_pid'] or 'none'} tmux={'UP' if monitor['tmux_alive'] else 'DOWN'} "
        f"lock={monitor['lock']} last_cycle={monitor['last_cycle']} "
        f"age_min={monitor['last_cycle_age_minutes']} scheduler={monitor['scheduler']}"
    )

    for finding in sections["findings"]:
        lines.append(f"{finding['level'].upper()} {finding['kind']} {finding['detail']}")

    summary = sections["summary"]
    lines.append(
        "SUMMARY verdict={verdict} lanes_up={lanes_up}/{lanes_total} "
        "in_scope={in_scope} stalled={stalled} gpu_max_temp={gpu_max_temp}C "
        "gpu_min_free={gpu_min_free}MiB models={models} monitor={monitor} "
        "errors_new={errors_new}".format(**summary)
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #


def run_command(argv: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, check=False
    )


def resolve_lms() -> str | None:
    found = shutil.which("lms")
    if found:
        return found
    return LMS_FALLBACK if Path(LMS_FALLBACK).is_file() else None


def read_process(argv0: list[str]) -> tuple[str | None, bool, float | None]:
    """Return (pid, alive, age_seconds) for the pid recorded in a pid file."""
    if not argv0:
        return None, False, None
    pid = argv0[0]
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.is_file():
        return pid, False, None
    try:
        raw = stat_path.read_bytes()
        state = raw.rsplit(b")", 1)[1].split()[0].decode()
        if state == "Z":
            return pid, False, None
    except (OSError, IndexError):
        return pid, False, None
    age = None
    try:
        start_epoch = process_start_epoch(
            pid, read_boot_epoch(), os.sysconf("SC_CLK_TCK")
        )
        if start_epoch is not None:
            age = max(0.0, datetime.now().timestamp() - start_epoch)
    except (OSError, ValueError):
        pass
    return pid, True, age


def read_cmdline(pid: str) -> list[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def read_pid_file(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    return [content] if content else []


def checkpoint_stats(dataset: str) -> dict[str, object] | None:
    """Read-only checkpoint stats. Never passes --snapshot."""
    try:
        result = run_command(
            [PYTHON_BIN, str(VERIFY_SCRIPT), "--dataset", dataset], timeout=120
        )
    except (subprocess.TimeoutExpired, OSError):
        return {"error": "verify_timeout"}
    if result.returncode != 0:
        return {"error": (result.stderr or result.stdout or "verify_failed").strip()[:300]}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": "verify_unparseable"}


def read_report_status(dataset: str, output_root: Path) -> str:
    path = output_root / f"{dataset}.report.json"
    if not path.is_file():
        return "none"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status", "unknown")
    except (OSError, json.JSONDecodeError):
        return "unreadable"


def read_models() -> list[tuple[str, str]]:
    lms = resolve_lms()
    if lms is None:
        return [(name, "UNKNOWN") for name in MODEL_IDENTIFIERS]
    try:
        result = run_command([lms, "ps", "--json"], timeout=30)
        entries = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
        return [(name, "UNKNOWN") for name in MODEL_IDENTIFIERS]
    loaded = {entry.get("identifier") for entry in entries if isinstance(entry, dict)}
    return [
        (name, "LOADED" if name in loaded else "MISSING") for name in MODEL_IDENTIFIERS
    ]


def read_gpus(previous: dict[str, object]) -> list[dict[str, object]]:
    try:
        result = run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,memory.free,"
                "utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            index, used, total, free, util, temp = (
                parts[0],
                int(parts[1]),
                int(parts[2]),
                int(parts[3]),
                int(parts[4]),
                int(parts[5]),
            )
        except ValueError:
            continue
        prior = previous.get(index)
        level, reasons = classify_gpu(free, temp)
        gpus.append(
            {
                "index": index,
                "used": used,
                "total": total,
                "free": free,
                "free_delta": (free - int(prior)) if prior is not None else 0,
                "util": util,
                "temp": temp,
                "level": level,
                "reasons": reasons,
            }
        )
    return gpus


def read_monitor(supervisor_log: Path, lock_path: Path, now: int) -> dict[str, object]:
    tmux_alive = False
    loop_pid = None
    try:
        probe = run_command(["tmux", "has-session", "-t", TMUX_SESSION], timeout=10)
        tmux_alive = probe.returncode == 0
        if tmux_alive:
            pane = run_command(
                ["tmux", "display-message", "-p", "-t", TMUX_SESSION, "#{pane_pid}"],
                timeout=10,
            )
            candidate = pane.stdout.strip()
            if candidate:
                argv = read_cmdline(candidate)
                if any("run_mr_test_v3_3_monitor_loop.sh" in token for token in argv):
                    loop_pid = candidate
                if any("monitor_mr_test_v3_3_full.sh" in token for token in argv):
                    loop_pid = loop_pid or candidate
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Non-blocking probe only: the loop holds this lock for its entire lifetime.
    lock_held = None
    if lock_path.is_file():
        try:
            probe = run_command(
                ["flock", "-n", str(lock_path), "-c", "true"], timeout=10
            )
            lock_held = probe.returncode != 0
        except (subprocess.TimeoutExpired, OSError):
            lock_held = None

    last_cycle = None
    age_minutes = None
    if supervisor_log.is_file():
        try:
            with supervisor_log.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if "supervisor cycle complete" in line or "one-shot supervisor check complete" in line:
                        last_cycle = line[1:20] if line.startswith("[") else line.strip()[:19]
        except OSError:
            pass
    if last_cycle:
        try:
            # supervisor.log stamps are local time; a naive datetime converted
            # with astimezone() is interpreted as local, which is what we want.
            stamp = datetime.strptime(last_cycle, "%Y-%m-%d %H:%M:%S").astimezone()
            age_minutes = max(0, int((now - stamp.timestamp()) // 60))
        except ValueError:
            age_minutes = None

    loop_alive = bool(loop_pid) or tmux_alive or bool(lock_held)
    return {
        "loop_alive": loop_alive,
        "loop_pid": loop_pid,
        "tmux_alive": tmux_alive,
        "lock": "held" if lock_held else ("free" if lock_held is False else "unknown"),
        "last_cycle": last_cycle or "none",
        "last_cycle_age_minutes": age_minutes if age_minutes is not None else "-",
        "scheduler": "loop+systemd",
    }


def load_state(path: Path) -> dict[str, object]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


@dataclass
class Paths:
    run_root: Path = RUN_ROOT
    output_root: Path = OUTPUT_ROOT
    state_path: Path = STATE_PATH
    supervisor_log: Path = field(default=None)  # type: ignore[assignment]
    lock_path: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.supervisor_log is None:
            self.supervisor_log = self.run_root / "supervisor.log"
        if self.lock_path is None:
            self.lock_path = self.run_root / "supervisor.lock"


def collect(paths: Paths, state: dict[str, object]) -> dict[str, object]:
    now = int(datetime.now().timestamp())
    previous_lanes = state.get("lanes") or {}
    findings: list[dict[str, str]] = []
    levels: list[str] = []
    lanes_out: list[dict[str, object]] = []
    errors_new = 0

    for dataset in DATASETS:
        lane_prev = previous_lanes.get(dataset) or {}
        pid_list = read_pid_file(paths.run_root / f"{dataset}.pid")
        pid, alive, age = read_process(pid_list)
        argv = read_cmdline(pid) if (pid and alive) else []
        lane_alive = is_dataset_generator(argv, dataset)
        if pid and alive and not lane_alive:
            # A live pid that is not this dataset's generator: stale pid file.
            lane_alive = False

        lane_args = parse_lane_args(argv) if lane_alive else {}
        if not lane_args:
            lane_args = dict(LANE_DEFAULTS.get(dataset, {}))
        workers = lane_args.get("workers") or "-"
        timeout = lane_args.get("request_timeout")
        if isinstance(timeout, float) and timeout.is_integer():
            timeout = int(timeout)
        timeout = timeout or "-"
        checkpoint_every = int(lane_args.get("checkpoint_every") or 10)
        budget = stall_budget_minutes(lane_args, dataset)
        warn_minutes, alert_minutes = stall_thresholds(budget)

        log_path = paths.run_root / f"{dataset}.stdout.log"
        carry, offset = scan_growth(
            log_path, int(lane_prev.get("offset") or 0), lane_prev, dataset
        )
        latest_pair = carry.get("latest_pair")
        stall = update_stall(
            lane_prev,
            latest_pair if isinstance(latest_pair, int) else None,
            pid if lane_alive else None,
            lane_alive,
            now,
            warn_minutes,
            alert_minutes,
        )
        stall_state = str(stall["stall_state"])
        report_status = read_report_status(dataset, paths.output_root)
        level = stall_level(stall_state, dataset, report_status)
        levels.append(level)
        if level != LEVEL_OK:
            findings.append(
                {
                    "level": level,
                    "kind": "stall",
                    "detail": (
                        f"dataset={dataset} latest_pair={latest_pair} "
                        f"stalled_minutes={stall['stalled_minutes']} "
                        f"state={stall_state} threshold_warn={warn_minutes} "
                        f"threshold_alert={alert_minutes}"
                    ),
                }
            )
        if stall_state == "down" and dataset in REPORT_ONLY_DATASETS:
            findings.append(
                {
                    "level": LEVEL_OK,
                    "kind": "report_only",
                    "detail": (
                        f"dataset={dataset} stopped; report-only lane, "
                        "no recovery attempted"
                    ),
                }
            )

        stats = checkpoint_stats(dataset)
        ckpt_pairs = stats.get("completed_pairs") if stats and "error" not in stats else None
        output_bytes = stats.get("output_bytes") if stats and "error" not in stats else None
        log_bytes = stats.get("log_bytes") if stats and "error" not in stats else None
        if stats and "error" in stats:
            levels.append(LEVEL_ALERT)
            findings.append(
                {
                    "level": LEVEL_ALERT,
                    "kind": "ckpt_invalid",
                    "detail": f"dataset={dataset} {stats['error']}",
                }
            )

        lag = None
        if isinstance(ckpt_pairs, int) and isinstance(latest_pair, int):
            lag = latest_pair + 1 - ckpt_pairs
        lag_level = classify_ckpt_lag(lag, checkpoint_every)
        if lag_level != LEVEL_OK:
            levels.append(lag_level)
            findings.append(
                {
                    "level": lag_level,
                    "kind": "ckpt_lag",
                    "detail": (
                        f"dataset={dataset} lag={lag} checkpoint_every={checkpoint_every}"
                    ),
                }
            )

        previous_errors = int(lane_prev.get("error_count") or 0)
        current_errors = int(carry.get("error_count") or 0)
        # Only a delta against a previous observation is "new". Treating the
        # first run's whole log as new would fire a spurious WARN on every
        # freshly created state file.
        if lane_prev and current_errors > previous_errors:
            errors_new += current_errors - previous_errors
            levels.append(LEVEL_WARN)
            findings.append(
                {
                    "level": LEVEL_WARN,
                    "kind": "runtime_error",
                    "detail": f"dataset={dataset} new={current_errors - previous_errors} "
                    f"last={carry.get('last_error') or 'none'}",
                }
            )

        change_epoch = int(stall.get("change_epoch") or now)
        lanes_out.append(
            {
                "dataset": dataset,
                "alive": lane_alive,
                # A stale pid file must not be presented as a live process.
                "pid": pid if lane_alive else None,
                "pid_file_value": pid,
                "age": format_duration(age) if lane_alive else "-",
                "workers": workers,
                "timeout": timeout,
                "checkpoint_every": checkpoint_every,
                "report": report_status,
                "budget_minutes": budget,
                "latest_pair": latest_pair if latest_pair is not None else "-",
                "latest_delta": (
                    (latest_pair - int(lane_prev.get("latest_pair")))
                    if isinstance(latest_pair, int)
                    and isinstance(lane_prev.get("latest_pair"), int)
                    else 0
                ),
                "latest_ago_minutes": max(0, (now - change_epoch) // 60),
                "streak": stall.get("streak", 0),
                "stalled_minutes": (
                    stall.get("stalled_minutes")
                    if stall.get("stalled_minutes") is not None
                    else "-"
                ),
                "stall_state": stall_state,
                "checkpoint_pairs": ckpt_pairs,
                "checkpoint_pairs_prev": lane_prev.get("checkpoint_pairs"),
                "ckpt_lag": lag if lag is not None else "-",
                "output_bytes": output_bytes,
                "output_bytes_prev": lane_prev.get("output_bytes"),
                "log_bytes": log_bytes,
                "log_bytes_prev": lane_prev.get("log_bytes"),
                "error_count": current_errors,
                "error_count_prev": lane_prev.get("error_count"),
                "last_error": carry.get("last_error") or "none",
                "progress_lines": carry.get("progress_lines") or 0,
                "offset": offset,
            }
        )

    models = read_models()
    for name, status in models:
        if status != "LOADED":
            levels.append(LEVEL_ALERT)
            findings.append(
                {
                    "level": LEVEL_ALERT,
                    "kind": "model_missing",
                    "detail": f"identifier={name} state={status}",
                }
            )

    gpus = read_gpus(state.get("gpus") or {})
    if not gpus:
        levels.append(LEVEL_WARN)
        findings.append(
            {"level": LEVEL_WARN, "kind": "gpu_query", "detail": "nvidia-smi returned nothing"}
        )
    for gpu in gpus:
        if gpu["level"] != LEVEL_OK:
            levels.append(gpu["level"])
            findings.append(
                {
                    "level": gpu["level"],
                    "kind": "gpu_resource",
                    "detail": f"index={gpu['index']} free={gpu['free']}MiB "
                    f"temp={gpu['temp']}C reasons={','.join(gpu['reasons'])} "
                    "(record-only; shared tenant)",
                }
            )

    monitor = read_monitor(paths.supervisor_log, paths.lock_path, now)
    if not monitor["tmux_alive"] and not monitor["loop_alive"]:
        levels.append(LEVEL_ALERT)
        findings.append(
            {
                "level": LEVEL_ALERT,
                "kind": "monitor_down",
                "detail": "tmux session and lock both absent; recovery layer not running",
            }
        )
    elif monitor["last_cycle_age_minutes"] not in ("-", None) and int(
        monitor["last_cycle_age_minutes"]
    ) > MONITOR_CYCLE_STALE_MIN:
        levels.append(LEVEL_WARN)
        findings.append(
            {
                "level": LEVEL_WARN,
                "kind": "monitor_stale",
                "detail": f"last cycle {monitor['last_cycle_age_minutes']}m ago",
            }
        )

    verdict, exit_code = verdict_for(levels)
    alerts = sum(1 for f in findings if f["level"] == LEVEL_ALERT)
    warns = sum(1 for f in findings if f["level"] == LEVEL_WARN)
    lanes_up = sum(1 for lane in lanes_out if lane["alive"])
    stalled = sum(
        1 for lane in lanes_out if lane["stall_state"] in (LEVEL_WARN, LEVEL_ALERT)
    )
    gpu_max_temp = max((gpu["temp"] for gpu in gpus), default="-")
    gpu_min_free = min((gpu["free"] for gpu in gpus), default="-")

    previous_runs = int(state.get("runs") or 0)
    header = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": previous_runs + 1,
        "verdict": verdict,
        "exit_code": exit_code,
        "alerts": alerts,
        "warns": warns,
    }
    summary = {
        "verdict": verdict,
        "lanes_up": lanes_up,
        "lanes_total": len(lanes_out),
        "in_scope": ",".join(RECOVER_DATASETS),
        "stalled": stalled,
        "gpu_max_temp": gpu_max_temp,
        "gpu_min_free": gpu_min_free,
        "models": f"{sum(1 for _, s in models if s == 'LOADED')}/{len(models)}",
        "monitor": "UP" if monitor["loop_alive"] else "DOWN",
        "errors_new": errors_new,
    }

    new_state = {
        "runs": previous_runs + 1,
        "last_run": header["ts"],
        "lanes": {lane["dataset"]: lane for lane in lanes_out},
        "gpus": {gpu["index"]: gpu["free"] for gpu in gpus},
    }

    return {
        "header": header,
        "lanes": lanes_out,
        "models": models,
        "gpus": gpus,
        "monitor": monitor,
        "findings": findings,
        "summary": summary,
        "levels": levels,
        "exit_code": exit_code,
        "verdict": verdict,
        "new_state": new_state,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--no-persist", action="store_true", help="do not write state")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    paths = Paths(
        run_root=args.run_root, output_root=args.output_root, state_path=args.state
    )
    state = {} if args.no_persist else load_state(args.state)
    try:
        sections = collect(paths, state)
    except Exception as exc:  # noqa: BLE001 - the checker must always report
        print(f"# MR33CHECK ERROR checker_failed: {type(exc).__name__}: {exc}")
        return 3

    if args.json:
        payload = {
            key: value for key, value in sections.items() if key not in ("new_state",)
        }
        print(json.dumps(payload, indent=2, default=str, sort_keys=True))
    else:
        print(format_report(sections))

    if not args.no_persist:
        try:
            save_state(args.state, sections["new_state"])
        except OSError as exc:
            print(f"# MR33CHECK WARN state_write_failed {exc}")
            return max(sections["exit_code"], 1)
    return int(sections["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
