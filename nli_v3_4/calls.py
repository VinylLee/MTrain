"""Durable per-request API call replay, without reissuing uncertain calls."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .data import digest, source_uid
from .protocol import InterruptedCallError

CALL_PROTOCOL = "durable-sequence-v1"


def validate_budget_reservation(entry, source, seed, version):
    """An unknown interrupted request may consume budget, never yield a row."""
    trace = entry.get("trace", {})
    if (entry.get("budget_reservation") is not True or entry.get("row") is not None
            or entry.get("pending_continuation")
            or trace.get("status") != "interrupted_uncertain_budget_reserved"
            or trace.get("actual_model_calls_known") is not False
            or trace.get("model_response_available") is not False
            or entry.get("source_uid") != source_uid(source)
            or type(entry.get("refill_round")) is not int
            or not 0 <= entry["refill_round"] < 4
            or entry.get("request_id") != digest([source_uid(source), entry["operation"],
                                                  entry["refill_round"], seed, version])):
        raise RuntimeError("invalid interruption budget reservation")


def _save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DurableRequestClient:
    """One logical request, replayed from the same starting trace and call order.

    Completed responses are replayed verbatim. A recorded call without a durable
    response consumes its original slot and raises an interruption error; it is
    never sent again. The existing generator decides whether budget remains.
    """

    def __init__(self, client, directory):
        self.client = client
        self.directory = Path(directory)
        self.chat = self.completions = self
        self.sequence = 0
        self.new_calls = self.replayed_calls = self.uncertain_calls = 0

    def create(self, **kwargs):
        sequence = self.sequence
        self.sequence += 1
        path = self.directory / f"{sequence:04d}.json"
        identity = {"protocol": CALL_PROTOCOL, "sequence": sequence,
                    "request_sha256": digest(kwargs)}
        if path.exists():
            record = json.loads(path.read_text())
            if any(record.get(key) != value for key, value in identity.items()):
                raise RuntimeError("durable call identity changed; do not reuse its response or budget")
            state = record.get("state")
            if state == "completed":
                if "content" not in record or not isinstance(record["content"], (str, type(None))):
                    raise RuntimeError("invalid saved completion; preserve it for investigation")
                self.replayed_calls += 1
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=record["content"]))])
            if state not in {"in_flight", "failed"}:
                raise RuntimeError("unknown durable call state")
            self.uncertain_calls += 1
            raise InterruptedCallError(f"call {sequence} has no durable response ({state}); slot consumed")
        # Persist before dispatch: SIGINT, SIGTERM or SIGKILL cannot silently
        # reset a sent call's budget when the logical request is rerun.
        record = {**identity, "state": "in_flight"}
        _save(path, record)
        self.new_calls += 1
        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not isinstance(content, (str, type(None))):
                raise RuntimeError("durable text completion received unsupported content")
        except Exception as exc:
            _save(path, {**record, "state": "failed", "error_type": type(exc).__name__})
            raise
        _save(path, {**record, "state": "completed", "content": content})
        return response
