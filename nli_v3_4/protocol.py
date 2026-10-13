"""Same-model response protocol: strict labels and verdicts, flexible JSON envelope."""
from __future__ import annotations
import json

from openai import APIConnectionError, APITimeoutError, APIStatusError
import augment_snli_all_label_mrs as legacy
from copy import deepcopy
from .data import source_uid

PROTOCOL_VERSION = "gemma-verdict-envelope-v2"


class ResponseFormatError(ValueError):
    pass


def parse_judgments(raw, candidate_count):
    """Accept a list or {evaluations: list}; never invent labels, indices or booleans."""
    try:
        value = legacy.extract_json(raw)
    except ValueError as exc:
        raise ResponseFormatError(str(exc)) from exc
    evaluations = value if isinstance(value, list) else value.get("evaluations") if isinstance(value, dict) else None
    if not isinstance(evaluations, list):
        raise ResponseFormatError("judge response lacks an evaluation array")
    judgments = {}
    indices = set()
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            continue
        index = evaluation.get("index")
        if type(index) is not int or not 0 <= index < candidate_count:
            raise ResponseFormatError("judge index is missing or outside the candidate range")
        if index in indices:
            raise ResponseFormatError("duplicate candidate index; ambiguous verdict")
        indices.add(index)
        # Missing/string booleans are invalid, never coerced into approval.
        required = ("grammatical", "source_equivalent", "operation_valid")
        if any(type(evaluation.get(k)) is not bool for k in required):
            continue
        labels = {}
        for field in ("label", "source_label"):
            text = evaluation.get(field)
            if not isinstance(text, str) or text.strip().lower() not in legacy.LABEL_IDS:
                break
            labels[field] = text.strip().lower()
        if len(labels) != 2:
            continue
        judgments[index] = {**labels, **{k: evaluation[k] for k in required},
                            "reason": str(evaluation.get("reason", "other"))}
    return judgments


def exception_fields(exc):
    if isinstance(exc, (APIConnectionError, APITimeoutError, APIStatusError)):
        return {"status": "error", "error_kind": "api", "error": f"{type(exc).__name__}: {exc}"}
    if isinstance(exc, ValueError):
        return {"status": "invalid_response", "error_kind": "response_format",
                "error": f"{type(exc).__name__}: {exc}"}
    # Unexpected programming failures must not disappear inside a model rejection.
    raise exc


def has_api_failure(trace):
    if isinstance(trace, dict):
        return trace.get("error_kind") == "api" or any(has_api_failure(v) for v in trace.values())
    if isinstance(trace, list):
        return any(has_api_failure(v) for v in trace)
    return False


def replay_atomic(source, operation, trace, candidate_parser, preserved_text=None):
    """Re-read saved model verdicts without another generation or judgment call."""
    log = deepcopy(trace)
    if log.get("source_uid") not in (None, source_uid(source)):
        raise ValueError("saved stage source identity differs from its reconstructed input")
    for attempt in log.get("attempts", []):
        raw_generation = attempt.get("raw_generation")
        raw_judgment = attempt.get("raw_judgment")
        if not raw_generation or not raw_judgment:
            continue
        try:
            candidates = candidate_parser(raw_generation, source, operation)
            if candidates != attempt.get("candidates"):
                raise ValueError("saved candidates differ from their raw response")
            verdicts = parse_judgments(raw_judgment, len(candidates))
        except ResponseFormatError:
            continue
        if preserved_text is None:
            chosen = legacy.choose_candidate(candidates, verdicts, source, operation)
        else:
            chosen = None
            for index, candidate in enumerate(candidates):
                if candidate == preserved_text and index in verdicts:
                    chosen = legacy.choose_candidate([candidate], {0: verdicts[index]}, source, operation)
                    if chosen:
                        break
        attempt["protocol_replay_judgments"] = verdicts
        if chosen:
            log["protocol_replay"] = {"protocol": PROTOCOL_VERSION, "original_status": trace.get("status")}
            log.update(status="accepted", selected_hypothesis=chosen[0])
            return chosen[0], log
    return None, log
