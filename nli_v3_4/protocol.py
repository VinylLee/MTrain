"""Same-model response protocol: strict labels and verdicts, flexible JSON envelope."""
from __future__ import annotations
import json
import re

from openai import APIConnectionError, APITimeoutError, APIStatusError
import augment_snli_all_label_mrs as legacy
from copy import deepcopy
from .data import source_uid

PROTOCOL_VERSION = "gemma-verdict-envelope-v2"


class ResponseFormatError(ValueError):
    pass


class InterruptedCallError(RuntimeError):
    """A previously dispatched API call has no durably recorded response."""


def generator_json(raw):
    """Read one complete JSON value, allowing only redundant trailing closers."""
    cleaned=legacy.strip_think(raw).strip()
    fence=re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```",cleaned,re.I)
    if fence:cleaned=fence.group(1).strip()
    try:value,end=json.JSONDecoder().raw_decode(cleaned)
    except ValueError:return legacy.extract_json(raw)
    suffix=cleaned[end:].strip()
    if suffix and (len(suffix)>3 or any(c not in '}]' for c in suffix)):
        raise ResponseFormatError("generator response has extra content after its JSON value")
    return value


def parse_judgments(raw, candidate_count, require_bidirectional=False):
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
        directions = {}
        if require_bidirectional:
            for field in ("source_to_candidate", "candidate_to_source"):
                text = evaluation.get(field)
                if not isinstance(text, str) or text.strip().lower() not in legacy.LABEL_IDS:
                    break
                directions[field] = text.strip().lower()
            if len(directions) != 2:
                continue
            # Same NLI label relative to P is insufficient for INV equivalence.
            # Each directional judgment concerns only the two hypotheses.
            if any(label != "entailment" for label in directions.values()):
                evaluation = {**evaluation, "source_equivalent": False}
        judgments[index] = {**labels, **{k: evaluation[k] for k in required},
                            **directions, "reason": str(evaluation.get("reason", "other"))}
    return judgments


def exception_fields(exc):
    if isinstance(exc, InterruptedCallError):
        return {"status": "interrupted_call", "error_kind": "interrupted_call",
                "error": f"{type(exc).__name__}: {exc}"}
    if isinstance(exc, (APIConnectionError, APITimeoutError, APIStatusError)):
        return {"status": "error", "error_kind": "api", "error": f"{type(exc).__name__}: {exc}"}
    if isinstance(exc, ValueError):
        return {"status": "invalid_response", "error_kind": "response_format",
                "error": f"{type(exc).__name__}: {exc}"}
    # Unexpected programming failures must not disappear inside a model rejection.
    raise exc


def parse_equivalence_judgments(raw, candidate_count):
    """Validate standalone H/H judgments without inventing P/H labels."""
    value = legacy.extract_json(raw)
    evaluations = value if isinstance(value, list) else value.get("evaluations") if isinstance(value, dict) else None
    if not isinstance(evaluations, list):
        raise ResponseFormatError("equivalence response lacks an evaluation array")
    result = {}; indices = set()
    for entry in evaluations:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        if type(index) is not int or not 0 <= index < candidate_count or index in indices:
            raise ResponseFormatError("ambiguous equivalence candidate index")
        indices.add(index)
        if any(type(entry.get(k)) is not bool for k in ("grammatical", "operation_valid", "source_equivalent")):
            continue
        directions = {}
        for key in ("source_to_candidate", "candidate_to_source"):
            label = entry.get(key)
            if not isinstance(label, str) or label.strip().lower() not in legacy.LABEL_IDS:
                break
            directions[key] = label.strip().lower()
        if len(directions) != 2:
            continue
        result[index] = {**directions, **{k:entry[k] for k in ("grammatical", "operation_valid", "source_equivalent")},
                         "reason": str(entry.get("reason", "other"))}
    return result


def merge_inv_judgments(pair_judgments, equivalence_judgments):
    """Require independent P/H label and both H/H entailment directions."""
    result = {}
    for index, pair in pair_judgments.items():
        eq = equivalence_judgments.get(index)
        if eq is None:
            continue
        result[index] = {**pair, "source_to_candidate":eq["source_to_candidate"],
                        "candidate_to_source":eq["candidate_to_source"],
                        "source_equivalent":eq["source_equivalent"] and all(eq[k] == "entailment"
                             for k in ("source_to_candidate", "candidate_to_source")),
                        "grammatical":pair["grammatical"] and eq["grammatical"],
                        "operation_valid":pair["operation_valid"] and eq["operation_valid"],
                        "equivalence_reason":eq["reason"]}
    return result


def choose_invariant_candidate(candidates, judgments, source, operation):
    """INV gold follows from verified equivalence, never a new label prediction."""
    if legacy.OPERATION_SPECS[operation]["family"] != "inv":
        raise ValueError("invariant chooser must not cross MR family")
    for index, candidate in enumerate(candidates):
        verdict = judgments.get(index)
        if verdict is None or not all(verdict.get(k) is True for k in
                ("grammatical", "operation_valid", "source_equivalent")):
            continue
        if any(verdict.get(k) != "entailment" for k in ("source_to_candidate", "candidate_to_source")):
            continue
        return candidate, {**verdict, "output_label": legacy.LABELS[source["label"]],
                           "label_provenance": "INV_equivalence_preserves_source_gold"}
    return None


def has_api_failure(trace):
    if isinstance(trace, dict):
        return trace.get("error_kind") == "api" or any(has_api_failure(v) for v in trace.values())
    if isinstance(trace, list):
        return any(has_api_failure(v) for v in trace)
    return False


def replay_atomic(source, operation, trace, candidate_parser, preserved_text=None,
                  require_bidirectional=False):
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
            if require_bidirectional:
                raw_equivalence = attempt.get("raw_equivalence_judgment")
                if not raw_equivalence:
                    continue
                verdicts = parse_equivalence_judgments(raw_equivalence, len(candidates))
            else:
                verdicts = parse_judgments(raw_judgment, len(candidates))
        except ResponseFormatError:
            continue
        chooser = choose_invariant_candidate if require_bidirectional else legacy.choose_candidate
        if preserved_text is None:
            chosen = chooser(candidates, verdicts, source, operation)
        else:
            chosen = None
            for index, candidate in enumerate(candidates):
                if candidate == preserved_text and index in verdicts:
                    chosen = chooser([candidate], {0: verdicts[index]}, source, operation)
                    if chosen:
                        break
        attempt["protocol_replay_judgments"] = verdicts
        if chosen:
            log["protocol_replay"] = {"protocol": PROTOCOL_VERSION, "original_status": trace.get("status")}
            log.update(status="accepted", selected_hypothesis=chosen[0])
            return chosen[0], log
    return None, log
