"""Strict dataset adapters and provenance utilities, with no model dependencies."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

LABELS = ("entailment", "neutral", "contradiction")
DATASETS = ("snli", "mnlim", "mnlimm", "sick")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def is_source(row):
    if "is_source" in row:
        if type(row["is_source"]) is not bool:
            raise ValueError("is_source must be boolean")
        flag = row["is_source"]
    elif "is_syn" in row:
        if type(row["is_syn"]) is not bool:
            raise ValueError("is_syn must be boolean")
        flag = not row["is_syn"]
    else:
        flag = row.get("mr_id", "none") == "none"
    if flag != (row.get("mr_id", "none") == "none"):
        raise ValueError("source markers disagree")
    return flag


def load_dataset(path):
    rows = read_jsonl(path)
    originals = {}
    for position, row in enumerate(rows):
        if type(row.get("label")) is not int or row["label"] not in (0, 1, 2):
            raise ValueError(f"invalid label at {path}:{position + 1}")
        if not all(isinstance(row.get(k), str) and row[k].strip()
                   for k in ("premise", "hypothesis")):
            raise ValueError(f"empty/non-text pair at {path}:{position + 1}")
        if type(row.get("pair_id")) is not int:
            raise ValueError(f"missing/integer pair_id at {path}:{position + 1}")
        if is_source(row):
            if row["pair_id"] in originals:
                raise ValueError("duplicate source pair_id")
            originals[row["pair_id"]] = row
    for row in rows:
        if row["pair_id"] not in originals:
            raise ValueError("orphan follow-up")
        if row["premise"] != originals[row["pair_id"]]["premise"]:
            raise ValueError("changed premise")
    return rows, list(originals.values())


def canonical_source(row):
    return {k: row[k] for k in ("premise", "hypothesis", "label", "pair_id")}


def source_uid(row):
    # Neither corpus name nor train/test role influences prompts or style choice.
    return digest(canonical_source(row))


def quotas(total):
    if total < 0:
        raise ValueError("negative quota")
    return {label: total // 3 + int(label < total % 3) for label in range(3)}


def audit(rows):
    originals = {r["pair_id"]: r for r in rows if is_source(r)}
    follows = [r for r in rows if not is_source(r)]
    groups = defaultdict(list)
    for r in rows:
        groups[(normalize(r["premise"]), normalize(r["hypothesis"]))].append(r)
    conflicts = [v for v in groups.values() if len({r["label"] for r in v}) > 1]
    unchanged = [r for r in follows if normalize(r["hypothesis"]) ==
                 normalize(originals[r["pair_id"]]["hypothesis"])]
    labels = Counter(r["label"] for r in follows)
    return {"rows": len(rows), "sources": len(originals), "followups": len(follows),
            "followup_labels": dict(labels),
            "followup_majority": max(labels.values()) / len(follows) if follows else None,
            "mr_counts": dict(Counter(r["mr_id"] for r in follows)),
            "label_conflicts": conflicts, "unchanged_followups": unchanged}


def export_source(source, schema):
    if schema == "train":
        return {**canonical_source(source), "is_syn": False, "weight": 1.0,
                "mr_id": "none", "mr_type": "inv"}
    return {"idx": -1, **canonical_source(source), "mr_id": "none",
            "mr_type": "inv", "is_source": True}


def export_followup(source, operation, hypothesis, label, family, steps, schema):
    row = {**canonical_source(source), "hypothesis": hypothesis, "label": label,
           "mr_id": operation, "mr_type": family}
    if steps:
        row["component_mrs"] = list(steps)
    if schema == "train":
        row.update(is_syn=True, weight=1.0)
    else:
        row.update(idx=-1, is_source=False)
    return row
