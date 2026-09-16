"""High-quality, pair-aware SNLI augmentation with an isolated judge pass.

The legacy augmentation code transforms the hypothesis without always exposing
the premise to the generator.  This script makes the NLI pair and target label
part of every generation prompt, asks for several candidates, and uses a fresh
stateless Gemma request to accept only candidates that satisfy the relation.

The default invocation is intentionally a small smoke run::

    python augment_snli_v2.py --sample-size 12 --overwrite

Use ``--all`` only after the smoke report passes the configured quality gates.
The input file is never modified.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI


PROMPT_VERSION = "snli-pair-aware-v2.3"
LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_IDS = {value: key for key, value in LABELS.items()}
NEGATION_TOKENS = {
    "no", "not", "never", "none", "nobody", "nothing", "neither", "without",
    "cannot",
}
CONTENT_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "has", "have", "had", "to", "of", "in",
    "on", "at", "for", "from", "with", "and", "or", "but", "as", "by",
    "this", "that", "these", "those", "there", "it",
}

STRICT_QUANTIFIERS = {"all", "every", "each", "many", "most", "few", "several"}
DEFAULT_DATA_DIR = Path(
    "nli_arg_data/snli_lr0.0037_gemma-3-4b-it-qat"
)

GENERATOR_SYSTEM_PROMPT = """You are a meticulous SNLI data editor.
Create short, natural English hypotheses for natural-language inference.
Treat the premise as the only known world. Use the strict SNLI definitions:
- entailment: the premise makes the hypothesis definitely true;
- contradiction: the premise makes the hypothesis definitely false;
- neutral: the hypothesis may be true, but the premise neither proves nor
  disproves it.
Do not rely on ordinary-world assumptions to call something a contradiction.
Do not explain your reasoning. Return only the requested JSON array."""

JUDGE_SYSTEM_PROMPT = """You are a strict, conservative SNLI quality judge.
Classify from the premise alone. Contradiction requires logical incompatibility,
not merely different, missing, unlikely, or newly added information. Entailment
requires the hypothesis to be guaranteed. Otherwise choose neutral.
Judge grammar and operation compliance independently. Return only JSON and do
not provide hidden reasoning; the reason field must be a short error tag."""

OPERATION_SPECS: Dict[str, Dict[str, str]] = {
    "inv": {
        "mr_id": "composite_inv_v2",
        "mr_type": "inv",
        "instruction": """Conservatively paraphrase the SOURCE HYPOTHESIS. Change
only one or two words or make one small syntactic change. Preserve every person,
object, count, named entity, attribute, action, location, and negation. Keep at
least half of the non-function content words unchanged. The result must have
exactly the same truth conditions and SNLI label. Do not generalize or specialize.
If exact preservation is unsafe, emit fewer candidates rather than guessing.""",
        "target": "source",
    },
    "flip": {
        "mr_id": "composite_flip_v2",
        "mr_type": "flip",
        "instruction": """Negate the entire SOURCE HYPOTHESIS so the result is
definitely contradicted by the PREMISE. Preserve its exact subject, predicate
words, objects, counts, named entities, and locations; only add or adjust the
minimum grammar needed for explicit negation. Example: source The dogs are
outside. -> The dogs are not outside. Never produce The dogs are not inside,
because that preserves rather than negates the meaning. Do not replace facts or
add unmentioned details. If safe negation is impossible, emit fewer candidates.""",
        "target": "contradiction",
    },
    "neutral": {
        "mr_id": "composite_neutral_v2",
        "mr_type": "neutral",
        "instruction": """Extend the SOURCE HYPOTHESIS into a short hypothesis
that is neutral relative to the PREMISE. Copy the entire source hypothesis as
the first clause, preserving its words, participants, and polarity, then add
exactly one concrete and plausible atomic detail that the premise neither states
nor rules out (for example an unmentioned purpose, relationship, or specific
attribute). Do not output only the new detail. Do not use generic weather/time
boilerplate or artificial conditionals. If no natural extension is available,
emit fewer candidates rather than guessing.""",
        "target": "neutral",
    },
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .!?\"'")


def content_tokens(text: str) -> set:
    tokens = set(re.findall(r"[a-z]+|[0-9]+", text.lower()))
    return tokens - CONTENT_STOPWORDS - NEGATION_TOKENS


def number_tokens(text: str) -> set:
    return set(re.findall(r"[0-9]+", text.lower()))


def has_negation(text: str) -> bool:
    lowered = text.lower().replace(chr(8217), chr(39))
    tokens = set(re.findall(r"[a-z]+", lowered))
    contraction = "n" + chr(39) + "t"
    return bool(tokens & NEGATION_TOKENS) or contraction in lowered


def is_direct_negation(source_text: str, candidate_text: str) -> bool:
    if normalize_text(source_text).startswith(("here is ", "here are ")):
        return False
    if has_negation(source_text) or not has_negation(candidate_text):
        return False
    without_not = re.sub(r"\bnot\b", "", normalize_text(candidate_text))
    without_not = re.sub(r"\s+", " ", without_not).strip()
    return without_not == normalize_text(source_text)



def is_safe_flip_source(source: Dict[str, Any]) -> bool:
    source_words = set(re.findall(r"[a-z]+", str(source["hypothesis"]).lower()))
    premise_words = set(re.findall(r"[a-z]+", str(source["premise"]).lower()))
    return not ((source_words & STRICT_QUANTIFIERS) - premise_words)

def strip_think(text: str) -> str:
    return re.sub(
        r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE
    ).strip()


def extract_json(text: str) -> Any:
    """Extract the first JSON object/array without accepting arbitrary prose."""
    cleaned = strip_think(text)
    fence = re.fullmatch(
        r"\s*```(?:json)?\s*([\s\S]*?)\s*```\s*", cleaned, re.IGNORECASE
    )
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("response does not contain valid JSON")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            if not all(key in value for key in ("premise", "hypothesis", "label")):
                raise ValueError(f"{path}:{line_number} lacks an NLI field")
            if int(value["label"]) not in LABELS:
                raise ValueError(f"{path}:{line_number} has invalid label")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def stratified_sample(
    rows: Sequence[Dict[str, Any]], size: int, seed: int
) -> List[Dict[str, Any]]:
    if size >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    buckets = {label: [] for label in LABELS}
    for row in rows:
        buckets[int(row["label"])].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    base, remainder = divmod(size, len(buckets))
    selected: List[Dict[str, Any]] = []
    for position, label in enumerate(sorted(buckets)):
        take = base + (1 if position < remainder else 0)
        selected.extend(buckets[label][:take])
    rng.shuffle(selected)
    return selected


def target_label(source: Dict[str, Any], operation: str) -> str:
    target = OPERATION_SPECS[operation]["target"]
    return LABELS[int(source["label"])] if target == "source" else target
def eligible(source: Dict[str, Any], operation: str) -> bool:
    if int(source["label"]) == 0 and not is_safe_flip_source(source):
        return False
    if operation == "inv":
        return True
    if int(source["label"]) != 0:
        return False
    return operation != "flip" or not has_negation(str(source["hypothesis"]))


def build_generation_prompt(
    source: Dict[str, Any], operation: str, candidate_count: int
) -> str:
    spec = OPERATION_SPECS[operation]
    target = target_label(source, operation)
    payload = {
        "premise": source["premise"],
        "source_hypothesis": source["hypothesis"],
        "source_label": LABELS[int(source["label"])],
        "target_label": target,
    }
    neutral_contract = ""
    if operation == "neutral":
        exact_prefix = str(source["hypothesis"]).strip().rstrip(".!?")
        neutral_contract = (
            "\nNEUTRAL FORMAT: Every hypothesis must begin exactly with:\n"
            + json.dumps(exact_prefix + ", and ", ensure_ascii=False)
            + "\nThen complete one new clause; do not paraphrase the prefix.\n"
        )
    return f"""TASK
{spec['instruction']}

PAIR
{json.dumps(payload, ensure_ascii=False, indent=2)}

OUTPUT CONTRACT
Return a JSON array containing up to {candidate_count} distinct objects.
Each object must have exactly these string fields:
{{"hypothesis": "...", "target_label": "{target}"}}
The hypothesis must be one self-contained English statement with no labels,
quotes, commentary, prefixes, or prompt text. Never copy the premise verbatim.
{neutral_contract}

SNLI PITFALLS
- An extra fact absent from the premise is neutral, not contradiction.
- A less specific statement may be entailed but is not an exact paraphrase.
- Changing person, number, color, location, direction, or negation changes truth
  conditions and is forbidden for inv.
- For flip, directly negate or replace a guaranteed atomic fact.
- For neutral, the complete source hypothesis must remain visibly intact as the
  first clause; append only one unsupported plausible fact."""


def build_judge_prompt(
    source: Dict[str, Any], operation: str, candidates: Sequence[str]
) -> str:
    payload = {
        "premise": source["premise"],
        "source_hypothesis": source["hypothesis"],
        "source_label": LABELS[int(source["label"])],
        "requested_operation": operation,
        "target_label": target_label(source, operation),
        "candidates": [
            {"index": index, "hypothesis": text}
            for index, text in enumerate(candidates)
        ],
    }
    return f"""Evaluate every candidate in this payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}

For each candidate independently determine:
1. source_label: the strict SNLI label of premise -> source_hypothesis; do not
   trust the supplied source_label blindly (playing is not necessarily enjoying,
   setting a table is not seated, and multiple does not necessarily mean many);
2. label: the strict SNLI label of premise -> candidate hypothesis;
3. grammatical: whether the candidate is natural, self-contained English;
4. source_equivalent: whether candidate and source have identical truth conditions;
5. operation_valid: whether the requested edit has no artifacts or shortcuts.

Return exactly one JSON object of this form and preserve candidate indices:
{{"evaluations":[{{"index":0,"source_label":"entailment","label":"entailment",
"grammatical":true,"source_equivalent":true,"operation_valid":true,"reason":"ok"}}]}}
Allowed reason values: ok, wrong_label, meaning_drift, grammar, artifact,
not_changed, or other. No prose outside JSON."""


def _completion(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def parse_candidates(
    raw: str,
    source: Dict[str, Any],
    operation: str,
    candidate_count: int,
) -> List[str]:
    if int(source["label"]) == 0 and not is_safe_flip_source(source):
        return []
    value = extract_json(raw)
    if not isinstance(value, list):
        raise ValueError("generator response is not a JSON array")
    expected = target_label(source, operation)
    source_norm = normalize_text(source["hypothesis"])
    premise_norm = normalize_text(source["premise"])
    maximum_words = max(30, len(source["hypothesis"].split()) * 2 + 8)
    source_content = content_tokens(str(source["hypothesis"]))
    premise_content = content_tokens(str(source["premise"]))
    minimum_overlap = max(1, (len(source_content) + 1) // 2)
    if operation == "flip":
        minimum_overlap = max(1, (3 * len(source_content) + 3) // 4)
    candidates: List[str] = []
    seen = set()
    for item in value[:candidate_count]:
        if not isinstance(item, dict):
            continue
        text = item.get("hypothesis")
        declared = str(item.get("target_label", "")).strip().lower()
        if not isinstance(text, str) or declared != expected:
            continue
        text = re.sub(r"\s+", " ", text).strip()
        norm = normalize_text(text)
        if not norm or norm in seen or norm in {source_norm, premise_norm}:
            continue
        if not 3 <= len(text.split()) <= maximum_words:
            continue
        # A neutral augmentation must extend, not replace, the supported fact.
        if operation == "neutral" and source_norm not in norm:
            continue
        candidate_content = content_tokens(text)
        if (
            operation in {"inv", "flip"}
            and source_content
            and len(source_content & candidate_content) < minimum_overlap
        ):
            continue
        if operation == "inv" and (
            number_tokens(str(source["hypothesis"])) != number_tokens(text)
            or has_negation(str(source["hypothesis"])) != has_negation(text)
            or bool(
                (candidate_content - source_content)
                & (premise_content - source_content)
            )
            or len(source_content ^ candidate_content) > 1
        ):
            continue
        if operation == "flip" and not is_direct_negation(
            str(source["hypothesis"]), text
        ):
            continue
        if re.search(
            r"(?i)(premise:|hypothesis:|target_label|source_label|```|<think>|\b(?:1|one)\s+(?:are|were|have)\b|\b(?:likely|maybe|perhaps|probably|possibly)\b|\b\d+\b[^,.!?;]*\b\d+\s+(?:is|are|was|were|has|have)\b)",
            text,
        ):
            continue
        seen.add(norm)
        candidates.append(text)
    return candidates


def parse_judgments(raw: str, candidate_count: int) -> Dict[int, Dict[str, Any]]:
    value = extract_json(raw)
    if not isinstance(value, dict) or not isinstance(value.get("evaluations"), list):
        raise ValueError("judge response lacks evaluations")
    judgments: Dict[int, Dict[str, Any]] = {}
    for evaluation in value["evaluations"]:
        if not isinstance(evaluation, dict):
            continue
        index = evaluation.get("index")
        if not isinstance(index, int) or not 0 <= index < candidate_count:
            continue
        label = str(evaluation.get("label", "")).strip().lower()
        source_label = str(evaluation.get("source_label", "")).strip().lower()
        if label not in LABEL_IDS or source_label not in LABEL_IDS:
            continue
        judgments[index] = {
            "source_label": source_label,
            "label": label,
            "grammatical": evaluation.get("grammatical") is True,
            "source_equivalent": evaluation.get("source_equivalent") is True,
            "operation_valid": evaluation.get("operation_valid") is True,
            "reason": str(evaluation.get("reason", "other")),
        }
    return judgments


def choose_candidate(
    candidates: Sequence[str],
    judgments: Dict[int, Dict[str, Any]],
    source: Dict[str, Any],
    operation: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    expected = target_label(source, operation)
    for index, candidate in enumerate(candidates):
        judgment = judgments.get(index)
        if not judgment:
            continue
        if judgment["source_label"] != LABELS[int(source["label"])]:
            continue
        if operation == "flip":
            if judgment["grammatical"] and judgment["operation_valid"] and is_direct_negation(
                str(source["hypothesis"]), candidate
            ):
                return candidate, judgment
            continue
        if judgment["label"] != expected:
            continue
        if not judgment["grammatical"] or not judgment["operation_valid"]:
            continue
        if operation == "inv" and not judgment["source_equivalent"]:
            continue
        return candidate, judgment
    return None


def generate_one(
    generator_client: OpenAI,
    judge_client: OpenAI,
    source: Dict[str, Any],
    operation: str,
    generator_model: str,
    judge_model: str,
    candidate_count: int,
    retries: int,
) -> Tuple[Optional[str], Dict[str, Any]]:
    log: Dict[str, Any] = {"operation": operation, "attempts": []}
    for attempt in range(1, retries + 1):
        attempt_log: Dict[str, Any] = {"attempt": attempt}
        try:
            raw_generation = _completion(
                generator_client,
                generator_model,
                GENERATOR_SYSTEM_PROMPT,
                build_generation_prompt(source, operation, candidate_count),
                temperature=0.65,
                max_tokens=420,
            )
            candidates = parse_candidates(
                raw_generation, source, operation, candidate_count
            )
            attempt_log["candidates"] = candidates
            if not candidates:
                attempt_log["status"] = "no_valid_json_candidate"
                log["attempts"].append(attempt_log)
                continue
            raw_judgment = _completion(
                judge_client,
                judge_model,
                JUDGE_SYSTEM_PROMPT,
                build_judge_prompt(source, operation, candidates),
                temperature=0.0,
                max_tokens=520,
            )
            judgments = parse_judgments(raw_judgment, len(candidates))
            attempt_log["judgments"] = judgments
            chosen = choose_candidate(candidates, judgments, source, operation)
            if chosen:
                attempt_log["status"] = "accepted"
                attempt_log["selected"] = chosen[0]
                log["attempts"].append(attempt_log)
                log["status"] = "accepted"
                return chosen[0], log
            attempt_log["status"] = "judge_rejected"
        except Exception as exc:  # preserve error evidence and retry a clean request
            attempt_log["status"] = "error"
            attempt_log["error"] = f"{type(exc).__name__}: {exc}"
        log["attempts"].append(attempt_log)
    log["status"] = "rejected"
    return None, log


def source_record(source: Dict[str, Any], pair_id: int) -> Dict[str, Any]:
    row = dict(source)
    row.update(
        {
            "pair_id": pair_id,
            "is_syn": False,
            "weight": float(source.get("weight", 1.0)),
            "mr_id": "none",
            "mr_type": "inv",
        }
    )
    return row


def synthetic_record(
    source: Dict[str, Any], pair_id: int, operation: str, hypothesis: str
) -> Dict[str, Any]:
    spec = OPERATION_SPECS[operation]
    return {
        "premise": source["premise"],
        "hypothesis": hypothesis,
        "label": LABEL_IDS[target_label(source, operation)],
        "pair_id": pair_id,
        "is_syn": True,
        "weight": 1.0,
        "mr_id": spec["mr_id"],
        "mr_type": spec["mr_type"],
    }


def summarize(
    selected: Sequence[Dict[str, Any]],
    output_rows: Sequence[Dict[str, Any]],
    logs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    synthetic = [row for row in output_rows if row.get("is_syn")]
    attempts = Counter(log["operation"] for log in logs)
    accepted = Counter(
        log["operation"] for log in logs if log.get("status") == "accepted"
    )
    duplicate_count = 0
    seen = set()
    for row in synthetic:
        key = (normalize_text(row["premise"]), normalize_text(row["hypothesis"]))
        if key in seen:
            duplicate_count += 1
        seen.add(key)
    operation_report = {}
    for operation in OPERATION_SPECS:
        total = attempts[operation]
        operation_report[operation] = {
            "attempted": total,
            "accepted": accepted[operation],
            "acceptance_rate": round(accepted[operation] / total, 4) if total else None,
        }
    return {
        "prompt_version": PROMPT_VERSION,
        "source_count": len(selected),
        "synthetic_count": len(synthetic),
        "source_label_counts": dict(Counter(int(row["label"]) for row in selected)),
        "synthetic_label_counts": dict(
            Counter(int(row["label"]) for row in synthetic)
        ),
        "operation_results": operation_report,
        "duplicate_synthetic_rows": duplicate_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_DIR / "labeled_data.json")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_DATA_DIR / "augmented_data_v2_sample.json"
    )
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--all", action="store_true", help="process every source row")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator-model", default="gemma-3-4b-it-qat")
    parser.add_argument("--judge-model", default="gemma-3-4b-it-qat")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--operations", default="inv,flip,neutral")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"{args.output} exists; pass --overwrite or --resume explicitly"
        )
    if args.candidate_count < 1 or args.retries < 1 or args.checkpoint_every < 1:
        raise ValueError("candidate-count, retries, and checkpoint-every must be positive")
    operations = [item.strip() for item in args.operations.split(",") if item.strip()]
    unknown = set(operations) - set(OPERATION_SPECS)
    if unknown:
        raise ValueError(f"unknown operations: {sorted(unknown)}")

    source_rows = read_jsonl(args.input)
    selected = (
        list(source_rows)
        if args.all
        else stratified_sample(source_rows, args.sample_size, args.seed)
    )
    base_url = f"http://{args.host}:{args.port}/v1"
    generator_client = OpenAI(api_key="local", base_url=base_url, timeout=120.0)
    judge_client = OpenAI(api_key="local", base_url=base_url, timeout=120.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".generation_log.jsonl")
    if args.resume:
        if not args.output.exists() or not log_path.exists():
            raise FileNotFoundError("resume requires both output and generation log")
        output_rows = read_jsonl(args.output)
        with log_path.open(encoding="utf-8") as handle:
            logs = [json.loads(line) for line in handle if line.strip()]
    else:
        output_rows = []
        logs = []
        write_jsonl(args.output, output_rows)
        write_jsonl(log_path, logs)
    completed_pair_ids = {
        int(row["pair_id"]) for row in output_rows if row.get("is_syn") is not True
    }
    original_pair_ids = {
        id(row): int(row.get("pair_id", index))
        for index, row in enumerate(source_rows)
    }
    selected_pair_ids = {original_pair_ids[id(row)] for row in selected}
    unexpected = completed_pair_ids - selected_pair_ids
    if unexpected:
        raise ValueError(f"resume output contains unexpected pair_ids: {sorted(unexpected)[:5]}")
    logs = [log for log in logs if int(log.get("pair_id", -1)) in completed_pair_ids]
    resumed_count = len(completed_pair_ids)

    started = time.time()
    for selected_index, source in enumerate(selected, 1):
        pair_id = original_pair_ids[id(source)]
        if pair_id in completed_pair_ids:
            print(f"[{selected_index}/{len(selected)}] pair_id={pair_id} (resumed)", flush=True)
            continue
        output_rows.append(source_record(source, pair_id))
        for operation in operations:
            if not eligible(source, operation):
                continue
            hypothesis, generation_log = generate_one(
                generator_client,
                judge_client,
                source,
                operation,
                args.generator_model,
                args.judge_model,
                args.candidate_count,
                args.retries,
            )
            generation_log.update(
                {
                    "pair_id": pair_id,
                    "source_label": int(source["label"]),
                    "target_label": target_label(source, operation),
                }
            )
            logs.append(generation_log)
            if hypothesis:
                output_rows.append(
                    synthetic_record(source, pair_id, operation, hypothesis)
                )
        print(f"[{selected_index}/{len(selected)}] pair_id={pair_id}", flush=True)
        if selected_index % args.checkpoint_every == 0 or selected_index == len(selected):
            write_jsonl(log_path, logs)
            write_jsonl(args.output, output_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output_rows)
    log_path = args.output.with_suffix(".generation_log.jsonl")
    write_jsonl(log_path, logs)
    report = summarize(selected, output_rows, logs)
    report.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "generator_model": args.generator_model,
            "judge_model": args.judge_model,
            "seed": args.seed,
            "candidate_count": args.candidate_count,
            "retries": args.retries,
            "checkpoint_every": args.checkpoint_every,
            "resumed_source_count": resumed_count,
            "elapsed_seconds": round(time.time() - started, 2),
        }
    )
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"generation log: {log_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
