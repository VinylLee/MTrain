"""Provenance helpers shared by the SNLI augmentation generator and its backfill.

This module is deliberately dependency-free (stdlib only). ``augment_snli_all_label_mrs``
imports ``openai``, and ``openai`` is not installed in every interpreter that needs to
inspect or repair the generated artifacts, so importing the generator from the backfill
tool is not an option. Both sides therefore agree on the recipes defined here instead.

Three things live here:

* ``config_fingerprint`` — a stable digest of the generation *contract*
  (``PROMPT_VERSION`` + ``OPERATION_SPECS`` + ``COMPOSITE_STAGE_PLANS``). It answers
  "was this artifact produced by the same MR specifications the code carries today?"
* ``component_mrs_from_log`` — the single place that decides which elementary MRs a
  composite row was built from, with fail-loud validation.
* small file-hashing utilities used by both callers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Union

# Versions the *provenance/record schema*, independently of PROMPT_VERSION (which names
# the prompt contract). The prompts did not change when component_mrs was introduced, so
# bumping PROMPT_VERSION would have been misleading; this constant can advance on its own.
PROVENANCE_SCHEMA_VERSION = 1

# Per-row field naming the elementary MRs composed into a composite row.
COMPONENT_MRS_FIELD = "component_mrs"


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Streaming SHA-256 of a file, so multi-MB artifacts do not need to fit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Deterministic JSON rendering used for every digest in this module.

    Sorted keys and no insignificant whitespace, so the digest depends on content only.
    """
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def config_fingerprint(
    prompt_version: str,
    operation_specs: Mapping[str, Any],
    composite_stage_plans: Mapping[str, Any],
) -> str:
    """Digest the generation contract.

    ``PROVENANCE_SCHEMA_VERSION`` is intentionally NOT part of the payload: the legacy
    artifact predates this constant, so including it would make the fingerprint
    impossible to reconstruct for already-generated data. Keep it out.

    Note on container types: in memory ``COMPOSITE_STAGE_PLANS`` holds tuples of tuples,
    while the backfill recovers it from source via ``ast.literal_eval``, which yields
    lists of lists. ``json.dumps`` serializes both as arrays, so the two paths produce the
    same digest. Do not "fix" this by normalizing one side — it is load-bearing.
    """
    payload = {
        "prompt_version": prompt_version,
        "operation_specs": operation_specs,
        "composite_stage_plans": composite_stage_plans,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def component_mrs_from_log(
    operation: str,
    log: Mapping[str, Any],
    *,
    composite_operations: Iterable[str],
    operation_specs: Mapping[str, Any],
) -> Optional[List[str]]:
    """Return the ordered component MR ids for a composite row, else ``None``.

    Atomic operations always return ``None``. Composite operations must carry an
    ``applied_steps`` entry recording which stage plan was accepted.

    Every failure below raises rather than returning a partial value. A composite row
    whose provenance is unknown is a corrupted research artifact, and silently omitting
    the field would produce a composite population that is quietly heterogeneous — the
    exact failure mode this field exists to eliminate. The generation loop checkpoints
    every ``--checkpoint-every`` rows and can replay from the log, so failing loud costs
    at most that many source rows.
    """
    composites = set(composite_operations)
    is_composite = operation in composites
    applied = log.get("applied_steps")

    if not is_composite:
        if applied is not None:
            raise ValueError(
                f"atomic operation {operation!r} unexpectedly carries applied_steps={applied!r}"
            )
        return None

    if applied is None:
        raise ValueError(
            f"composite operation {operation!r} has no applied_steps in its generation log"
        )
    if isinstance(applied, (str, bytes)) or not isinstance(applied, Sequence):
        raise ValueError(
            f"composite operation {operation!r} has non-sequence applied_steps={applied!r}"
        )
    steps = list(applied)
    if len(steps) != 2:
        raise ValueError(
            f"composite operation {operation!r} expected 2 component MRs, got {len(steps)}: {steps!r}"
        )
    for step in steps:
        if step not in operation_specs:
            raise ValueError(
                f"composite operation {operation!r} references unknown MR {step!r}"
            )
        if step in composites:
            raise ValueError(
                f"composite operation {operation!r} references nested composite MR {step!r}"
            )
    return steps
