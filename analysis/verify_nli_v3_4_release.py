"""Independently verify full generation, frozen identity, shortcuts and human quality."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nli_v3_4 import core
from nli_v3_4.data import DATASETS, canonical_source, is_source, load_dataset, quotas, sha256, write_json
from nli_v3_4.workflow import _code_hashes, _validate_manifest
from analysis.hypothesis_only_logreg import CONFIG


def verify(manifest_path, freeze_path, generated_root, shortcut_path, quality_path, output):
    if Path(output).exists():raise FileExistsError(output)
    manifest=_validate_manifest(manifest_path);frozen=json.loads(Path(freeze_path).read_text())
    shortcut=json.loads(Path(shortcut_path).read_text());quality=json.loads(Path(quality_path).read_text())
    checks={"frozen_contract":frozen.get("contract_sha256")==core.fingerprint()
        and frozen.get("code_hashes")==_code_hashes() and frozen.get("manifest_sha256")==sha256(manifest_path),
        "development_gates":frozen.get("quality_passed") is True and frozen.get("development_shortcut_passed") is True,
        "diagnostic_config":shortcut.get("config")==CONFIG and shortcut.get("inputs_unchanged") is True,
        "human_quality":quality.get("stage")=="production" and quality.get("contract_sha256")==core.fingerprint()
        and quality.get("human_review_complete") is True and quality.get("quality_passed") is True,
        "all_four_sources":set(shortcut.get("evaluations",{}))==set(DATASETS)}
    files={}
    for dataset,info in manifest["datasets"].items():
        path=Path(generated_root)/f"{dataset}.jsonl"
        rows,sources=load_dataset(path);_,originals=load_dataset(info["input"])
        meta=json.loads(path.with_suffix(".run.json").read_text());report=json.loads(path.with_suffix(".report.json").read_text())
        follows=[r for r in rows if not is_source(r)]
        wanted=quotas(info["target_followups"])
        actual={l:sum(r["label"]==l for r in follows) for l in range(3)}
        source_by_id={r["pair_id"]:r for r in sources}
        valid_operations=all(r["mr_id"] in core.OPERATION_SPECS
            and core.eligible(source_by_id[r["pair_id"]],r["mr_id"])
            and r["label"]==core.LABEL_IDS[core.target_label(source_by_id[r["pair_id"]],r["mr_id"])]
            and (r["mr_id"] not in core.COMPOSITE_STAGE_PLANS or
                 tuple(r.get("component_mrs",())) in core.COMPOSITE_STAGE_PLANS[r["mr_id"]]) for r in follows)
        checks[f"{dataset}:inputs_preserved"]=sha256(info["input"])==info["sha256"]
        checks[f"{dataset}:sources_preserved"]=[canonical_source(r) for r in sources]==[canonical_source(r) for r in originals]
        checks[f"{dataset}:complete_quotas"]=actual==wanted and report.get("generation_complete") is True
        checks[f"{dataset}:production_identity"]=meta.get("stage")=="production" and meta.get("contract_sha256")==core.fingerprint() and meta.get("code_hashes")==_code_hashes()
        checks[f"{dataset}:no_external_nli"]=meta.get("external_nli_in_pipeline") is False
        checks[f"{dataset}:valid_operations"]=valid_operations
        checks[f"{dataset}:provenance"]=report.get("output_sha256")==sha256(path) and report.get("journal_sha256")==sha256(path.with_suffix(".generation_log.jsonl"))
        checks[f"{dataset}:diagnostic_identity"]=shortcut.get("input_sha256",{}).get(str(path.resolve()))==sha256(path)
        reviewed=quality.get("datasets",{}).get(dataset,{})
        checks[f"{dataset}:human_review_identity"]=reviewed.get("v3_4_path")==str(path.resolve()) and reviewed.get("v3_4_sha256")==sha256(path)
        if dataset in DATASETS:
            metrics=shortcut.get("evaluations",{}).get(dataset,{})
            checks[f"{dataset}:full_accuracy_le_60"]=metrics.get("full",{}).get("accuracy",1)<=.6
            checks[f"{dataset}:confirmation_accuracy_le_60"]=metrics.get("nondevelopment",{}).get("accuracy",1)<=.6
        files[dataset]={"path":str(path.resolve()),"sha256":sha256(path),"sources":len(sources),"followups":len(follows)}
    result={"overall_complete":all(checks.values()),"checks":checks,"files":files,
            "remote_push":False,"external_nli_in_production":False}
    write_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","freeze","generated-root","shortcut","quality","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();r=verify(a.manifest,a.freeze,a.generated_root,a.shortcut,a.quality,a.output)
    return 0 if r["overall_complete"] else 2


if __name__=="__main__":sys.exit(main())
